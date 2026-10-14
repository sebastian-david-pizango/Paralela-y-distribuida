// =====================================================================
//  ANALIZADOR PARALELO DE VENTAS
//
//  SITUACION: una cadena de tiendas registra millones de ventas al dia.
//  Gerencia necesita, lo mas rapido posible:
//     - total vendido, venta minima y maxima, ventas sospechosas
//     - ventas y cantidad por categoria (histograma)
//     - puntaje de riesgo de fraude de CADA venta (calculo intensivo)
//     - las 10 ventas mas grandes (Top-K)
//
//  ALGORITMOS PARALELOS USADOS
//     1. Generacion de datos   -> paralelismo de datos (bloques independientes)
//     2. Reduccion paralela    -> total, minimo, maximo, sospechosas, riesgo
//     3. Histograma paralelo   -> privatizacion + fusion
//     4. Top-K paralelo        -> heap local por hilo + fusion
//
//  DOS IMPLEMENTACIONES DEL MISMO ANALISIS (se comparan entre si):
//     - std::thread  : control explicito de hilos, reparto dinamico, mutex, atomic
//     - OpenMP       : paradigma de alto nivel (se activa compilando con -fopenmp)
//
//  Compilar: g++ -O2 -std=c++17 -pthread -fopenmp ventas_paralelo.cpp -o ventas
//            (sin -fopenmp tambien compila; solo se omite la version OpenMP)
//  Usar:     ./ventas [millones=20] [max_hilos=auto] [segundos_sostenido=0]
//                     [--forzar]        permitir mas hilos que nucleos logicos
//                     [--demo-carrera]  mostrar una version SIN sincronizacion
// =====================================================================

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <queue>
#include <string>
#include <thread>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
using Reloj = chrono::steady_clock;

constexpr int      CATEGORIAS  = 8;
constexpr int      TOP_K       = 10;
constexpr size_t   BLOQUE      = 1 << 16;     // 65 536 registros = 256 KB de montos (cabe en cache L2)
constexpr uint32_t UMBRAL      = 500000;      // 5 000.00 en centavos -> venta sospechosa
constexpr int      RONDAS      = 24;          // costo del modelo de riesgo (calculo por registro)
constexpr uint32_t RIESGO_ALTO = 950;         // puntaje (0..999) a partir del cual se marca alto riesgo
constexpr uint64_t SEMILLA     = 20261012;
const char* NOMBRES[CATEGORIAS] = {"Alimentos", "Electro", "Ropa", "Hogar",
                                   "Juguetes", "Libros", "Farmacia", "Deportes"};

// ---------------------------------------------------------------------
//  Datos: estructura de arreglos (SoA). Cada hilo recorre memoria
//  contigua -> buen uso de cache y prefetch del hardware.
//  Los montos son enteros (centavos): las sumas son EXACTAS y el
//  resultado paralelo se puede comparar bit a bit con el secuencial.
// ---------------------------------------------------------------------
struct Datos {
    vector<uint32_t> monto;
    vector<uint8_t>  categoria;
    explicit Datos(size_t n) : monto(n), categoria(n) {}
    size_t n() const { return monto.size(); }
    size_t bloques() const { return (n() + BLOQUE - 1) / BLOQUE; }
};

static inline uint64_t splitmix64(uint64_t& x) {
    uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// El generador se siembra POR BLOQUE (no por hilo): los datos son identicos
// sin importar cuantos hilos se usen. Cada bloque escribe una zona distinta
// del arreglo, asi que no hay condiciones de carrera.
void generarBloque(Datos& d, size_t b) {
    size_t ini = b * BLOQUE, fin = min(d.n(), ini + BLOQUE);
    uint64_t s = SEMILLA ^ (b * 0xD1B54A32D192ED03ULL);
    for (size_t i = ini; i < fin; ++i) {
        uint64_t r = splitmix64(s);
        uint32_t m = 100 + (uint32_t)(r % 50000);        // $1.00 .. $500.99
        if ((r >> 20) % 1000 == 0) m *= 20;               // 0.1 % de ventas enormes
        d.monto[i] = m;
        d.categoria[i] = (uint8_t)((r >> 48) & (CATEGORIAS - 1));
    }
}

// ---------------------------------------------------------------------
//  Modelo de riesgo (simulado): calculo ENTERO y determinista, con costo
//  alto por registro. Hace que el programa dependa del CPU y no solo del
//  ancho de banda de memoria, que es lo que permite escalar con los nucleos.
// ---------------------------------------------------------------------
static inline uint32_t puntajeRiesgo(uint32_t monto, uint8_t cat, uint64_t idx) {
    uint64_t x = ((uint64_t)monto << 8) ^ cat ^ (idx * 0x9E3779B97F4A7C15ULL);
    for (int k = 0; k < RONDAS; ++k) { x ^= x >> 29; x *= 0xBF58476D1CE4E5B9ULL; x ^= x >> 32; }
    return (uint32_t)(x % 800) + (monto >= UMBRAL ? 200u : 0u);   // 0..999
}

// ---------------------------------------------------------------------
//  Gestion de hilos: lanza N hilos y SIEMPRE los une (join), incluso si
//  falla la creacion de alguno (no quedan hilos huerfanos).
// ---------------------------------------------------------------------
template <class F>
void lanzar(int hilos, F&& f) {
    vector<thread> v;
    v.reserve(hilos);
    try {
        for (int i = 0; i < hilos; ++i) v.emplace_back(f, i);
    } catch (...) {
        for (auto& t : v) t.join();
        throw;
    }
    for (auto& t : v) t.join();
}

void generarSecuencial(Datos& d) {
    for (size_t b = 0; b < d.bloques(); ++b) generarBloque(d, b);
}

// Reparto DINAMICO de bloques: cada hilo toma el siguiente bloque libre con
// un unico fetch_add. Si un hilo va mas lento, los demas absorben su trabajo.
void generarParalelo(Datos& d, int hilos) {
    alignas(64) atomic<size_t> siguiente{0};
    size_t nb = d.bloques();
    lanzar(hilos, [&](int) {
        for (;;) {
            size_t b = siguiente.fetch_add(1, memory_order_relaxed);
            if (b >= nb) break;
            generarBloque(d, b);
        }
    });
}

// ---------------------------------------------------------------------
//  Resultado del analisis
// ---------------------------------------------------------------------
struct Venta { uint32_t valor; uint64_t idx; };
// "a es mejor que b": mayor valor; en empate, menor indice (orden total => resultado determinista)
struct MejorQue {
    bool operator()(const Venta& a, const Venta& b) const {
        return a.valor > b.valor || (a.valor == b.valor && a.idx < b.idx);
    }
};
using MinHeap = priority_queue<Venta, vector<Venta>, MejorQue>;   // en el tope queda la PEOR de las K

struct Resultado {
    uint64_t total = 0, sospechosas = 0, altoRiesgo = 0;
    uint32_t minimo = UINT32_MAX, maximo = 0;
    uint64_t cant[CATEGORIAS] = {}, suma[CATEGORIAS] = {}, riesgo[CATEGORIAS] = {};
    MinHeap  top;

    inline void ofrecer(Venta v) {
        if (top.size() < TOP_K) { top.push(v); return; }
        if (v.valor < top.top().valor) return;                    // descarte rapido
        if (MejorQue()(v, top.top())) { top.pop(); top.push(v); }
    }
    void fusionar(Resultado& o) {                                 // combina dos resultados parciales
        total += o.total; sospechosas += o.sospechosas; altoRiesgo += o.altoRiesgo;
        minimo = min(minimo, o.minimo); maximo = max(maximo, o.maximo);
        for (int c = 0; c < CATEGORIAS; ++c) { cant[c] += o.cant[c]; suma[c] += o.suma[c]; riesgo[c] += o.riesgo[c]; }
        while (!o.top.empty()) { ofrecer(o.top.top()); o.top.pop(); }
    }
};

void procesarRango(const Datos& d, size_t ini, size_t fin, Resultado& r) {
    for (size_t i = ini; i < fin; ++i) {
        uint32_t m = d.monto[i];
        uint8_t  c = d.categoria[i];
        r.total += m;
        if (m < r.minimo) r.minimo = m;
        if (m > r.maximo) r.maximo = m;
        if (m >= UMBRAL) ++r.sospechosas;
        ++r.cant[c];
        r.suma[c] += m;
        uint32_t rs = puntajeRiesgo(m, c, i);
        r.riesgo[c] += rs;
        if (rs >= RIESGO_ALTO) ++r.altoRiesgo;
        r.ofrecer({m, i});
    }
}

Resultado analizarSecuencial(const Datos& d) {
    Resultado r;
    procesarRango(d, 0, d.n(), r);
    return r;
}

// Version std::thread: cada hilo acumula en estructuras PRIVADAS (sin compartir
// memoria mientras trabaja) y solo al final entra UNA vez a la seccion critica.
Resultado analizarParalelo(const Datos& d, int hilos) {
    Resultado global;
    mutex cerrojo;
    alignas(64) atomic<size_t> siguiente{0};
    size_t nb = d.bloques();
    lanzar(hilos, [&](int) {
        Resultado local;
        for (;;) {
            size_t b = siguiente.fetch_add(1, memory_order_relaxed);
            if (b >= nb) break;
            size_t ini = b * BLOQUE;
            procesarRango(d, ini, min(d.n(), ini + BLOQUE), local);
        }
        lock_guard<mutex> lk(cerrojo);                            // seccion critica minima
        global.fusionar(local);
    });
    return global;
}

#ifdef _OPENMP
// Version OpenMP: mismo algoritmo (privatizacion + fusion) con el paradigma de
// alto nivel. schedule(dynamic) = reparto dinamico; critical = fusion segura.
Resultado analizarOpenMP(const Datos& d, int hilos) {
    Resultado global;
    long nb = (long)d.bloques();
    #pragma omp parallel num_threads(hilos)
    {
        Resultado local;
        #pragma omp for schedule(dynamic, 1) nowait
        for (long b = 0; b < nb; ++b) {
            size_t ini = (size_t)b * BLOQUE;
            procesarRango(d, ini, min(d.n(), ini + BLOQUE), local);
        }
        #pragma omp critical(fusion)
        global.fusionar(local);
    }
    return global;
}
#endif

vector<Venta> topOrdenado(Resultado r) {                          // copia por valor a proposito
    vector<Venta> v;
    while (!r.top.empty()) { v.push_back(r.top.top()); r.top.pop(); }
    sort(v.begin(), v.end(), MejorQue());
    return v;
}

bool iguales(const Resultado& a, const Resultado& b) {
    if (a.total != b.total || a.sospechosas != b.sospechosas || a.altoRiesgo != b.altoRiesgo ||
        a.minimo != b.minimo || a.maximo != b.maximo) return false;
    if (memcmp(a.cant, b.cant, sizeof a.cant) || memcmp(a.suma, b.suma, sizeof a.suma) ||
        memcmp(a.riesgo, b.riesgo, sizeof a.riesgo)) return false;
    auto ta = topOrdenado(a), tb = topOrdenado(b);
    if (ta.size() != tb.size()) return false;
    for (size_t i = 0; i < ta.size(); ++i)
        if (ta[i].valor != tb[i].valor || ta[i].idx != tb[i].idx) return false;
    return true;
}

// ---------------------------------------------------------------------
//  EXPERIMENTO DE SINCRONIZACION: el mismo histograma con varias estrategias
// ---------------------------------------------------------------------
struct Hist {
    uint64_t cant[CATEGORIAS] = {}, suma[CATEGORIAS] = {};
    bool operator==(const Hist& o) const {
        return !memcmp(cant, o.cant, sizeof cant) && !memcmp(suma, o.suma, sizeof suma);
    }
    uint64_t totalCant() const { uint64_t s = 0; for (auto v : cant) s += v; return s; }
};

Hist histSecuencial(const Datos& d, size_t m) {
    Hist h;
    for (size_t i = 0; i < m; ++i) { ++h.cant[d.categoria[i]]; h.suma[d.categoria[i]] += d.monto[i]; }
    return h;
}

// X) SIN sincronizacion (DEMOSTRACION, solo con --demo-carrera): varios hilos
//    modifican los mismos contadores a la vez -> condicion de carrera (comportamiento
//    indefinido). Se pierden actualizaciones y el resultado es incorrecto.
Hist histInseguro(const Datos& d, size_t m, int hilos) {
    Hist h;
    lanzar(hilos, [&](int t) {
        for (size_t i = m * t / hilos; i < m * (t + 1) / hilos; ++i) {
            ++h.cant[d.categoria[i]]; h.suma[d.categoria[i]] += d.monto[i];
        }
    });
    return h;
}

// A) Un mutex que se toma por CADA elemento: correcto, pero los hilos se serializan.
Hist histMutex(const Datos& d, size_t m, int hilos) {
    Hist h; mutex cerrojo;
    lanzar(hilos, [&](int t) {
        for (size_t i = m * t / hilos; i < m * (t + 1) / hilos; ++i) {
            lock_guard<mutex> lk(cerrojo);
            ++h.cant[d.categoria[i]]; h.suma[d.categoria[i]] += d.monto[i];
        }
    });
    return h;
}

// B) Contadores atomicos compartidos: sin cerrojo, pero todos los hilos pelean
//    por las mismas lineas de cache (contencion + falso compartido).
Hist histAtomico(const Datos& d, size_t m, int hilos) {
    atomic<uint64_t> cant[CATEGORIAS] = {}, suma[CATEGORIAS] = {};
    lanzar(hilos, [&](int t) {
        for (size_t i = m * t / hilos; i < m * (t + 1) / hilos; ++i) {
            cant[d.categoria[i]].fetch_add(1, memory_order_relaxed);
            suma[d.categoria[i]].fetch_add(d.monto[i], memory_order_relaxed);
        }
    });
    Hist h;
    for (int c = 0; c < CATEGORIAS; ++c) { h.cant[c] = cant[c]; h.suma[c] = suma[c]; }
    return h;
}

// C) Privatizacion: cada hilo cuenta en su copia local y se fusiona UNA sola vez.
Hist histLocal(const Datos& d, size_t m, int hilos) {
    Hist h; mutex cerrojo;
    lanzar(hilos, [&](int t) {
        Hist loc;
        for (size_t i = m * t / hilos; i < m * (t + 1) / hilos; ++i) {
            ++loc.cant[d.categoria[i]]; loc.suma[d.categoria[i]] += d.monto[i];
        }
        lock_guard<mutex> lk(cerrojo);
        for (int c = 0; c < CATEGORIAS; ++c) { h.cant[c] += loc.cant[c]; h.suma[c] += loc.suma[c]; }
    });
    return h;
}

// ---------------------------------------------------------------------
//  Utilidades de medicion
// ---------------------------------------------------------------------
double ms(Reloj::time_point a, Reloj::time_point b) { return chrono::duration<double, milli>(b - a).count(); }

template <class F>
double medir(F&& f, int reps = 3) {                               // mejor de varias corridas
    double mejor = 1e300;
    for (int i = 0; i < reps; ++i) { auto t0 = Reloj::now(); f(); mejor = min(mejor, ms(t0, Reloj::now())); }
    return mejor;
}

int main(int argc, char* argv[]) {
    int hw = (int)thread::hardware_concurrency(); if (hw <= 0) hw = 4;
    double millones = 20.0;
    int maxHilos = hw, sostenido = 0, pos = 0;
    bool forzar = false, demoCarrera = false;
    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        if (a == "--forzar") forzar = true;
        else if (a == "--demo-carrera") demoCarrera = true;
        else {
            if (pos == 0) millones = atof(argv[i]);
            else if (pos == 1) maxHilos = atoi(argv[i]);
            else if (pos == 2) sostenido = atoi(argv[i]);
            ++pos;
        }
    }
    if (millones < 0.1) millones = 0.1;
    if (maxHilos < 1) maxHilos = 1;
    size_t N = (size_t)(millones * 1e6);

    printf("=== ANALIZADOR PARALELO DE VENTAS ===\n");
    if (maxHilos > hw && !forzar) {
        printf("AVISO: pediste %d hilos pero el sistema tiene %d nucleos logicos; se limita a %d "
               "(usa --forzar para ignorar el limite).\n", maxHilos, hw, hw);
        maxHilos = hw;
    }
    printf("Registros: %zu | Nucleos logicos: %d | Hilos a usar (max): %d | OpenMP: %s\n\n", N, hw, maxHilos,
#ifdef _OPENMP
           "si"
#else
           "no (compila con -fopenmp)"
#endif
    );

    try {
        // ---------- 1. Generacion de datos ----------
        printf("[1] Generacion de datos\n");
        Datos datos(N), ref(N);
        double tGenSeq = medir([&] { generarSecuencial(ref); }, 1);
        double tGenPar = medir([&] { generarParalelo(datos, maxHilos); }, 1);
        bool genOk = datos.monto == ref.monto && datos.categoria == ref.categoria;
        printf("    secuencial: %8.1f ms | paralelo (%d hilos): %8.1f ms | speedup %.2fx | identicos: %s\n\n",
               tGenSeq, maxHilos, tGenPar, tGenSeq / tGenPar, genOk ? "SI" : "NO");
        { Datos vacio(0); ref.monto.swap(vacio.monto); ref.categoria.swap(vacio.categoria); }  // liberar memoria

        // ---------- 2. Analisis completo: escalabilidad ----------
        printf("[2] Analisis completo (reduccion + histograma + riesgo + Top-%d): escalabilidad\n", TOP_K);
        Resultado base;
        double tSeq = medir([&] { base = analizarSecuencial(datos); });
        printf("    %-16s %12s %10s %12s %10s\n", "version/hilos", "tiempo (ms)", "speedup", "eficiencia", "correcto");
        printf("    %-16s %12.2f %10s %12s %10s\n", "secuencial (1)", tSeq, "1.00x", "100%", "ref");
        vector<int> lista;
        for (int h = 2; h < maxHilos; h *= 2) lista.push_back(h);
        if (maxHilos >= 2) lista.push_back(maxHilos);
        for (int h : lista) {
            Resultado par;
            double t = medir([&] { par = analizarParalelo(datos, h); });
            double sp = tSeq / t;
            char et[32]; snprintf(et, sizeof et, "std::thread %d", h);
            printf("    %-16s %12.2f %9.2fx %11.0f%% %10s\n", et, t, sp, 100.0 * sp / h, iguales(base, par) ? "SI" : "NO");
        }
#ifdef _OPENMP
        {
            Resultado par;
            double t = medir([&] { par = analizarOpenMP(datos, maxHilos); });
            double sp = tSeq / t;
            char et[32]; snprintf(et, sizeof et, "OpenMP %d", maxHilos);
            printf("    %-16s %12.2f %9.2fx %11.0f%% %10s\n", et, t, sp, 100.0 * sp / maxHilos, iguales(base, par) ? "SI" : "NO");
        }
#endif
        printf("\n");

        // ---------- 3. Sincronizacion de memoria ----------
        size_t M = min(N, (size_t)2000000);
        int hs = max(2, maxHilos);
        printf("[3] Sincronizacion: mismo histograma con %d hilos sobre %zu registros\n", hs, M);
        Hist refH = histSecuencial(datos, M);
        Hist hm, ha, hl;
        double tM = medir([&] { hm = histMutex(datos, M, hs); }, 1);
        double tA = medir([&] { ha = histAtomico(datos, M, hs); }, 1);
        double tL = medir([&] { hl = histLocal(datos, M, hs); }, 1);
        printf("    %-38s %10.2f ms  correcto: %s\n", "A) mutex por elemento", tM, hm == refH ? "SI" : "NO");
        printf("    %-38s %10.2f ms  correcto: %s\n", "B) atomicos compartidos", tA, ha == refH ? "SI" : "NO");
        printf("    %-38s %10.2f ms  correcto: %s\n", "C) privatizacion + fusion (usada)", tL, hl == refH ? "SI" : "NO");
        if (demoCarrera) {
            Hist hx; double tX = medir([&] { hx = histInseguro(datos, M, hs); }, 1);
            uint64_t perdidas = refH.totalCant() - min(refH.totalCant(), hx.totalCant());
            printf("    %-38s %10.2f ms  correcto: %s  (actualizaciones perdidas: %llu)\n",
                   "X) SIN sincronizacion (demo)", tX, hx == refH ? "SI (por suerte)" : "NO", (unsigned long long)perdidas);
        }
        printf("\n");

        // ---------- 4. Reporte de negocio ----------
        Resultado r = analizarParalelo(datos, maxHilos);
        printf("[4] Reporte para gerencia\n");
        printf("    Total vendido     : $%.2f\n", r.total / 100.0);
        printf("    Ticket promedio   : $%.2f\n", r.total / 100.0 / N);
        printf("    Venta min / max   : $%.2f / $%.2f\n", r.minimo / 100.0, r.maximo / 100.0);
        printf("    Ventas sospechosas: %llu (monto >= $%.2f)\n", (unsigned long long)r.sospechosas, UMBRAL / 100.0);
        printf("    Alto riesgo       : %llu ventas (puntaje >= %u)\n", (unsigned long long)r.altoRiesgo, RIESGO_ALTO);
        printf("    Por categoria:\n");
        for (int c = 0; c < CATEGORIAS; ++c)
            printf("      %-10s %10llu ventas   $%-16.2f riesgo prom. %.1f\n", NOMBRES[c],
                   (unsigned long long)r.cant[c], r.suma[c] / 100.0, (double)r.riesgo[c] / r.cant[c]);
        printf("    Top %d ventas:\n", TOP_K);
        int k = 1;
        for (auto& v : topOrdenado(r))
            printf("      #%-2d registro %-10llu $%.2f  (%s)\n", k++, (unsigned long long)v.idx, v.valor / 100.0, NOMBRES[datos.categoria[v.idx]]);

        // ---------- 5. Carga sostenida (opcional) ----------
        if (sostenido > 0) {
            printf("\n[5] Carga sostenida durante %d s con %d hilos...\n", sostenido, maxHilos);
            auto limite = Reloj::now() + chrono::seconds(sostenido);
            long vueltas = 0; bool todoOk = true;
            while (Reloj::now() < limite) { todoOk &= iguales(base, analizarParalelo(datos, maxHilos)); ++vueltas; }
            printf("    Analisis completos: %ld | todos correctos: %s\n", vueltas, todoOk ? "SI" : "NO");
        }
    } catch (const bad_alloc&) {
        fprintf(stderr, "Memoria insuficiente: usa menos millones de registros.\n");
        return 1;
    } catch (const exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
    return 0;
}
