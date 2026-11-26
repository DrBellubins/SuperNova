/* hip_count.c — LD_PRELOAD shim that counts HIP launch / copy calls.
 *
 * Why: this box has no rocprofv2/rocprofv3, but Docs/design-plan.md §4.4 (3) needs
 * a real per-token kernel-launch count from the reference fork. Interposing the HIP
 * entry points libggml-hip.so calls through its PLT gives exactly that.
 *
 * Counters are printed (and reset) on SIGUSR1, and at exit:
 *   HIPCOUNTER hipLaunchKernel=123 hipGraphLaunch=1 ...
 *
 * Build:  gcc -O2 -fPIC -shared -o libhipcount.so Tools/hip_count.c -ldl
 * Use:    LD_PRELOAD=./libhipcount.so llama-server ...
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int (*fn_hipLaunchKernel)(const void *, void *, void *, void **, size_t, void *);
typedef int (*fn_hipLaunchKernelExC)(const void *, const void *, void **, size_t, void *);
typedef int (*fn_hipGraphLaunch)(void *, void *);
typedef int (*fn_hipMemcpyAsync)(void *, const void *, size_t, int, void *);
typedef int (*fn_hipMemcpy)(void *, const void *, size_t, int);

static fn_hipLaunchKernel real_launch;
static fn_hipLaunchKernelExC real_launch_exc;
static fn_hipGraphLaunch real_graph_launch;
static fn_hipMemcpyAsync real_memcpy_async;
static fn_hipMemcpy real_memcpy;

static struct { const char * name; unsigned long long n; } counters[] = {
    {"hipLaunchKernel", 0},
    {"hipLaunchKernelExC", 0},
    {"hipGraphLaunch", 0},
    {"hipMemcpyAsync", 0},
    {"hipMemcpy", 0},
};
static int in_hook; /* avoid recursion from printf/dlsym */

#define NCT (sizeof(counters) / sizeof(counters[0]))

static void bump(int i) { counters[i].n++; }

static void init(void) {
    if (real_launch) return;
    real_launch       = (fn_hipLaunchKernel)   dlsym(RTLD_NEXT, "hipLaunchKernel");
    real_launch_exc   = (fn_hipLaunchKernelExC)dlsym(RTLD_NEXT, "hipLaunchKernelExC");
    real_graph_launch = (fn_hipGraphLaunch)    dlsym(RTLD_NEXT, "hipGraphLaunch");
    real_memcpy_async = (fn_hipMemcpyAsync)    dlsym(RTLD_NEXT, "hipMemcpyAsync");
    real_memcpy       = (fn_hipMemcpy)         dlsym(RTLD_NEXT, "hipMemcpy");
}

static void report(const char * tag) {
    if (in_hook) return;
    in_hook = 1;
    fprintf(stderr, "HIPCOUNTER %s", tag);
    for (size_t i = 0; i < NCT; i++) fprintf(stderr, " %s=%llu", counters[i].name, counters[i].n);
    fprintf(stderr, "\n");
    fflush(stderr);
    for (size_t i = 0; i < NCT; i++) counters[i].n = 0;
    in_hook = 0;
}

static void on_usr1(int sig) { (void)sig; report("SIGUSR1"); }

__attribute__((constructor)) static void ctor(void) {
    init();
    signal(SIGUSR1, on_usr1);
}

__attribute__((destructor)) static void dtor(void) { report("EXIT"); }

int hipLaunchKernel(const void * f, void * gridDim, void * blockDim, void ** args,
                    size_t sharedMem, void * stream) {
    bump(0);
    return real_launch ? real_launch(f, gridDim, blockDim, args, sharedMem, stream) : 0;
}

int hipLaunchKernelExC(const void * config, const void * f, void ** args, size_t sharedMem, void * stream) {
    bump(1);
    return real_launch_exc ? real_launch_exc(config, f, args, sharedMem, stream) : 0;
}

int hipGraphLaunch(void * graphExec, void * stream) {
    bump(2);
    return real_graph_launch ? real_graph_launch(graphExec, stream) : 0;
}

int hipMemcpyAsync(void * dst, const void * src, size_t cnt, int kind, void * stream) {
    bump(3);
    return real_memcpy_async ? real_memcpy_async(dst, src, cnt, kind, stream) : 0;
}

int hipMemcpy(void * dst, const void * src, size_t cnt, int kind) {
    bump(4);
    return real_memcpy ? real_memcpy(dst, src, cnt, kind) : 0;
}
