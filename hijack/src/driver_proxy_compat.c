/* Modified for Lilt (2026): local proxy and HP/BE scheduling. See LICENSE. */
#include "../include/lilt_config.h"
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "../include/cuda-helper.h"
#include "../include/hijack.h"
#include "../include/graph_scheduler.h"

#define LILT_GETPROC_PTSZ_FLAG (1ull << 1)

typedef CUresult (*lilt_getproc_v11030_fn)(const char *, void **, int,
                                           cuuint64_t);
typedef CUresult (*lilt_getproc_v12000_fn)(
    const char *, void **, int, cuuint64_t,
    CUdriverProcAddressQueryResult *);
typedef CUresult (*lilt_launch_ex_fn)(const CUlaunchConfig *, CUfunction,
                                     void **, void **);
typedef CUresult (*lilt_occupancy_clusters_fn)(int *, CUfunction,
                                               const CUlaunchConfig *);
typedef CUresult (*lilt_library_load_data_fn)(
    CUlibrary *, const void *, CUjit_option *, void **, unsigned int,
    CUlibraryOption *, void **, unsigned int);
typedef CUresult (*lilt_library_load_file_fn)(
    CUlibrary *, const char *, CUjit_option *, void **, unsigned int,
    CUlibraryOption *, void **, unsigned int);
typedef CUresult (*lilt_library_unload_fn)(CUlibrary);
typedef CUresult (*lilt_library_get_kernel_fn)(CUkernel *, CUlibrary,
                                               const char *);
typedef CUresult (*lilt_kernel_get_attribute_fn)(int *, CUfunction_attribute,
                                                 CUkernel, CUdevice);
typedef CUresult (*lilt_kernel_set_attribute_fn)(CUfunction_attribute, int,
                                                 CUkernel, CUdevice);

#define LILT_REAL_FORWARD_OR_NOT_SUPPORTED(symbol, type) \
  type real_function = (type)cuda_real_symbol(symbol);   \
  if (real_function == NULL) {                           \
    return CUDA_ERROR_NOT_SUPPORTED;                     \
  }

CUresult cuLaunchKernel(CUfunction, unsigned int, unsigned int, unsigned int,
                        unsigned int, unsigned int, unsigned int, unsigned int,
                        CUstream, void **, void **);
CUresult cuLaunchKernel_ptsz(
    CUfunction, unsigned int, unsigned int, unsigned int, unsigned int,
    unsigned int, unsigned int, unsigned int, CUstream, void **, void **);
CUresult cuLaunchCooperativeKernel(
    CUfunction, unsigned int, unsigned int, unsigned int, unsigned int,
    unsigned int, unsigned int, unsigned int, CUstream, void **);
CUresult cuLaunchCooperativeKernel_ptsz(
    CUfunction, unsigned int, unsigned int, unsigned int, unsigned int,
    unsigned int, unsigned int, unsigned int, CUstream, void **);

CUresult cuLaunchKernelEx(const CUlaunchConfig *, CUfunction, void **, void **);
CUresult cuLaunchKernelEx_ptsz(const CUlaunchConfig *, CUfunction, void **,
                              void **);
CUresult cuGetProcAddress(const char *, void **, int, cuuint64_t);
CUresult cuGetProcAddress_v2(const char *, void **, int, cuuint64_t,
                            CUdriverProcAddressQueryResult *);
CUresult cuGraphLaunch(CUgraphExec, CUstream);
CUresult cuGraphLaunch_ptsz(CUgraphExec, CUstream);
CUresult cuGraphExecDestroy(CUgraphExec);
CUresult cuGraphInstantiate(CUgraphExec *, CUgraph, CUgraphNode *, char *,
                            size_t);
CUresult cuGraphInstantiate_v2(CUgraphExec *, CUgraph, CUgraphNode *, char *,
                               size_t);
CUresult cuGraphExecUpdate(CUgraphExec, CUgraph, CUgraphNode *,
                           CUgraphExecUpdateResult *);

static unsigned long long g_getproc_queries;
static unsigned long long g_getproc_replacements;
static unsigned long long g_dlsym_cuda_queries;
static unsigned long long g_dlsym_wrapper_replacements;
static unsigned long long g_dlsym_real_forwards;

static void *direct_cuda_wrapper(const char *symbol) {
  if (strcmp(symbol, "cuGetProcAddress") == 0) {
    return (void *)&cuGetProcAddress;
  }
  if (strcmp(symbol, "cuGetProcAddress_v2") == 0) {
    return (void *)&cuGetProcAddress_v2;
  }
  if (strcmp(symbol, "cuLaunchKernel") == 0) {
    return (void *)&cuLaunchKernel;
  }
  if (strcmp(symbol, "cuLaunchKernel_ptsz") == 0) {
    return (void *)&cuLaunchKernel_ptsz;
  }
  if (strcmp(symbol, "cuLaunchKernelEx") == 0) {
    return (void *)&cuLaunchKernelEx;
  }
  if (strcmp(symbol, "cuLaunchKernelEx_ptsz") == 0) {
    return (void *)&cuLaunchKernelEx_ptsz;
  }
  if (strcmp(symbol, "cuLaunchCooperativeKernel") == 0) {
    return (void *)&cuLaunchCooperativeKernel;
  }
  if (strcmp(symbol, "cuLaunchCooperativeKernel_ptsz") == 0) {
    return (void *)&cuLaunchCooperativeKernel_ptsz;
  }
  if (strcmp(symbol, "cuGraphLaunch") == 0) {
    return (void *)&cuGraphLaunch;
  }
  if (strcmp(symbol, "cuGraphLaunch_ptsz") == 0) {
    return (void *)&cuGraphLaunch_ptsz;
  }
  if (strcmp(symbol, "cuGraphExecDestroy") == 0) {
    return (void *)&cuGraphExecDestroy;
  }
  if (strcmp(symbol, "cuGraphInstantiate") == 0) {
    return (void *)&cuGraphInstantiate;
  }
  if (strcmp(symbol, "cuGraphInstantiate_v2") == 0) {
    return (void *)&cuGraphInstantiate_v2;
  }
  if (strcmp(symbol, "cuGraphInstantiateWithFlags") == 0) {
    return (void *)&cuGraphInstantiateWithFlags;
  }
  if (strcmp(symbol, "cuGraphInstantiateWithParams") == 0) {
    return (void *)&cuGraphInstantiateWithParams;
  }
  if (strcmp(symbol, "cuGraphInstantiateWithParams_ptsz") == 0) {
    return (void *)&cuGraphInstantiateWithParams_ptsz;
  }
  if (strcmp(symbol, "cuGraphNodeSetEnabled") == 0) {
    return (void *)&cuGraphNodeSetEnabled;
  }
  if (strcmp(symbol, "cuGraphExecUpdate") == 0) {
    return (void *)&cuGraphExecUpdate;
  }
  if (strcmp(symbol, "cuGraphExecUpdate_v2") == 0) {
    return (void *)&cuGraphExecUpdate_v2;
  }
  return NULL;
}

static int is_cuda_driver_symbol(const char *symbol) {
  return symbol[0] == 'c' && symbol[1] == 'u' && symbol[2] >= 'A' &&
         symbol[2] <= 'Z';
}

void *dlsym(void *handle, const char *symbol) {
  void *wrapper = direct_cuda_wrapper(symbol);
  void *resolved;

  /*
   * Profilers and other interposers use RTLD_NEXT to build a wrapper chain.
   * Replacing that lookup with our own wrapper can send the caller back up the
   * chain and break tools such as Nsight Systems. The application-facing
   * RTLD_DEFAULT/explicit-handle paths remain covered below.
   */
  if (handle == RTLD_NEXT) {
    return lilt_system_dlsym(handle, symbol);
  }

  if (is_cuda_driver_symbol(symbol)) {
    __sync_fetch_and_add(&g_dlsym_cuda_queries, 1);
  }
  if (wrapper != NULL) {
    __sync_fetch_and_add(&g_dlsym_wrapper_replacements, 1);
    return wrapper;
  }
  if (is_cuda_driver_symbol(symbol)) {
    resolved = cuda_real_symbol(symbol);
    if (resolved != NULL) {
      __sync_fetch_and_add(&g_dlsym_real_forwards, 1);
      return resolved;
    }
  }
  return lilt_system_dlsym(handle, symbol);
}

static int symbol_is(const char *symbol, const char *base,
                     const char *ptsz) {
  return strcmp(symbol, base) == 0 || strcmp(symbol, ptsz) == 0;
}

static void replace_getproc_pointer(const char *symbol, void **pfn,
                                    int cuda_version, cuuint64_t flags) {
  void *returned = *pfn;
  void *real_legacy;
  void *real_ptsz;
  void *replacement = NULL;

  if (symbol_is(symbol, "cuLaunchKernel", "cuLaunchKernel_ptsz")) {
    real_legacy = cuda_real_symbol("cuLaunchKernel");
    real_ptsz = cuda_real_symbol("cuLaunchKernel_ptsz");
    if (returned == real_ptsz || strcmp(symbol, "cuLaunchKernel_ptsz") == 0 ||
        (returned != real_legacy && (flags & LILT_GETPROC_PTSZ_FLAG) != 0)) {
      replacement = (void *)&cuLaunchKernel_ptsz;
    } else {
      replacement = (void *)&cuLaunchKernel;
    }
  } else if (symbol_is(symbol, "cuLaunchKernelEx",
                       "cuLaunchKernelEx_ptsz")) {
    real_legacy = cuda_real_symbol("cuLaunchKernelEx");
    real_ptsz = cuda_real_symbol("cuLaunchKernelEx_ptsz");
    if (returned == real_ptsz ||
        strcmp(symbol, "cuLaunchKernelEx_ptsz") == 0 ||
        (returned != real_legacy && (flags & LILT_GETPROC_PTSZ_FLAG) != 0)) {
      replacement = (void *)&cuLaunchKernelEx_ptsz;
    } else {
      replacement = (void *)&cuLaunchKernelEx;
    }
  } else if (symbol_is(symbol, "cuLaunchCooperativeKernel",
                       "cuLaunchCooperativeKernel_ptsz")) {
    real_legacy = cuda_real_symbol("cuLaunchCooperativeKernel");
    real_ptsz = cuda_real_symbol("cuLaunchCooperativeKernel_ptsz");
    if (returned == real_ptsz ||
        strcmp(symbol, "cuLaunchCooperativeKernel_ptsz") == 0 ||
        (returned != real_legacy && (flags & LILT_GETPROC_PTSZ_FLAG) != 0)) {
      replacement = (void *)&cuLaunchCooperativeKernel_ptsz;
    } else {
      replacement = (void *)&cuLaunchCooperativeKernel;
    }
  } else if (symbol_is(symbol, "cuGraphLaunch",
                       "cuGraphLaunch_ptsz")) {
    real_legacy = cuda_real_symbol("cuGraphLaunch");
    real_ptsz = cuda_real_symbol("cuGraphLaunch_ptsz");
    if (returned == real_ptsz || strcmp(symbol, "cuGraphLaunch_ptsz") == 0 ||
        (returned != real_legacy && (flags & LILT_GETPROC_PTSZ_FLAG) != 0)) {
      replacement = (void *)&cuGraphLaunch_ptsz;
    } else {
      replacement = (void *)&cuGraphLaunch;
    }
  } else if (strcmp(symbol, "cuGraphInstantiateWithFlags") == 0) {
    replacement = (void *)&cuGraphInstantiateWithFlags;
  } else if (symbol_is(symbol, "cuGraphInstantiateWithParams",
                       "cuGraphInstantiateWithParams_ptsz")) {
    real_legacy = cuda_real_symbol("cuGraphInstantiateWithParams");
    real_ptsz = cuda_real_symbol("cuGraphInstantiateWithParams_ptsz");
    if (returned == real_ptsz ||
        strcmp(symbol, "cuGraphInstantiateWithParams_ptsz") == 0 ||
        (returned != real_legacy && (flags & LILT_GETPROC_PTSZ_FLAG) != 0)) {
      replacement = (void *)&cuGraphInstantiateWithParams_ptsz;
    } else {
      replacement = (void *)&cuGraphInstantiateWithParams;
    }
  } else if (strcmp(symbol, "cuGraphInstantiate") == 0 ||
             strcmp(symbol, "cuGraphInstantiate_v2") == 0) {
    void *real_flags = cuda_real_symbol("cuGraphInstantiateWithFlags");
    void *real_v2 = cuda_real_symbol("cuGraphInstantiate_v2");
    if (returned == real_flags) {
      replacement = (void *)&cuGraphInstantiateWithFlags;
    } else if (returned == real_v2 ||
               strcmp(symbol, "cuGraphInstantiate_v2") == 0) {
      replacement = (void *)&cuGraphInstantiate_v2;
    } else {
      replacement = (void *)&cuGraphInstantiate;
    }
  } else if (strcmp(symbol, "cuGraphExecDestroy") == 0) {
    replacement = (void *)&cuGraphExecDestroy;
  } else if (strcmp(symbol, "cuGraphNodeSetEnabled") == 0) {
    replacement = (void *)&cuGraphNodeSetEnabled;
  } else if (strcmp(symbol, "cuGraphExecUpdate_v2") == 0) {
    replacement = (void *)&cuGraphExecUpdate_v2;
  } else if (strcmp(symbol, "cuGraphExecUpdate") == 0) {
    void *real_v2 = cuda_real_symbol("cuGraphExecUpdate_v2");
    replacement = returned == real_v2 ? (void *)&cuGraphExecUpdate_v2
                                      : (void *)&cuGraphExecUpdate;
  } else if (strcmp(symbol, "cuGetProcAddress_v2") == 0) {
    replacement = (void *)&cuGetProcAddress_v2;
  } else if (strcmp(symbol, "cuGetProcAddress") == 0) {
    real_legacy = cuda_real_symbol("cuGetProcAddress");
    real_ptsz = cuda_real_symbol("cuGetProcAddress_v2");
    if (returned == real_ptsz ||
        (returned != real_legacy && cuda_version >= 12000)) {
      replacement = (void *)&cuGetProcAddress_v2;
    } else {
      replacement = (void *)&cuGetProcAddress;
    }
  }

  if (replacement != NULL) {
    *pfn = replacement;
    __sync_fetch_and_add(&g_getproc_replacements, 1);
  }
}

CUresult cuGetProcAddress(const char *symbol, void **pfn, int cuda_version,
                          cuuint64_t flags) {
  lilt_getproc_v11030_fn real_getproc =
      (lilt_getproc_v11030_fn)cuda_real_symbol("cuGetProcAddress");
  CUresult result;

  if (real_getproc == NULL) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  result = real_getproc(symbol, pfn, cuda_version, flags);
  __sync_fetch_and_add(&g_getproc_queries, 1);
  if (result == CUDA_SUCCESS && pfn != NULL && *pfn != NULL) {
    replace_getproc_pointer(symbol, pfn, cuda_version, flags);
  }
  return result;
}

CUresult cuGetProcAddress_v2(
    const char *symbol, void **pfn, int cuda_version, cuuint64_t flags,
    CUdriverProcAddressQueryResult *symbol_status) {
  lilt_getproc_v12000_fn real_getproc =
      (lilt_getproc_v12000_fn)cuda_real_symbol("cuGetProcAddress_v2");
  CUresult result;

  if (real_getproc == NULL) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  result = real_getproc(symbol, pfn, cuda_version, flags, symbol_status);
  __sync_fetch_and_add(&g_getproc_queries, 1);
  if (result == CUDA_SUCCESS && pfn != NULL && *pfn != NULL) {
    replace_getproc_pointer(symbol, pfn, cuda_version, flags);
  }
  return result;
}

static CUresult launch_kernel_ex(const char *symbol,
                                 const CUlaunchConfig *config, CUfunction f,
                                 void **kernel_params, void **extra) {
  lilt_launch_ex_fn real_launch =
      (lilt_launch_ex_fn)cuda_real_symbol(symbol);

  if (real_launch == NULL) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  if (config != NULL) {
    uint64_t blocks = (uint64_t)config->gridDimX * config->gridDimY *
                      config->gridDimZ;
    lilt_before_kernel_launch_ex(
        f, blocks, config->hStream,
        strcmp(symbol, "cuLaunchKernelEx_ptsz") == 0);
  }
  CUresult result = real_launch(config, f, kernel_params, extra);
  if (config != NULL) {
    lilt_after_kernel_launch(config->hStream,
                            strcmp(symbol, "cuLaunchKernelEx_ptsz") == 0,
                            result);
  }
  return result;
}

CUresult cuLaunchKernelEx(const CUlaunchConfig *config, CUfunction f,
                          void **kernel_params, void **extra) {
  return launch_kernel_ex("cuLaunchKernelEx", config, f, kernel_params, extra);
}

CUresult cuLaunchKernelEx_ptsz(const CUlaunchConfig *config, CUfunction f,
                               void **kernel_params, void **extra) {
  return launch_kernel_ex("cuLaunchKernelEx_ptsz", config, f, kernel_params,
                          extra);
}

CUresult cuOccupancyMaxActiveClusters(int *num_clusters, CUfunction function,
                                      const CUlaunchConfig *config) {
  LILT_REAL_FORWARD_OR_NOT_SUPPORTED("cuOccupancyMaxActiveClusters",
                                    lilt_occupancy_clusters_fn);
  return real_function(num_clusters, function, config);
}

CUresult cuOccupancyMaxPotentialClusterSize(int *cluster_size,
                                            CUfunction function,
                                            const CUlaunchConfig *config) {
  LILT_REAL_FORWARD_OR_NOT_SUPPORTED("cuOccupancyMaxPotentialClusterSize",
                                    lilt_occupancy_clusters_fn);
  return real_function(cluster_size, function, config);
}

CUresult cuLibraryLoadData(CUlibrary *library, const void *code,
                           CUjit_option *jit_options,
                           void **jit_option_values,
                           unsigned int num_jit_options,
                           CUlibraryOption *library_options,
                           void **library_option_values,
                           unsigned int num_library_options) {
  LILT_REAL_FORWARD_OR_NOT_SUPPORTED("cuLibraryLoadData",
                                    lilt_library_load_data_fn);
  return real_function(library, code, jit_options, jit_option_values,
                       num_jit_options, library_options, library_option_values,
                       num_library_options);
}

CUresult cuLibraryLoadFromFile(CUlibrary *library, const char *file_name,
                               CUjit_option *jit_options,
                               void **jit_option_values,
                               unsigned int num_jit_options,
                               CUlibraryOption *library_options,
                               void **library_option_values,
                               unsigned int num_library_options) {
  LILT_REAL_FORWARD_OR_NOT_SUPPORTED("cuLibraryLoadFromFile",
                                    lilt_library_load_file_fn);
  return real_function(library, file_name, jit_options, jit_option_values,
                       num_jit_options, library_options, library_option_values,
                       num_library_options);
}

CUresult cuLibraryUnload(CUlibrary library) {
  LILT_REAL_FORWARD_OR_NOT_SUPPORTED("cuLibraryUnload", lilt_library_unload_fn);
  return real_function(library);
}

CUresult cuLibraryGetKernel(CUkernel *kernel, CUlibrary library,
                            const char *name) {
  LILT_REAL_FORWARD_OR_NOT_SUPPORTED("cuLibraryGetKernel",
                                    lilt_library_get_kernel_fn);
  return real_function(kernel, library, name);
}

CUresult cuKernelGetAttribute(int *value, CUfunction_attribute attribute,
                              CUkernel kernel, CUdevice device) {
  LILT_REAL_FORWARD_OR_NOT_SUPPORTED("cuKernelGetAttribute",
                                    lilt_kernel_get_attribute_fn);
  return real_function(value, attribute, kernel, device);
}

CUresult cuKernelSetAttribute(CUfunction_attribute attribute, int value,
                              CUkernel kernel, CUdevice device) {
  LILT_REAL_FORWARD_OR_NOT_SUPPORTED("cuKernelSetAttribute",
                                    lilt_kernel_set_attribute_fn);
  return real_function(attribute, value, kernel, device);
}

__attribute__((destructor)) static void print_getproc_summary() {
  if (g_getproc_queries != 0 || g_getproc_replacements != 0) {
    fprintf(stderr,
            "[Lilt proxy] cuGetProcAddress queries=%llu replacements=%llu\n",
            g_getproc_queries, g_getproc_replacements);
  }
  if (g_dlsym_cuda_queries != 0) {
    fprintf(stderr,
            "[Lilt proxy] dlsym CUDA queries=%llu wrappers=%llu real=%llu\n",
            g_dlsym_cuda_queries, g_dlsym_wrapper_replacements,
            g_dlsym_real_forwards);
  }
}
