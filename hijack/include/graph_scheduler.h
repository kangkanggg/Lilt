#ifndef LILT_GRAPH_POLICY_H
#define LILT_GRAPH_POLICY_H

#include "cuda-subset.h"

void lilt_graph_record(CUgraphExec exec, CUgraph graph,
                      unsigned long long instantiate_flags);
void lilt_graph_invalidate(CUgraphExec exec, const char *reason);
void lilt_graph_forget(CUgraphExec exec);
CUresult lilt_graph_launch(CUgraphExec exec, CUstream stream,
                          int per_thread_default_stream);

/* Modern CUDA Graph entry points are not present in the CUDA 11-era
 * cuda-subset.h used by upstream TGS. They are exported by the proxy with
 * ABI-compatible opaque parameter pointers. */
CUresult cuGraphInstantiateWithFlags(CUgraphExec *exec, CUgraph graph,
                                     unsigned long long flags);
CUresult cuGraphInstantiateWithParams(CUgraphExec *exec, CUgraph graph,
                                      void *params);
CUresult cuGraphInstantiateWithParams_ptsz(CUgraphExec *exec, CUgraph graph,
                                           void *params);
CUresult cuGraphNodeSetEnabled(CUgraphExec exec, CUgraphNode node,
                               unsigned int enabled);
CUresult cuGraphExecUpdate_v2(CUgraphExec exec, CUgraph graph,
                              void *result_info);

#endif
