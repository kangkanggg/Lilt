/* Modified for Lilt (2026): local proxy and HP/BE scheduling. See LICENSE. */
#include "../include/lilt_config.h"
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/activity_tracker.h"
#include "../include/graph_scheduler.h"
#include "../include/hijack.h"

#define LILT_GRAPH_INSTANTIATE_FLAG_UPLOAD 2ULL
#define LILT_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH 4ULL
#define LILT_GRAPHLET_MAX_NODES_DEFAULT 8U
#define LILT_GRAPHLET_MAX_NODES_LIMIT 4096U

typedef CUresult (*graph_get_nodes_fn)(CUgraph, CUgraphNode *, size_t *);
typedef CUresult (*graph_get_edges_fn)(CUgraph, CUgraphNode *, CUgraphNode *,
                                       size_t *);
typedef CUresult (*graph_node_type_fn)(CUgraphNode, CUgraphNodeType *);
typedef CUresult (*graph_clone_fn)(CUgraph *, CUgraph);
typedef CUresult (*graph_find_clone_fn)(CUgraphNode *, CUgraphNode, CUgraph);
typedef CUresult (*graph_destroy_node_fn)(CUgraphNode);
typedef CUresult (*graph_destroy_fn)(CUgraph);
typedef CUresult (*graph_exec_destroy_fn)(CUgraphExec);
typedef CUresult (*graph_instantiate_flags_fn)(CUgraphExec *, CUgraph,
                                               unsigned long long);
typedef CUresult (*graph_instantiate_legacy_fn)(CUgraphExec *, CUgraph,
                                                CUgraphNode *, char *, size_t);
typedef CUresult (*graph_launch_fn)(CUgraphExec, CUstream);
typedef CUresult (*graph_instantiate_params_fn)(CUgraphExec *, CUgraph,
                                                void *);
typedef CUresult (*graph_node_set_enabled_fn)(CUgraphExec, CUgraphNode,
                                              unsigned int);
typedef CUresult (*graph_exec_update_v2_fn)(CUgraphExec, CUgraph, void *);

typedef enum {
  LILT_GRAPH_MODE_OFF = 0,
  LILT_GRAPH_MODE_WHOLE = 1,
  LILT_GRAPH_MODE_GRAPHLET = 2,
} lilt_graph_mode_t;

typedef struct {
  CUgraph graph;
  CUgraphExec exec;
  size_t node_count;
  unsigned int first_layer;
  unsigned int last_layer;
} lilt_graphlet_t;

typedef struct lilt_graph_metadata {
  CUgraphExec original_exec;
  size_t node_count;
  size_t edge_count;
  unsigned long long instantiate_flags;
  int splittable;
  char fallback_reason[64];
  size_t graphlet_count;
  lilt_graphlet_t *graphlets;
  struct lilt_graph_metadata *next;
} lilt_graph_metadata_t;

static pthread_once_t g_graph_config_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t g_graph_lock = PTHREAD_MUTEX_INITIALIZER;
static lilt_graph_metadata_t *g_graph_metadata;
static lilt_graph_mode_t g_graph_mode = LILT_GRAPH_MODE_WHOLE;
static unsigned int g_graphlet_max_nodes = LILT_GRAPHLET_MAX_NODES_DEFAULT;
static int g_graph_audit;

static unsigned long long g_graph_instantiations;
static unsigned long long g_graph_splittable;
static unsigned long long g_graph_fallbacks;
static unsigned long long g_graph_invalidations;
static unsigned long long g_graph_launches;
static unsigned long long g_graph_whole_launches;
static unsigned long long g_graph_split_launches;
static unsigned long long g_graphlet_launches;
static unsigned long long g_graph_build_failures;
static unsigned long long g_graph_launch_failures;

static unsigned long long graph_parse_u64(const char *name,
                                          unsigned long long fallback,
                                          unsigned long long minimum,
                                          unsigned long long maximum) {
  const char *value = lilt_getenv(name);
  if (value == NULL || value[0] == '\0') {
    return fallback;
  }
  char *end = NULL;
  unsigned long long parsed = strtoull(value, &end, 10);
  if (end == value || *end != '\0' || parsed < minimum || parsed > maximum) {
    fprintf(stderr, "[GRAPH] invalid %s=%s; using %llu\n", name, value,
            fallback);
    return fallback;
  }
  return parsed;
}

static const char *graph_mode_name(lilt_graph_mode_t mode) {
  if (mode == LILT_GRAPH_MODE_OFF) {
    return "off";
  }
  return mode == LILT_GRAPH_MODE_GRAPHLET ? "graphlet" : "whole";
}

static void graph_initialize_config(void) {
#if defined(LILT_PROXY_BE)
  const char *mode = lilt_getenv("LILT_EVENT_BE_GRAPH_MODE");
  if (mode != NULL && strcmp(mode, "off") == 0) {
    g_graph_mode = LILT_GRAPH_MODE_OFF;
  } else if (mode != NULL && strcmp(mode, "graphlet") == 0) {
    g_graph_mode = LILT_GRAPH_MODE_GRAPHLET;
  } else {
    if (mode != NULL && mode[0] != '\0' && strcmp(mode, "whole") != 0) {
      fprintf(stderr,
              "[GRAPH][LP] invalid TGS_EVENT_LP_GRAPH_MODE=%s; using whole\n",
              mode);
    }
    g_graph_mode = LILT_GRAPH_MODE_WHOLE;
  }
#else
  g_graph_mode = LILT_GRAPH_MODE_WHOLE;
#endif
  g_graphlet_max_nodes = (unsigned int)graph_parse_u64(
      "LILT_EVENT_BE_GRAPHLET_MAX_NODES", LILT_GRAPHLET_MAX_NODES_DEFAULT, 1,
      LILT_GRAPHLET_MAX_NODES_LIMIT);
  const char *audit = lilt_getenv("LILT_EVENT_GRAPH_AUDIT");
  g_graph_audit = audit != NULL && strcmp(audit, "0") != 0;
  if (lilt_event_policy_enabled() || g_graph_audit) {
    fprintf(stderr, "[GRAPH][%s] mode=%s graphlet_max_nodes=%u audit=%d\n",
#if defined(LILT_PROXY_BE)
            "LP",
#else
            "HP",
#endif
            graph_mode_name(g_graph_mode), g_graphlet_max_nodes,
            g_graph_audit);
  }
}

static lilt_graph_metadata_t *graph_find_locked(CUgraphExec exec) {
  for (lilt_graph_metadata_t *item = g_graph_metadata; item != NULL;
       item = item->next) {
    if (item->original_exec == exec) {
      return item;
    }
  }
  return NULL;
}

static int graph_node_index(const CUgraphNode *nodes, size_t count,
                            CUgraphNode needle) {
  for (size_t i = 0; i < count; ++i) {
    if (nodes[i] == needle) {
      return (int)i;
    }
  }
  return -1;
}

static int graph_node_supported(CUgraphNodeType type) {
  return type == CU_GRAPH_NODE_TYPE_KERNEL ||
         type == CU_GRAPH_NODE_TYPE_MEMCPY ||
         type == CU_GRAPH_NODE_TYPE_MEMSET ||
         type == CU_GRAPH_NODE_TYPE_EMPTY;
}

static void graph_destroy_graphlets(lilt_graph_metadata_t *metadata) {
  graph_exec_destroy_fn destroy_exec =
      (graph_exec_destroy_fn)cuda_real_symbol("cuGraphExecDestroy");
  graph_destroy_fn destroy_graph =
      (graph_destroy_fn)cuda_real_symbol("cuGraphDestroy");
  if (metadata->graphlets != NULL) {
    for (size_t i = 0; i < metadata->graphlet_count; ++i) {
      if (metadata->graphlets[i].exec != NULL && destroy_exec != NULL) {
        destroy_exec(metadata->graphlets[i].exec);
      }
      if (metadata->graphlets[i].graph != NULL && destroy_graph != NULL) {
        destroy_graph(metadata->graphlets[i].graph);
      }
    }
  }
  free(metadata->graphlets);
  metadata->graphlets = NULL;
  metadata->graphlet_count = 0;
  metadata->splittable = 0;
}

static void graph_set_fallback(lilt_graph_metadata_t *metadata,
                               const char *reason) {
  metadata->splittable = 0;
  snprintf(metadata->fallback_reason, sizeof(metadata->fallback_reason),
           "%s", reason);
}

static CUresult graph_instantiate_clone(CUgraphExec *exec, CUgraph graph,
                                        unsigned long long flags) {
  graph_instantiate_flags_fn instantiate =
      (graph_instantiate_flags_fn)cuda_real_symbol(
          "cuGraphInstantiateWithFlags");
  if (instantiate != NULL) {
    return instantiate(exec, graph,
                       flags & ~LILT_GRAPH_INSTANTIATE_FLAG_UPLOAD);
  }
  if (flags == 0) {
    graph_instantiate_legacy_fn legacy =
        (graph_instantiate_legacy_fn)cuda_real_symbol(
            "cuGraphInstantiate_v2");
    if (legacy != NULL) {
      return legacy(exec, graph, NULL, NULL, 0);
    }
  }
  return CUDA_ERROR_NOT_SUPPORTED;
}

static int graph_build_graphlets(lilt_graph_metadata_t *metadata,
                                  CUgraph graph) {
  graph_get_nodes_fn get_nodes =
      (graph_get_nodes_fn)cuda_real_symbol("cuGraphGetNodes");
  graph_get_edges_fn get_edges =
      (graph_get_edges_fn)cuda_real_symbol("cuGraphGetEdges");
  graph_node_type_fn get_type =
      (graph_node_type_fn)cuda_real_symbol("cuGraphNodeGetType");
  graph_clone_fn clone_graph =
      (graph_clone_fn)cuda_real_symbol("cuGraphClone");
  graph_find_clone_fn find_clone =
      (graph_find_clone_fn)cuda_real_symbol("cuGraphNodeFindInClone");
  graph_destroy_node_fn destroy_node =
      (graph_destroy_node_fn)cuda_real_symbol("cuGraphDestroyNode");
  if (get_nodes == NULL || get_edges == NULL || get_type == NULL ||
      clone_graph == NULL || find_clone == NULL || destroy_node == NULL) {
    graph_set_fallback(metadata, "missing_driver_api");
    return 0;
  }

  size_t node_count = 0;
  if (get_nodes(graph, NULL, &node_count) != CUDA_SUCCESS || node_count == 0) {
    graph_set_fallback(metadata, "empty_or_unreadable");
    return 0;
  }
  metadata->node_count = node_count;
  if ((metadata->instantiate_flags &
       LILT_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH) != 0) {
    graph_set_fallback(metadata, "device_launch");
    return 0;
  }

  CUgraphNode *nodes = calloc(node_count, sizeof(*nodes));
  CUgraphNodeType *types = calloc(node_count, sizeof(*types));
  unsigned int *indegree = calloc(node_count, sizeof(*indegree));
  unsigned int *layers = calloc(node_count, sizeof(*layers));
  unsigned char *processed = calloc(node_count, sizeof(*processed));
  if (nodes == NULL || types == NULL || indegree == NULL || layers == NULL ||
      processed == NULL) {
    graph_set_fallback(metadata, "allocation_failure");
    goto fail;
  }
  size_t nodes_capacity = node_count;
  if (get_nodes(graph, nodes, &nodes_capacity) != CUDA_SUCCESS ||
      nodes_capacity != node_count) {
    graph_set_fallback(metadata, "node_query_failure");
    goto fail;
  }
  for (size_t i = 0; i < node_count; ++i) {
    if (get_type(nodes[i], &types[i]) != CUDA_SUCCESS ||
        !graph_node_supported(types[i])) {
      graph_set_fallback(metadata, "unsupported_node");
      goto fail;
    }
  }

  size_t edge_count = 0;
  if (get_edges(graph, NULL, NULL, &edge_count) != CUDA_SUCCESS) {
    graph_set_fallback(metadata, "edge_query_failure");
    goto fail;
  }
  metadata->edge_count = edge_count;
  CUgraphNode *edge_from = edge_count == 0
                               ? NULL
                               : calloc(edge_count, sizeof(*edge_from));
  CUgraphNode *edge_to = edge_count == 0
                             ? NULL
                             : calloc(edge_count, sizeof(*edge_to));
  if (edge_count != 0 && (edge_from == NULL || edge_to == NULL)) {
    free(edge_from);
    free(edge_to);
    graph_set_fallback(metadata, "allocation_failure");
    goto fail;
  }
  size_t edges_capacity = edge_count;
  if (edge_count != 0 &&
      (get_edges(graph, edge_from, edge_to, &edges_capacity) != CUDA_SUCCESS ||
       edges_capacity != edge_count)) {
    free(edge_from);
    free(edge_to);
    graph_set_fallback(metadata, "edge_query_failure");
    goto fail;
  }
  for (size_t i = 0; i < edge_count; ++i) {
    int destination = graph_node_index(nodes, node_count, edge_to[i]);
    if (destination < 0) {
      free(edge_from);
      free(edge_to);
      graph_set_fallback(metadata, "foreign_edge");
      goto fail;
    }
    ++indegree[destination];
  }

  size_t remaining = node_count;
  unsigned int layer_count = 0;
  while (remaining != 0) {
    size_t this_layer = 0;
    for (size_t i = 0; i < node_count; ++i) {
      if (!processed[i] && indegree[i] == 0) {
        layers[i] = layer_count;
        processed[i] = 2;
        ++this_layer;
      }
    }
    if (this_layer == 0) {
      free(edge_from);
      free(edge_to);
      graph_set_fallback(metadata, "cyclic_or_invalid_dag");
      goto fail;
    }
    for (size_t i = 0; i < node_count; ++i) {
      if (processed[i] != 2) {
        continue;
      }
      processed[i] = 1;
      --remaining;
      for (size_t edge = 0; edge < edge_count; ++edge) {
        if (edge_from[edge] != nodes[i]) {
          continue;
        }
        int destination = graph_node_index(nodes, node_count, edge_to[edge]);
        if (destination >= 0 && indegree[destination] > 0) {
          --indegree[destination];
        }
      }
    }
    ++layer_count;
  }
  free(edge_from);
  free(edge_to);

  size_t graphlet_count = 0;
  size_t accumulated = 0;
  for (unsigned int layer = 0; layer < layer_count; ++layer) {
    size_t layer_nodes = 0;
    for (size_t i = 0; i < node_count; ++i) {
      layer_nodes += layers[i] == layer;
    }
    if (accumulated != 0 &&
        accumulated + layer_nodes > g_graphlet_max_nodes) {
      ++graphlet_count;
      accumulated = 0;
    }
    accumulated += layer_nodes;
  }
  if (accumulated != 0) {
    ++graphlet_count;
  }
  metadata->graphlets = calloc(graphlet_count, sizeof(*metadata->graphlets));
  if (metadata->graphlets == NULL) {
    graph_set_fallback(metadata, "allocation_failure");
    goto fail;
  }
  metadata->graphlet_count = graphlet_count;

  unsigned int first_layer = 0;
  accumulated = 0;
  size_t graphlet_index = 0;
  for (unsigned int layer = 0; layer < layer_count; ++layer) {
    size_t layer_nodes = 0;
    for (size_t i = 0; i < node_count; ++i) {
      layer_nodes += layers[i] == layer;
    }
    if (accumulated != 0 &&
        accumulated + layer_nodes > g_graphlet_max_nodes) {
      metadata->graphlets[graphlet_index].first_layer = first_layer;
      metadata->graphlets[graphlet_index].last_layer = layer - 1U;
      metadata->graphlets[graphlet_index].node_count = accumulated;
      ++graphlet_index;
      first_layer = layer;
      accumulated = 0;
    }
    accumulated += layer_nodes;
  }
  metadata->graphlets[graphlet_index].first_layer = first_layer;
  metadata->graphlets[graphlet_index].last_layer = layer_count - 1U;
  metadata->graphlets[graphlet_index].node_count = accumulated;

  for (size_t group = 0; group < graphlet_count; ++group) {
    lilt_graphlet_t *graphlet = &metadata->graphlets[group];
    if (clone_graph(&graphlet->graph, graph) != CUDA_SUCCESS) {
      graph_set_fallback(metadata, "clone_failure");
      goto fail_graphlets;
    }
    CUgraphNode *cloned_nodes = calloc(node_count, sizeof(*cloned_nodes));
    if (cloned_nodes == NULL) {
      graph_set_fallback(metadata, "allocation_failure");
      goto fail_graphlets;
    }
    for (size_t i = 0; i < node_count; ++i) {
      if (find_clone(&cloned_nodes[i], nodes[i], graphlet->graph) !=
          CUDA_SUCCESS) {
        free(cloned_nodes);
        graph_set_fallback(metadata, "clone_mapping_failure");
        goto fail_graphlets;
      }
    }
    for (size_t i = 0; i < node_count; ++i) {
      if (layers[i] < graphlet->first_layer ||
          layers[i] > graphlet->last_layer) {
        if (destroy_node(cloned_nodes[i]) != CUDA_SUCCESS) {
          free(cloned_nodes);
          graph_set_fallback(metadata, "clone_prune_failure");
          goto fail_graphlets;
        }
      }
    }
    free(cloned_nodes);
    if (graph_instantiate_clone(&graphlet->exec, graphlet->graph,
                                metadata->instantiate_flags) != CUDA_SUCCESS) {
      graph_set_fallback(metadata, "graphlet_instantiate_failure");
      goto fail_graphlets;
    }
  }

  metadata->splittable = 1;
  metadata->fallback_reason[0] = '\0';
  free(nodes);
  free(types);
  free(indegree);
  free(layers);
  free(processed);
  return 1;

fail_graphlets:
  graph_destroy_graphlets(metadata);
fail:
  free(nodes);
  free(types);
  free(indegree);
  free(layers);
  free(processed);
  return 0;
}

void lilt_graph_record(CUgraphExec exec, CUgraph graph,
                      unsigned long long instantiate_flags) {
  if (exec == NULL || graph == NULL) {
    return;
  }
  pthread_once(&g_graph_config_once, graph_initialize_config);
  lilt_graph_metadata_t *metadata = calloc(1, sizeof(*metadata));
  if (metadata == NULL) {
    ++g_graph_build_failures;
    return;
  }
  metadata->original_exec = exec;
  metadata->instantiate_flags = instantiate_flags;
  graph_set_fallback(metadata, "whole_mode");

#if defined(LILT_PROXY_BE)
  int should_split = lilt_event_policy_enabled() &&
                     !lilt_event_passthrough_enabled() &&
                     g_graph_mode == LILT_GRAPH_MODE_GRAPHLET;
#else
  int should_split = 0;
#endif
  if (should_split && !graph_build_graphlets(metadata, graph)) {
    ++g_graph_build_failures;
  } else if (!should_split) {
    graph_get_nodes_fn get_nodes =
        (graph_get_nodes_fn)cuda_real_symbol("cuGraphGetNodes");
    if (get_nodes != NULL) {
      get_nodes(graph, NULL, &metadata->node_count);
    }
  }

  pthread_mutex_lock(&g_graph_lock);
  lilt_graph_metadata_t **link = &g_graph_metadata;
  while (*link != NULL) {
    if ((*link)->original_exec == exec) {
      lilt_graph_metadata_t *old = *link;
      *link = old->next;
      graph_destroy_graphlets(old);
      free(old);
      break;
    }
    link = &(*link)->next;
  }
  metadata->next = g_graph_metadata;
  g_graph_metadata = metadata;
  ++g_graph_instantiations;
  if (metadata->splittable) {
    ++g_graph_splittable;
  } else {
    ++g_graph_fallbacks;
  }
  if (g_graph_audit) {
    fprintf(stderr,
            "[GRAPH][%s] instantiate exec=%p nodes=%zu edges=%zu "
            "splittable=%d graphlets=%zu flags=%llu fallback=%s\n",
#if defined(LILT_PROXY_BE)
            "LP",
#else
            "HP",
#endif
            (void *)exec, metadata->node_count, metadata->edge_count,
            metadata->splittable, metadata->graphlet_count,
            metadata->instantiate_flags,
            metadata->splittable ? "none" : metadata->fallback_reason);
  }
  pthread_mutex_unlock(&g_graph_lock);
}

void lilt_graph_invalidate(CUgraphExec exec, const char *reason) {
  if (exec == NULL) {
    return;
  }
  pthread_once(&g_graph_config_once, graph_initialize_config);
  pthread_mutex_lock(&g_graph_lock);
  lilt_graph_metadata_t *metadata = graph_find_locked(exec);
  if (metadata != NULL && metadata->splittable) {
    graph_destroy_graphlets(metadata);
    graph_set_fallback(metadata, reason == NULL ? "updated" : reason);
    ++g_graph_invalidations;
    if (g_graph_audit) {
      fprintf(stderr, "[GRAPH][LP] invalidate exec=%p reason=%s\n",
              (void *)exec, metadata->fallback_reason);
    }
  }
  pthread_mutex_unlock(&g_graph_lock);
}

void lilt_graph_forget(CUgraphExec exec) {
  if (exec == NULL) {
    return;
  }
  pthread_once(&g_graph_config_once, graph_initialize_config);
  pthread_mutex_lock(&g_graph_lock);
  lilt_graph_metadata_t **link = &g_graph_metadata;
  while (*link != NULL) {
    if ((*link)->original_exec == exec) {
      lilt_graph_metadata_t *metadata = *link;
      *link = metadata->next;
      graph_destroy_graphlets(metadata);
      free(metadata);
      break;
    }
    link = &(*link)->next;
  }
  pthread_mutex_unlock(&g_graph_lock);
}

CUresult lilt_graph_launch(CUgraphExec exec, CUstream stream,
                          int per_thread_default_stream) {
  pthread_once(&g_graph_config_once, graph_initialize_config);
  const char *symbol = per_thread_default_stream ? "cuGraphLaunch_ptsz"
                                                  : "cuGraphLaunch";
  graph_launch_fn real_launch =
      (graph_launch_fn)cuda_real_symbol(symbol);
  if (real_launch == NULL) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }

#if defined(LILT_PROXY_BE)
  if (lilt_event_policy_enabled() && g_graph_mode == LILT_GRAPH_MODE_OFF) {
    ++g_graph_launches;
    ++g_graph_whole_launches;
    return real_launch(exec, stream);
  }
#endif

  pthread_mutex_lock(&g_graph_lock);
  lilt_graph_metadata_t *metadata = graph_find_locked(exec);
#if defined(LILT_PROXY_BE)
  int split = lilt_event_policy_enabled() &&
              !lilt_event_passthrough_enabled() &&
              g_graph_mode == LILT_GRAPH_MODE_GRAPHLET && metadata != NULL &&
              metadata->splittable;
#else
  int split = 0;
#endif
  ++g_graph_launches;
  if (split) {
    ++g_graph_split_launches;
    CUresult result = CUDA_SUCCESS;
    for (size_t i = 0; i < metadata->graphlet_count; ++i) {
      lilt_graphlet_t *graphlet = &metadata->graphlets[i];
      lilt_before_kernel_launch_ex((CUfunction)graphlet->exec,
                                  graphlet->node_count, stream,
                                  per_thread_default_stream);
      result = real_launch(graphlet->exec, stream);
      lilt_after_kernel_launch(stream, per_thread_default_stream, result);
      ++g_graphlet_launches;
      if (result != CUDA_SUCCESS) {
        ++g_graph_launch_failures;
        break;
      }
      /* First version: a hard completion boundary prevents later graphlets
       * from becoming irrevocably queued before a new HP arrival is seen. */
      lilt_graphlet_boundary();
    }
    pthread_mutex_unlock(&g_graph_lock);
    return result;
  }
  size_t node_count = metadata != NULL && metadata->node_count != 0
                          ? metadata->node_count
                          : 1;
  ++g_graph_whole_launches;
  pthread_mutex_unlock(&g_graph_lock);

  lilt_before_kernel_launch_ex((CUfunction)exec, node_count, stream,
                              per_thread_default_stream);
  CUresult result = real_launch(exec, stream);
  lilt_after_kernel_launch(stream, per_thread_default_stream, result);
#if defined(LILT_PROXY_BE)
  if (lilt_event_policy_enabled() && !lilt_event_passthrough_enabled()) {
    /* An opaque or unsupported BE Graph must remain a singleton. */
    lilt_graphlet_boundary();
  }
#endif
  if (result != CUDA_SUCCESS) {
    ++g_graph_launch_failures;
  }
  return result;
}


CUresult cuGraphInstantiateWithFlags(CUgraphExec *exec, CUgraph graph,
                                     unsigned long long flags) {
  graph_instantiate_flags_fn real_instantiate =
      (graph_instantiate_flags_fn)cuda_real_symbol(
          "cuGraphInstantiateWithFlags");
  if (real_instantiate == NULL) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  CUresult result = real_instantiate(exec, graph, flags);
  if (result == CUDA_SUCCESS && exec != NULL) {
    lilt_graph_record(*exec, graph, flags);
  }
  return result;
}

static CUresult graph_instantiate_with_params(const char *symbol,
                                              CUgraphExec *exec,
                                              CUgraph graph, void *params) {
  graph_instantiate_params_fn real_instantiate =
      (graph_instantiate_params_fn)cuda_real_symbol(symbol);
  if (real_instantiate == NULL) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  CUresult result = real_instantiate(exec, graph, params);
  if (result == CUDA_SUCCESS && exec != NULL) {
    /* flags is the first field of CUDA_GRAPH_INSTANTIATE_PARAMS. */
    unsigned long long flags =
        params != NULL ? *(const unsigned long long *)params : 0;
    lilt_graph_record(*exec, graph, flags);
  }
  return result;
}

CUresult cuGraphInstantiateWithParams(CUgraphExec *exec, CUgraph graph,
                                      void *params) {
  return graph_instantiate_with_params("cuGraphInstantiateWithParams", exec,
                                       graph, params);
}

CUresult cuGraphInstantiateWithParams_ptsz(CUgraphExec *exec, CUgraph graph,
                                           void *params) {
  return graph_instantiate_with_params("cuGraphInstantiateWithParams_ptsz",
                                       exec, graph, params);
}

CUresult cuGraphNodeSetEnabled(CUgraphExec exec, CUgraphNode node,
                               unsigned int enabled) {
  graph_node_set_enabled_fn real_set_enabled =
      (graph_node_set_enabled_fn)cuda_real_symbol("cuGraphNodeSetEnabled");
  if (real_set_enabled == NULL) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  CUresult result = real_set_enabled(exec, node, enabled);
  if (result == CUDA_SUCCESS) {
    lilt_graph_invalidate(exec, "node_enable_update");
  }
  return result;
}

CUresult cuGraphExecUpdate_v2(CUgraphExec exec, CUgraph graph,
                              void *result_info) {
  graph_exec_update_v2_fn real_update =
      (graph_exec_update_v2_fn)cuda_real_symbol("cuGraphExecUpdate_v2");
  if (real_update == NULL) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  CUresult result = real_update(exec, graph, result_info);
  if (result == CUDA_SUCCESS) {
    lilt_graph_invalidate(exec, "graph_exec_update_v2");
  }
  return result;
}

__attribute__((destructor)) static void graph_print_summary(void) {
  if (g_graph_instantiations == 0 && g_graph_launches == 0) {
    return;
  }
  fprintf(stderr,
          "[GRAPH][%s] instantiations=%llu splittable=%llu fallbacks=%llu "
          "invalidations=%llu launches=%llu whole=%llu split=%llu "
          "graphlet_launches=%llu build_failures=%llu launch_failures=%llu\n",
#if defined(LILT_PROXY_BE)
          "LP",
#else
          "HP",
#endif
          g_graph_instantiations, g_graph_splittable, g_graph_fallbacks,
          g_graph_invalidations, g_graph_launches, g_graph_whole_launches,
          g_graph_split_launches, g_graphlet_launches,
          g_graph_build_failures, g_graph_launch_failures);
}
