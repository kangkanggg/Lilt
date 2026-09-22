/* Modified for Lilt (2026): local proxy and HP/BE scheduling. See LICENSE. */
#include "../include/lilt_config.h"
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <errno.h>
#include <fcntl.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>
#include <math.h>
#include <sys/socket.h>
#include <netdb.h>
#include <sys/types.h>
#include <assert.h>
#include <sys/un.h>
#include <stdint.h>
#include <sched.h>
#include <time.h>

#include "../include/cuda-helper.h"
#include "../include/activity_tracker.h"
#include "../include/hijack.h"
#include "../include/nvml-helper.h"
#include "../include/nvml-subset.h"

extern entry_t cuda_library_entry[];
extern entry_t nvml_library_entry[];
extern char pid_path[];
static const size_t g_spare_memory = 1ull << 30;
static size_t g_used_memory = 0;

static int g_block_x = 1, g_block_y = 1, g_block_z = 1;
static uint32_t g_block_locker = 0;

#define GPU_MAX_NUM 8

static long long g_rate_counter[GPU_MAX_NUM] = {};
static long long g_rate_limit[GPU_MAX_NUM] = {};
static long long g_rate_control_flag[GPU_MAX_NUM] = {};
static long long g_current_rate[GPU_MAX_NUM] = {};
static int g_active_gpu[GPU_MAX_NUM] = {};
static CUuuid g_uuid[GPU_MAX_NUM];
static int g_gpu_id[GPU_MAX_NUM];

const long long LIMIT_INITIALIZER = 20000;
const long long RATE_MIN = 1000;

#define TGS_SLOW_START 0
#define TGS_CONGESTION_AVOIDANCE 1

static const struct timespec g_cycle = {
    .tv_sec = 0,
    .tv_nsec = TIME_TICK * MILLISEC,
};


struct MemRange {
  CUdeviceptr devPtr;
  size_t count;
  CUdevice device;
  struct MemRange *successor, *precursor;
};

static struct MemRange *list_head = NULL;
static size_t list_size = 0;
static pthread_mutex_t g_map_mutex = PTHREAD_MUTEX_INITIALIZER;

static void activate_rate_watcher();
static void *rate_watcher(void *);
static void rate_limiter(const long long);
static void activate_limit_manager();
static void *limit_manager(void *);
static void init_rate_limit(long long, volatile long long *, int *);
static void *memory_transfer_routine(CUdevice device);
static unsigned long long g_driver_launch_calls;
static unsigned long long g_driver_launch_blocks;

#define EVENT_DISABLE_TIMING 2U
#define EVENT_HP_STARTUP_WAIT_NS (2ULL * 1000 * 1000 * 1000)
#define EVENT_BE_WINDOW_DEFAULT 1U
#define EVENT_BE_WINDOW_MAX 64U
#define EVENT_BE_COUNT_DEFAULT 1U
#define EVENT_BE_COUNT_MAX 64U

#define EVENT_BE_BLOCK_BUDGET_DEFAULT 20000ULL
#define EVENT_BE_GAP_FRACTION_DEFAULT 90U
#define EVENT_BE_GAP_GUARD_NS_DEFAULT (50ULL * 1000)
#define EVENT_BE_GAP_MIN_SAMPLES 4U
#define EVENT_BE_TIME_BUDGET_NS_DEFAULT (100ULL * 1000)
#define EVENT_BE_UNKNOWN_KERNEL_NS_DEFAULT (20ULL * 1000)
#define EVENT_BE_PROFILE_TABLE_SIZE 2048U
#define EVENT_BE_PROFILE_SAMPLES 3U
#define EVENT_BE_SURVIVAL_MIN_SAMPLES_DEFAULT 32U
#define EVENT_BE_SURVIVAL_CONFIDENCE_DEFAULT 50U
#define EVENT_BE_SURVIVAL_PROMOTE_CONFIDENCE_DEFAULT 90U
#define EVENT_BE_SURVIVAL_RECHECK_NS_DEFAULT (20ULL * 1000)
#define EVENT_BE_SURVIVAL_COLD_START_NS_DEFAULT (500ULL * 1000)
#define EVENT_BE_SURVIVAL_CONTEXT_NS_DEFAULT (141ULL * 1000)
#define EVENT_BE_SURVIVAL_INTERNAL_WINDOW_DEFAULT 2U
#define EVENT_BE_SURVIVAL_PROMOTED_WINDOW_DEFAULT 8U
#define EVENT_BE_SURVIVAL_INTERNAL_BUDGET_NS_DEFAULT (40ULL * 1000)
#define EVENT_BE_SURVIVAL_PROMOTED_BUDGET_NS_DEFAULT (100ULL * 1000)

typedef enum {
  EVENT_BE_POLICY_FIXED = 0,
  EVENT_BE_POLICY_BLOCKS = 1,
  EVENT_BE_POLICY_GAP = 2,
  EVENT_BE_POLICY_TIME = 3,
  EVENT_BE_POLICY_SURVIVAL = 4,
  EVENT_BE_POLICY_TIME_SUM = 5,
} event_be_policy_t;

typedef struct {
  CUfunction function;
  uint64_t blocks;
  uint64_t duration_ewma_ns;
  uint32_t samples;
  uint32_t pending_samples;
  int occupied;
} lp_kernel_profile_t;

typedef struct {
  CUevent event;
  CUevent timing_start;
  CUevent timing_stop;
  CUstream stream;
  uint64_t blocks;
  uint64_t predicted_ns;
  uint32_t profile_index;
  int per_thread_default_stream;
  int event_created;
  int timing_created;
  int timing_sample;
  int active;
} lp_event_slot_t;

static pthread_mutex_t g_be_event_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t g_be_window_once = PTHREAD_ONCE_INIT;
static lp_event_slot_t g_be_event_slots[EVENT_BE_WINDOW_MAX];
static lp_kernel_profile_t g_be_profiles[EVENT_BE_PROFILE_TABLE_SIZE];
static event_be_policy_t g_be_policy = EVENT_BE_POLICY_FIXED;
static unsigned int g_be_count = EVENT_BE_COUNT_DEFAULT;
static unsigned int g_be_window = EVENT_BE_WINDOW_DEFAULT;
static uint64_t g_be_block_budget = EVENT_BE_BLOCK_BUDGET_DEFAULT;
static unsigned int g_be_gap_fraction = EVENT_BE_GAP_FRACTION_DEFAULT;
static uint64_t g_be_gap_guard_ns = EVENT_BE_GAP_GUARD_NS_DEFAULT;
static uint64_t g_be_time_budget_ns = EVENT_BE_TIME_BUDGET_NS_DEFAULT;
static uint64_t g_be_unknown_kernel_ns = EVENT_BE_UNKNOWN_KERNEL_NS_DEFAULT;
static unsigned int g_be_survival_min_samples =
    EVENT_BE_SURVIVAL_MIN_SAMPLES_DEFAULT;
static unsigned int g_be_survival_confidence =
    EVENT_BE_SURVIVAL_CONFIDENCE_DEFAULT;
static unsigned int g_be_survival_promote_confidence =
    EVENT_BE_SURVIVAL_PROMOTE_CONFIDENCE_DEFAULT;
static uint64_t g_be_survival_recheck_ns =
    EVENT_BE_SURVIVAL_RECHECK_NS_DEFAULT;
static uint64_t g_be_survival_cold_start_ns =
    EVENT_BE_SURVIVAL_COLD_START_NS_DEFAULT;
static uint64_t g_be_survival_context_ns =
    EVENT_BE_SURVIVAL_CONTEXT_NS_DEFAULT;
static unsigned int g_be_survival_internal_window =
    EVENT_BE_SURVIVAL_INTERNAL_WINDOW_DEFAULT;
static unsigned int g_be_survival_promoted_window =
    EVENT_BE_SURVIVAL_PROMOTED_WINDOW_DEFAULT;
static uint64_t g_be_survival_internal_budget_ns =
    EVENT_BE_SURVIVAL_INTERNAL_BUDGET_NS_DEFAULT;
static uint64_t g_be_survival_promoted_budget_ns =
    EVENT_BE_SURVIVAL_PROMOTED_BUDGET_NS_DEFAULT;
static unsigned int g_be_window_head;
static unsigned int g_be_window_tail;
static unsigned int g_be_outstanding_launches;
static unsigned int g_be_max_outstanding;
static uint64_t g_be_outstanding_blocks;
static uint64_t g_be_max_outstanding_blocks;
static uint64_t g_be_current_launch_blocks;
static uint64_t g_be_outstanding_predicted_ns;
static uint64_t g_be_max_outstanding_predicted_ns;
static uint64_t g_be_current_predicted_ns;
static int g_be_current_profile_index = -1;
static int g_be_current_prediction_unknown;
static int g_be_current_timing_sample;
static int g_be_current_launch_uses_window;
static int g_be_startup_wait_done;
static unsigned long long g_be_window_launches;
static unsigned long long g_be_window_full_waits;
static unsigned long long g_be_block_budget_waits;
static unsigned long long g_be_gap_budget_decisions;
static unsigned long long g_be_gap_late_decisions;
static unsigned long long g_be_long_kernel_singletons;
static unsigned long long g_be_time_budget_waits;
static unsigned long long g_be_time_profiles_created;
static unsigned long long g_be_time_samples;
static unsigned long long g_be_time_sample_failures;
static unsigned long long g_be_time_profile_table_full;
static unsigned long long g_be_unknown_predictions;
static unsigned long long g_be_unknown_singleton_launches;
static unsigned long long g_be_completion_waits;
static unsigned long long g_be_hp_wait_episodes;
static unsigned long long g_be_event_create_failures;
static unsigned long long g_be_event_record_failures;
static unsigned long long g_be_event_sync_failures;
static uint64_t g_be_completion_wait_ns;
static uint64_t g_be_hp_wait_ns;
static uint64_t g_be_effective_budget_sum;
static uint64_t g_be_effective_budget_min = UINT64_MAX;
static uint64_t g_be_effective_budget_max;
static uint64_t g_be_survival_episode_last_idle_ns;
static unsigned int g_be_survival_episode_launches;
static unsigned long long g_be_survival_admissions;
static unsigned long long g_be_survival_promotions;
static unsigned long long g_be_survival_rejections;
static unsigned long long g_be_survival_cold_admissions;
static unsigned long long g_be_survival_histogram_decisions;
static unsigned long long g_be_survival_window_waits;

typedef CUresult (*lilt_stream_is_capturing_fn)(CUstream,
                                                   CUstreamCaptureStatus *);
static __thread int g_tgs_capture_bypass_after;
static unsigned long long g_tgs_capture_bypassed_launches;

static int lilt_stream_is_capturing(CUstream stream,
                                   int per_thread_default_stream) {
  const char *symbol = per_thread_default_stream
                           ? "cuStreamIsCapturing_ptsz"
                           : "cuStreamIsCapturing";
  lilt_stream_is_capturing_fn query =
      (lilt_stream_is_capturing_fn)cuda_real_symbol(symbol);
  CUstreamCaptureStatus status = CU_STREAM_CAPTURE_STATUS_NONE;
  return query != NULL && query(stream, &status) == CUDA_SUCCESS &&
         status != CU_STREAM_CAPTURE_STATUS_NONE;
}

static void initialization();
static void event_be_before_launch(CUfunction function, uint64_t blocks,
                                   CUstream stream,
                                   int per_thread_default_stream);
static void event_be_after_launch(CUstream, int, CUresult);
static void event_be_initialize_window(void);
static void event_be_drain_window_locked(void);

void lilt_before_kernel_launch(uint64_t blocks) {
  lilt_before_kernel_launch_ex(NULL, blocks, NULL, 0);
}

void lilt_before_kernel_launch_ex(CUfunction function, uint64_t blocks,
                                 CUstream stream,
                                 int per_thread_default_stream) {
  __sync_fetch_and_add(&g_driver_launch_calls, 1);
  __sync_fetch_and_add(&g_driver_launch_blocks, blocks);
  g_tgs_capture_bypass_after = 0;
  if (lilt_event_passthrough_enabled()) {
    return;
  }
  if (lilt_event_policy_enabled() &&
      lilt_stream_is_capturing(stream, per_thread_default_stream)) {
    g_tgs_capture_bypass_after = 1;
    __sync_fetch_and_add(&g_tgs_capture_bypassed_launches, 1);
    return;
  }
  if (lilt_event_policy_enabled()) {
    event_be_before_launch(function, blocks, stream,
                           per_thread_default_stream);
  } else {
    rate_limiter((long long)blocks);
  }
}

void lilt_after_kernel_launch(CUstream stream, int per_thread_default_stream,
                             CUresult launch_result) {
  if (g_tgs_capture_bypass_after) {
    g_tgs_capture_bypass_after = 0;
    return;
  }
  if (lilt_event_policy_enabled() && !lilt_event_passthrough_enabled()) {
    event_be_after_launch(stream, per_thread_default_stream, launch_result);
  }
}

void lilt_graphlet_boundary(void) {
  if (!lilt_event_policy_enabled() || lilt_event_passthrough_enabled()) {
    return;
  }
  pthread_once(&g_be_window_once, event_be_initialize_window);
  pthread_mutex_lock(&g_be_event_lock);
  event_be_drain_window_locked();
  pthread_mutex_unlock(&g_be_event_lock);
}

static uint64_t event_monotonic_ns(void) {
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  return (uint64_t)now.tv_sec * 1000000000ULL + (uint64_t)now.tv_nsec;
}

static uint64_t event_be_parse_u64(const char *name, uint64_t fallback,
                                   uint64_t minimum, uint64_t maximum) {
  const char *value = lilt_getenv(name);
  if (value == NULL || value[0] == '\0') {
    return fallback;
  }
  char *end = NULL;
  errno = 0;
  unsigned long long parsed = strtoull(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' || parsed < minimum ||
      parsed > maximum) {
    fprintf(stderr,
            "[EVENT][LP] invalid %s='%s'; using %llu "
            "(valid range: %llu-%llu)\n",
            name, value, (unsigned long long)fallback,
            (unsigned long long)minimum, (unsigned long long)maximum);
    return fallback;
  }
  return (uint64_t)parsed;
}

static const char *event_be_policy_name(event_be_policy_t policy) {
  if (policy == EVENT_BE_POLICY_BLOCKS) {
    return "blocks";
  }
  if (policy == EVENT_BE_POLICY_TIME) {
    return "time";
  }
  if (policy == EVENT_BE_POLICY_TIME_SUM) {
    return "time-sum";
  }
  if (policy == EVENT_BE_POLICY_SURVIVAL) {
    return "survival";
  }
  return policy == EVENT_BE_POLICY_GAP ? "gap" : "fixed";
}

static uint64_t event_be_partition_budget(uint64_t total,
                                          unsigned int lp_count) {
  uint64_t partition = total / lp_count;
  return partition == 0 ? 1 : partition;
}

static unsigned int event_be_partition_window(unsigned int total,
                                              unsigned int lp_count) {
  unsigned int partition = total / lp_count;
  return partition == 0 ? 1U : partition;
}

static void event_be_initialize_window(void) {
  g_be_count = (unsigned int)event_be_parse_u64(
      "LILT_EVENT_BE_COUNT", EVENT_BE_COUNT_DEFAULT, 1,
      EVENT_BE_COUNT_MAX);
  g_be_window = (unsigned int)event_be_parse_u64(
      "LILT_EVENT_BE_WINDOW", EVENT_BE_WINDOW_DEFAULT, 1,
      EVENT_BE_WINDOW_MAX);
  g_be_block_budget = event_be_parse_u64(
      "LILT_EVENT_BE_BLOCK_BUDGET", EVENT_BE_BLOCK_BUDGET_DEFAULT, 1,
      UINT64_MAX);
  g_be_gap_fraction = (unsigned int)event_be_parse_u64(
      "LILT_EVENT_BE_GAP_FRACTION_PCT", EVENT_BE_GAP_FRACTION_DEFAULT, 1,
      100);
  g_be_gap_guard_ns = event_be_parse_u64(
      "LILT_EVENT_BE_GAP_GUARD_NS", EVENT_BE_GAP_GUARD_NS_DEFAULT, 0,
      1000ULL * 1000 * 1000);
  g_be_time_budget_ns = event_be_parse_u64(
      "LILT_EVENT_BE_TIME_BUDGET_NS", EVENT_BE_TIME_BUDGET_NS_DEFAULT, 1,
      1000ULL * 1000 * 1000);
  g_be_unknown_kernel_ns = event_be_parse_u64(
      "LILT_EVENT_BE_UNKNOWN_KERNEL_NS", EVENT_BE_UNKNOWN_KERNEL_NS_DEFAULT,
      1, 1000ULL * 1000 * 1000);
  g_be_survival_min_samples = (unsigned int)event_be_parse_u64(
      "LILT_EVENT_BE_SURVIVAL_MIN_SAMPLES",
      EVENT_BE_SURVIVAL_MIN_SAMPLES_DEFAULT, 1, UINT32_MAX);
  g_be_survival_confidence = (unsigned int)event_be_parse_u64(
      "LILT_EVENT_BE_SURVIVAL_CONFIDENCE_PCT",
      EVENT_BE_SURVIVAL_CONFIDENCE_DEFAULT, 1, 100);
  g_be_survival_promote_confidence = (unsigned int)event_be_parse_u64(
      "LILT_EVENT_BE_SURVIVAL_PROMOTE_CONFIDENCE_PCT",
      EVENT_BE_SURVIVAL_PROMOTE_CONFIDENCE_DEFAULT, 1, 100);
  g_be_survival_recheck_ns = event_be_parse_u64(
      "LILT_EVENT_BE_SURVIVAL_RECHECK_NS",
      EVENT_BE_SURVIVAL_RECHECK_NS_DEFAULT, 1000, 1000ULL * 1000 * 1000);
  g_be_survival_cold_start_ns = event_be_parse_u64(
      "LILT_EVENT_BE_SURVIVAL_COLD_START_NS",
      EVENT_BE_SURVIVAL_COLD_START_NS_DEFAULT, 1, 1000ULL * 1000 * 1000);
  g_be_survival_context_ns = event_be_parse_u64(
      "LILT_EVENT_BE_SURVIVAL_CONTEXT_NS",
      EVENT_BE_SURVIVAL_CONTEXT_NS_DEFAULT, 0, 1000ULL * 1000 * 1000);
  g_be_survival_internal_window = (unsigned int)event_be_parse_u64(
      "LILT_EVENT_BE_SURVIVAL_INTERNAL_WINDOW",
      EVENT_BE_SURVIVAL_INTERNAL_WINDOW_DEFAULT, 1, EVENT_BE_WINDOW_MAX);
  g_be_survival_promoted_window = (unsigned int)event_be_parse_u64(
      "LILT_EVENT_BE_SURVIVAL_PROMOTED_WINDOW",
      EVENT_BE_SURVIVAL_PROMOTED_WINDOW_DEFAULT, 1, EVENT_BE_WINDOW_MAX);
  g_be_survival_internal_budget_ns = event_be_parse_u64(
      "LILT_EVENT_BE_SURVIVAL_INTERNAL_BUDGET_NS",
      EVENT_BE_SURVIVAL_INTERNAL_BUDGET_NS_DEFAULT, 1,
      1000ULL * 1000 * 1000);
  g_be_survival_promoted_budget_ns = event_be_parse_u64(
      "LILT_EVENT_BE_SURVIVAL_PROMOTED_BUDGET_NS",
      EVENT_BE_SURVIVAL_PROMOTED_BUDGET_NS_DEFAULT, 1,
      1000ULL * 1000 * 1000);

  const char *policy = lilt_getenv("LILT_EVENT_BE_POLICY");
  if (policy == NULL || policy[0] == '\0' || strcmp(policy, "fixed") == 0) {
    g_be_policy = EVENT_BE_POLICY_FIXED;
  } else if (strcmp(policy, "blocks") == 0) {
    g_be_policy = EVENT_BE_POLICY_BLOCKS;
  } else if (strcmp(policy, "gap") == 0) {
    g_be_policy = EVENT_BE_POLICY_GAP;
  } else if (strcmp(policy, "time") == 0) {
    g_be_policy = EVENT_BE_POLICY_TIME;
  } else if (strcmp(policy, "time-sum") == 0) {
    g_be_policy = EVENT_BE_POLICY_TIME_SUM;
  } else if (strcmp(policy, "survival") == 0) {
    g_be_policy = EVENT_BE_POLICY_SURVIVAL;
  } else {
    fprintf(stderr,
            "[EVENT][LP] invalid TGS_EVENT_LP_POLICY='%s'; using fixed\n",
            policy);
    g_be_policy = EVENT_BE_POLICY_FIXED;
  }

  /*
   * Every BE process owns an independent event window. Without partitioning,
   * N BE tasks multiply the amount of work that may already be queued when HP
   * becomes active. LILT_EVENT_BE_COUNT makes the configured window/budgets
   * aggregate limits and gives each BE an equal conservative share. It does
   * not coordinate or enforce fairness between BE tasks; the CUDA driver remains
   * responsible for scheduling their contexts.
   */
  g_be_window = event_be_partition_window(g_be_window, g_be_count);
  g_be_block_budget =
      event_be_partition_budget(g_be_block_budget, g_be_count);
  g_be_time_budget_ns =
      event_be_partition_budget(g_be_time_budget_ns, g_be_count);
  g_be_survival_internal_window = event_be_partition_window(
      g_be_survival_internal_window, g_be_count);
  g_be_survival_promoted_window = event_be_partition_window(
      g_be_survival_promoted_window, g_be_count);
  g_be_survival_internal_budget_ns = event_be_partition_budget(
      g_be_survival_internal_budget_ns, g_be_count);
  g_be_survival_promoted_budget_ns = event_be_partition_budget(
      g_be_survival_promoted_budget_ns, g_be_count);

  fprintf(stderr,
          "[EVENT][LP] policy=%s lp_count=%u hard_window=%u "
          "block_budget=%llu "
          "time_budget_ns=%llu unknown_kernel_ns=%llu "
          "gap_fraction_pct=%u gap_guard_ns=%llu "
          "survival_min_samples=%u survival_confidence_pct=%u "
          "survival_promote_confidence_pct=%u survival_recheck_ns=%llu "
          "survival_cold_start_ns=%llu survival_context_ns=%llu "
          "survival_windows=%u/%u survival_budgets_ns=%llu/%llu\n",
          event_be_policy_name(g_be_policy), g_be_count, g_be_window,
          (unsigned long long)g_be_block_budget,
          (unsigned long long)g_be_time_budget_ns,
          (unsigned long long)g_be_unknown_kernel_ns, g_be_gap_fraction,
          (unsigned long long)g_be_gap_guard_ns, g_be_survival_min_samples,
          g_be_survival_confidence, g_be_survival_promote_confidence,
          (unsigned long long)g_be_survival_recheck_ns,
          (unsigned long long)g_be_survival_cold_start_ns,
          (unsigned long long)g_be_survival_context_ns,
          g_be_survival_internal_window, g_be_survival_promoted_window,
          (unsigned long long)g_be_survival_internal_budget_ns,
          (unsigned long long)g_be_survival_promoted_budget_ns);
}

static uint32_t event_be_profile_hash(CUfunction function, uint64_t blocks) {
  uint64_t value = ((uint64_t)(uintptr_t)function >> 4U) ^ blocks;
  value ^= value >> 33U;
  value *= 0xff51afd7ed558ccdULL;
  value ^= value >> 33U;
  return (uint32_t)value & (EVENT_BE_PROFILE_TABLE_SIZE - 1U);
}

static int event_be_profile_lookup_locked(CUfunction function,
                                          uint64_t blocks, int create) {
  if (function == NULL) {
    return -1;
  }
  uint32_t start = event_be_profile_hash(function, blocks);
  for (uint32_t offset = 0; offset < EVENT_BE_PROFILE_TABLE_SIZE; ++offset) {
    uint32_t index = (start + offset) & (EVENT_BE_PROFILE_TABLE_SIZE - 1U);
    lp_kernel_profile_t *profile = &g_be_profiles[index];
    if (profile->occupied) {
      if (profile->function == function && profile->blocks == blocks) {
        return (int)index;
      }
      continue;
    }
    if (!create) {
      return -1;
    }
    profile->function = function;
    profile->blocks = blocks;
    profile->occupied = 1;
    ++g_be_time_profiles_created;
    return (int)index;
  }
  if (create) {
    ++g_be_time_profile_table_full;
  }
  return -1;
}

static uint64_t event_be_predict_kernel_ns_locked(CUfunction function,
                                                  uint64_t blocks) {
  int index = event_be_profile_lookup_locked(function, blocks, 1);
  g_be_current_profile_index = index;
  g_be_current_prediction_unknown = 0;
  if (index < 0 || g_be_profiles[index].samples == 0) {
    g_be_current_prediction_unknown = 1;
    ++g_be_unknown_predictions;
    return g_be_unknown_kernel_ns;
  }
  uint64_t learned = g_be_profiles[index].duration_ewma_ns;
  uint64_t margin = learned / 4U;
  return UINT64_MAX - learned < margin ? UINT64_MAX : learned + margin;
}

static void event_be_update_profile_locked(uint32_t index,
                                           uint64_t duration_ns) {
  if (index >= EVENT_BE_PROFILE_TABLE_SIZE || duration_ns == 0) {
    ++g_be_time_sample_failures;
    return;
  }
  lp_kernel_profile_t *profile = &g_be_profiles[index];
  if (!profile->occupied) {
    ++g_be_time_sample_failures;
    return;
  }
  if (profile->samples == 0) {
    profile->duration_ewma_ns = duration_ns;
  } else if (duration_ns >= profile->duration_ewma_ns) {
    profile->duration_ewma_ns +=
        (duration_ns - profile->duration_ewma_ns) / 4U;
  } else {
    profile->duration_ewma_ns -=
        (profile->duration_ewma_ns - duration_ns) / 4U;
  }
  ++profile->samples;
  ++g_be_time_samples;
}

static int event_be_prepare_timing_sample_locked(
    CUstream stream, int per_thread_default_stream, uint32_t profile_index) {
  lp_event_slot_t *slot = &g_be_event_slots[g_be_window_tail];
  assert(!slot->active);
  if (!slot->timing_created) {
    CUresult start_result = CUDA_ENTRY_CALL(
        cuda_library_entry, cuEventCreate, &slot->timing_start, 0U);
    CUresult stop_result = start_result == CUDA_SUCCESS
                               ? CUDA_ENTRY_CALL(cuda_library_entry,
                                                 cuEventCreate,
                                                 &slot->timing_stop, 0U)
                               : start_result;
    if (start_result != CUDA_SUCCESS || stop_result != CUDA_SUCCESS) {
      ++g_be_time_sample_failures;
      return 0;
    }
    slot->timing_created = 1;
  }
  CUresult result;
  if (per_thread_default_stream) {
    result = CUDA_ENTRY_CALL(cuda_library_entry, cuEventRecord_ptsz,
                             slot->timing_start, stream);
  } else {
    result = CUDA_ENTRY_CALL(cuda_library_entry, cuEventRecord,
                             slot->timing_start, stream);
  }
  if (result != CUDA_SUCCESS) {
    ++g_be_time_sample_failures;
    return 0;
  }
  ++g_be_profiles[profile_index].pending_samples;
  slot->profile_index = profile_index;
  return 1;
}

static void event_be_retire_oldest_locked(void) {
  if (g_be_outstanding_launches == 0) {
    return;
  }

  lp_event_slot_t *slot = &g_be_event_slots[g_be_window_head];
  assert(slot->active);
  uint64_t wait_start = event_monotonic_ns();
  CUevent completion_event =
      slot->timing_sample ? slot->timing_stop : slot->event;
  CUresult result = CUDA_ENTRY_CALL(cuda_library_entry, cuEventSynchronize,
                                    completion_event);
  g_be_completion_wait_ns += event_monotonic_ns() - wait_start;
  ++g_be_completion_waits;
  if (result != CUDA_SUCCESS) {
    ++g_be_event_sync_failures;
    fprintf(stderr, "[EVENT][LP] cuEventSynchronize failed: %d\n",
            (int)result);
    if (slot->per_thread_default_stream) {
      CUDA_ENTRY_CALL(cuda_library_entry, cuStreamSynchronize_ptsz,
                      slot->stream);
    } else {
      CUDA_ENTRY_CALL(cuda_library_entry, cuStreamSynchronize, slot->stream);
    }
  }

  if (slot->timing_sample &&
      g_be_profiles[slot->profile_index].pending_samples > 0) {
    --g_be_profiles[slot->profile_index].pending_samples;
  }
  if (result == CUDA_SUCCESS && slot->timing_sample) {
    float duration_ms = 0.0f;
    CUresult elapsed_result = CUDA_ENTRY_CALL(
        cuda_library_entry, cuEventElapsedTime, &duration_ms,
        slot->timing_start, slot->timing_stop);
    if (elapsed_result == CUDA_SUCCESS && duration_ms > 0.0f) {
      event_be_update_profile_locked(
          slot->profile_index, (uint64_t)((double)duration_ms * 1000000.0));
    } else {
      ++g_be_time_sample_failures;
    }
  }

  assert(g_be_outstanding_blocks >= slot->blocks);
  g_be_outstanding_blocks -= slot->blocks;
  assert(g_be_outstanding_predicted_ns >= slot->predicted_ns);
  g_be_outstanding_predicted_ns -= slot->predicted_ns;
  slot->blocks = 0;
  slot->predicted_ns = 0;
  slot->timing_sample = 0;
  slot->active = 0;
  g_be_window_head = (g_be_window_head + 1U) % g_be_window;
  --g_be_outstanding_launches;
}

static void event_be_drain_window_locked(void) {
  while (g_be_outstanding_launches > 0) {
    event_be_retire_oldest_locked();
  }
}

static void event_be_wait_for_hp_registration_locked(void) {
  if (g_be_startup_wait_done) {
    return;
  }
  const struct timespec retry = {.tv_sec = 0, .tv_nsec = 1000 * 1000};
  uint64_t deadline = event_monotonic_ns() + EVENT_HP_STARTUP_WAIT_NS;
  while (lilt_event_hp_owner() <= 0 && event_monotonic_ns() < deadline) {
    nanosleep(&retry, NULL);
  }
  g_be_startup_wait_done = 1;
}

static uint64_t event_be_effective_block_budget_locked(void) {
  uint64_t budget = g_be_block_budget;
  if (g_be_policy != EVENT_BE_POLICY_GAP) {
    return budget;
  }

  lilt_event_hp_timing_t timing;
  if (lilt_event_hp_timing_snapshot(&timing) == 0 &&
      timing.hp_state == LILT_EVENT_HP_IDLE &&
      timing.idle_samples >= EVENT_BE_GAP_MIN_SAMPLES &&
      timing.idle_ewma_ns > 0 && timing.last_idle_ns > 0) {
    uint64_t now = event_monotonic_ns();
    uint64_t elapsed = now > timing.last_idle_ns ? now - timing.last_idle_ns : 0;
    uint64_t consumed = elapsed;
    if (UINT64_MAX - consumed < g_be_gap_guard_ns) {
      consumed = UINT64_MAX;
    } else {
      consumed += g_be_gap_guard_ns;
    }
    uint64_t remaining = consumed < timing.idle_ewma_ns
                             ? timing.idle_ewma_ns - consumed
                             : 0;
    double ratio = ((double)remaining / (double)timing.idle_ewma_ns) *
                   ((double)g_be_gap_fraction / 100.0);
    budget = (uint64_t)((double)g_be_block_budget * ratio);
    if (budget == 0) {
      budget = 1;
      ++g_be_gap_late_decisions;
    }
    ++g_be_gap_budget_decisions;
    g_be_effective_budget_sum += budget;
    if (budget < g_be_effective_budget_min) {
      g_be_effective_budget_min = budget;
    }
    if (budget > g_be_effective_budget_max) {
      g_be_effective_budget_max = budget;
    }
  }
  return budget;
}

static uint64_t event_be_effective_time_budget_locked(void) {
  uint64_t budget = g_be_time_budget_ns;
  /* time-sum is intentionally independent of HP-gap prediction: its
   * admission decision is solely the sum of online BE-kernel duration
   * estimates. EVENT_BE_WINDOW remains only a fail-safe ring bound. */
  if (g_be_policy == EVENT_BE_POLICY_TIME_SUM) {
    return budget;
  }
  lilt_event_hp_timing_t timing;
  if (lilt_event_hp_timing_snapshot(&timing) == 0 &&
      timing.hp_state == LILT_EVENT_HP_IDLE &&
      timing.idle_samples >= EVENT_BE_GAP_MIN_SAMPLES &&
      timing.idle_ewma_ns > 0 && timing.last_idle_ns > 0) {
    uint64_t now = event_monotonic_ns();
    uint64_t elapsed = now > timing.last_idle_ns ? now - timing.last_idle_ns : 0;
    uint64_t consumed = elapsed;
    if (UINT64_MAX - consumed < g_be_gap_guard_ns) {
      consumed = UINT64_MAX;
    } else {
      consumed += g_be_gap_guard_ns;
    }
    uint64_t remaining = consumed < timing.idle_ewma_ns
                             ? timing.idle_ewma_ns - consumed
                             : 0;
    uint64_t safe_remaining =
        (uint64_t)((double)remaining *
                   ((double)g_be_gap_fraction / 100.0));
    if (safe_remaining < budget) {
      budget = safe_remaining;
    }
    if (budget == 0) {
      budget = 1;
      ++g_be_gap_late_decisions;
    }
    ++g_be_gap_budget_decisions;
    g_be_effective_budget_sum += budget;
    if (budget < g_be_effective_budget_min) {
      g_be_effective_budget_min = budget;
    }
    if (budget > g_be_effective_budget_max) {
      g_be_effective_budget_max = budget;
    }
  }
  return budget;
}

static int event_be_blocks_exceed_budget(uint64_t incoming,
                                         uint64_t budget) {
  return incoming > budget || g_be_outstanding_blocks > budget - incoming;
}

static int event_be_time_exceeds_budget(uint64_t incoming, uint64_t budget) {
  return incoming > budget ||
         g_be_outstanding_predicted_ns > budget - incoming;
}

static uint64_t event_be_add_saturating(uint64_t left, uint64_t right) {
  return UINT64_MAX - left < right ? UINT64_MAX : left + right;
}

static uint64_t event_be_survival_denominator(
    const lilt_event_hp_timing_t *timing, uint64_t elapsed_ns) {
  uint64_t denominator = timing->idle_samples;
  for (uint32_t i = 0; i < LILT_EVENT_IDLE_SURVIVAL_BUCKETS; ++i) {
    uint64_t threshold = lilt_event_idle_survival_threshold_ns(i);
    if (threshold == 0 || threshold > elapsed_ns) {
      break;
    }
    denominator = timing->idle_survival_counts[i];
  }
  return denominator;
}

static uint64_t event_be_survival_numerator(
    const lilt_event_hp_timing_t *timing, uint64_t target_ns) {
  for (uint32_t i = 0; i < LILT_EVENT_IDLE_SURVIVAL_BUCKETS; ++i) {
    uint64_t threshold = lilt_event_idle_survival_threshold_ns(i);
    if (threshold >= target_ns) {
      return timing->idle_survival_counts[i];
    }
  }
  return 0;
}

static unsigned int event_be_survival_probability_pct(
    const lilt_event_hp_timing_t *timing, uint64_t elapsed_ns,
    uint64_t additional_ns, uint64_t *denominator_out) {
  uint64_t denominator = event_be_survival_denominator(timing, elapsed_ns);
  uint64_t target = event_be_add_saturating(elapsed_ns, additional_ns);
  uint64_t numerator = event_be_survival_numerator(timing, target);
  if (denominator_out != NULL) {
    *denominator_out = denominator;
  }
  if (denominator == 0) {
    return 0;
  }
  double probability = 100.0 * (double)numerator / (double)denominator;
  if (probability >= 100.0) {
    return 100U;
  }
  return (unsigned int)probability;
}

static int event_be_survival_admission_locked(
    uint64_t incoming_ns, unsigned int *episode_window,
    uint64_t *episode_budget_ns) {
  lilt_event_hp_timing_t timing;
  if (lilt_event_hp_timing_snapshot(&timing) != 0 ||
      timing.hp_state != LILT_EVENT_HP_IDLE || timing.last_idle_ns == 0) {
    return 0;
  }

  if (timing.last_idle_ns != g_be_survival_episode_last_idle_ns) {
    g_be_survival_episode_last_idle_ns = timing.last_idle_ns;
    g_be_survival_episode_launches = 0;
  }
  uint64_t now = event_monotonic_ns();
  uint64_t elapsed_ns =
      now > timing.last_idle_ns ? now - timing.last_idle_ns : 0;
  uint64_t context_ns =
      g_be_survival_episode_launches == 0 ? g_be_survival_context_ns : 0;
  uint64_t additional_ns = event_be_add_saturating(
      context_ns,
      event_be_add_saturating(g_be_outstanding_predicted_ns, incoming_ns));
  additional_ns = event_be_add_saturating(additional_ns, g_be_gap_guard_ns);

  uint64_t denominator = 0;
  unsigned int probability = event_be_survival_probability_pct(
      &timing, elapsed_ns, additional_ns, &denominator);
  int global_histogram_ready =
      timing.idle_samples >= g_be_survival_min_samples;
  int histogram_ready =
      global_histogram_ready && denominator >= g_be_survival_min_samples;
  if (histogram_ready) {
    ++g_be_survival_histogram_decisions;
  }

  uint64_t longest_threshold = lilt_event_idle_survival_threshold_ns(
      LILT_EVENT_IDLE_SURVIVAL_BUCKETS - 1U);
  int very_long_observation =
      longest_threshold > 0 && elapsed_ns >= longest_threshold;
  int cold_admission = !global_histogram_ready &&
                       elapsed_ns >= g_be_survival_cold_start_ns;
  int sparse_tail_admission = global_histogram_ready && !histogram_ready &&
                              elapsed_ns >= g_be_survival_cold_start_ns;
  if (!very_long_observation && !cold_admission && !sparse_tail_admission &&
      (!histogram_ready || probability < g_be_survival_confidence)) {
    ++g_be_survival_rejections;
    return 0;
  }

  unsigned int promoted_probability = event_be_survival_probability_pct(
      &timing, elapsed_ns,
      event_be_add_saturating(
          context_ns,
          event_be_add_saturating(g_be_outstanding_predicted_ns,
                                  g_be_survival_promoted_budget_ns)),
      NULL);
  int promoted = very_long_observation ||
                 (histogram_ready && promoted_probability >=
                                          g_be_survival_promote_confidence);
  unsigned int selected_window = promoted
                                     ? g_be_survival_promoted_window
                                     : g_be_survival_internal_window;
  if (selected_window > g_be_window) {
    selected_window = g_be_window;
  }
  *episode_window = selected_window;
  *episode_budget_ns = promoted ? g_be_survival_promoted_budget_ns
                                : g_be_survival_internal_budget_ns;
  ++g_be_survival_admissions;
  if (promoted) {
    ++g_be_survival_promotions;
  }
  if (cold_admission) {
    ++g_be_survival_cold_admissions;
  }
  return 1;
}

static void event_be_before_launch(CUfunction function, uint64_t blocks,
                                   CUstream stream, int per_thread_default_stream) {
  pthread_once(&g_be_window_once, event_be_initialize_window);
  pthread_mutex_lock(&g_be_event_lock);
  g_be_current_launch_blocks = blocks;
  g_be_current_predicted_ns = 0;
  g_be_current_profile_index = -1;
  g_be_current_prediction_unknown = 0;
  g_be_current_timing_sample = 0;
  event_be_wait_for_hp_registration_locked();

  pid_t owner = lilt_event_hp_owner();
  /* There is no HP work to protect before registration or after the HP exits.
   * Bypassing the BE window here also keeps framework compilation/autotuning
   * out of the measured scheduling policy. The next launch observes a newly
   * registered owner and immediately re-enters the normal event policy. */
  if (owner <= 0) {
    event_be_drain_window_locked();
    g_be_current_launch_uses_window = 0;
    return;
  }

  if (owner > 0) {
    int waited = lilt_event_be_wait_until_idle(&g_be_hp_wait_ns);
    if (waited > 0) {
      ++g_be_hp_wait_episodes;
    }
    if (g_be_policy == EVENT_BE_POLICY_SURVIVAL) {
      g_be_current_predicted_ns =
          event_be_predict_kernel_ns_locked(function, blocks);
      if (g_be_current_predicted_ns > g_be_survival_internal_budget_ns) {
        ++g_be_long_kernel_singletons;
      }
      const struct timespec recheck = {
          .tv_sec = (time_t)(g_be_survival_recheck_ns / 1000000000ULL),
          .tv_nsec =
              (long)(g_be_survival_recheck_ns % 1000000000ULL),
      };
      for (;;) {
        waited = lilt_event_be_wait_until_idle(&g_be_hp_wait_ns);
        if (waited > 0) {
          ++g_be_hp_wait_episodes;
        }
        unsigned int episode_window = 1U;
        uint64_t episode_budget_ns = 1U;
        if (!event_be_survival_admission_locked(
                g_be_current_predicted_ns, &episode_window,
                &episode_budget_ns)) {
          if (g_be_outstanding_launches > 0) {
            event_be_retire_oldest_locked();
          } else {
            nanosleep(&recheck, NULL);
          }
          continue;
        }
        /* An unseen signature must not fill a speculative K>1 window. */
        if (g_be_current_prediction_unknown) {
          episode_window = 1U;
        }
        if (g_be_outstanding_launches > 0 &&
            (g_be_outstanding_launches >= episode_window ||
             event_be_time_exceeds_budget(g_be_current_predicted_ns,
                                          episode_budget_ns))) {
          ++g_be_survival_window_waits;
          if (g_be_outstanding_launches >= episode_window) {
            ++g_be_window_full_waits;
          } else {
            ++g_be_time_budget_waits;
          }
          event_be_retire_oldest_locked();
          continue;
        }
        break;
      }
      if (g_be_current_prediction_unknown) {
        ++g_be_unknown_singleton_launches;
      }
      if (g_be_current_profile_index >= 0 &&
          g_be_profiles[g_be_current_profile_index].samples +
                  g_be_profiles[g_be_current_profile_index].pending_samples <
              EVENT_BE_PROFILE_SAMPLES) {
        g_be_current_timing_sample = event_be_prepare_timing_sample_locked(
            stream, per_thread_default_stream,
            (uint32_t)g_be_current_profile_index);
      }
    } else if (g_be_policy == EVENT_BE_POLICY_TIME ||
               g_be_policy == EVENT_BE_POLICY_TIME_SUM) {
      g_be_current_predicted_ns =
          event_be_predict_kernel_ns_locked(function, blocks);
      uint64_t effective_budget = event_be_effective_time_budget_locked();
      if (g_be_current_predicted_ns > g_be_time_budget_ns) {
        ++g_be_long_kernel_singletons;
      }
      while (g_be_outstanding_launches > 0 &&
             (g_be_current_prediction_unknown ||
              g_be_outstanding_launches >= g_be_window ||
              event_be_time_exceeds_budget(g_be_current_predicted_ns,
                                           effective_budget))) {
        if (g_be_outstanding_launches >= g_be_window) {
          ++g_be_window_full_waits;
        } else {
          ++g_be_time_budget_waits;
        }
        event_be_retire_oldest_locked();
      }
      if (g_be_current_profile_index >= 0 &&
          g_be_profiles[g_be_current_profile_index].samples +
                  g_be_profiles[g_be_current_profile_index].pending_samples <
              EVENT_BE_PROFILE_SAMPLES) {
        g_be_current_timing_sample = event_be_prepare_timing_sample_locked(
            stream, per_thread_default_stream,
            (uint32_t)g_be_current_profile_index);
      }
    } else {
      uint64_t effective_budget = event_be_effective_block_budget_locked();
      if (g_be_policy != EVENT_BE_POLICY_FIXED &&
          blocks > g_be_block_budget) {
        ++g_be_long_kernel_singletons;
      }
      while (g_be_outstanding_launches > 0 &&
             (g_be_outstanding_launches >= g_be_window ||
              (g_be_policy != EVENT_BE_POLICY_FIXED &&
               event_be_blocks_exceed_budget(blocks, effective_budget)))) {
        if (g_be_outstanding_launches >= g_be_window) {
          ++g_be_window_full_waits;
        } else {
          ++g_be_block_budget_waits;
        }
        event_be_retire_oldest_locked();
      }
    }
    g_be_current_launch_uses_window = 1;
    ++g_be_window_launches;
  } else {
    g_be_current_launch_uses_window = 0;
  }
}

static void event_be_after_launch(CUstream stream,
                                  int per_thread_default_stream,
                                  CUresult launch_result) {
  if (launch_result == CUDA_SUCCESS && g_be_current_launch_uses_window) {
    lp_event_slot_t *slot = &g_be_event_slots[g_be_window_tail];
    assert(!slot->active);
    if (!g_be_current_timing_sample && !slot->event_created) {
      CUresult create_result = CUDA_ENTRY_CALL(
          cuda_library_entry, cuEventCreate, &slot->event,
          EVENT_DISABLE_TIMING);
      if (create_result == CUDA_SUCCESS) {
        slot->event_created = 1;
      } else {
        ++g_be_event_create_failures;
        fprintf(stderr, "[EVENT][LP] cuEventCreate failed: %d\n",
                (int)create_result);
      }
    }

    CUresult record_result = CUDA_ERROR_UNKNOWN;
    CUevent completion_event = g_be_current_timing_sample
                                   ? slot->timing_stop
                                   : slot->event;
    if (g_be_current_timing_sample || slot->event_created) {
      if (per_thread_default_stream) {
        record_result = CUDA_ENTRY_CALL(cuda_library_entry, cuEventRecord_ptsz,
                                        completion_event, stream);
      } else {
        record_result = CUDA_ENTRY_CALL(cuda_library_entry, cuEventRecord,
                                        completion_event, stream);
      }
    }
    if (record_result == CUDA_SUCCESS) {
      slot->stream = stream;
      slot->blocks = g_be_current_launch_blocks;
      slot->predicted_ns = g_be_current_predicted_ns;
      slot->profile_index = g_be_current_profile_index >= 0
                                ? (uint32_t)g_be_current_profile_index
                                : 0U;
      slot->per_thread_default_stream = per_thread_default_stream;
      slot->timing_sample = g_be_current_timing_sample;
      slot->active = 1;
      g_be_window_tail = (g_be_window_tail + 1U) % g_be_window;
      ++g_be_outstanding_launches;
      g_be_outstanding_blocks += slot->blocks;
      g_be_outstanding_predicted_ns += slot->predicted_ns;
      if (g_be_outstanding_launches > g_be_max_outstanding) {
        g_be_max_outstanding = g_be_outstanding_launches;
      }
      if (g_be_outstanding_blocks > g_be_max_outstanding_blocks) {
        g_be_max_outstanding_blocks = g_be_outstanding_blocks;
      }
      if (g_be_outstanding_predicted_ns >
          g_be_max_outstanding_predicted_ns) {
        g_be_max_outstanding_predicted_ns =
            g_be_outstanding_predicted_ns;
      }
    } else {
      ++g_be_event_record_failures;
      if (g_be_current_timing_sample) {
        if (g_be_current_profile_index >= 0 &&
            g_be_profiles[g_be_current_profile_index].pending_samples > 0) {
          --g_be_profiles[g_be_current_profile_index].pending_samples;
        }
        ++g_be_time_sample_failures;
      }
      if (per_thread_default_stream) {
        CUDA_ENTRY_CALL(cuda_library_entry, cuStreamSynchronize_ptsz, stream);
      } else {
        CUDA_ENTRY_CALL(cuda_library_entry, cuStreamSynchronize, stream);
      }
    }
    if (g_be_policy == EVENT_BE_POLICY_SURVIVAL) {
      ++g_be_survival_episode_launches;
    }
  }
  if (launch_result != CUDA_SUCCESS && g_be_current_timing_sample &&
      g_be_current_profile_index >= 0) {
    if (g_be_profiles[g_be_current_profile_index].pending_samples > 0) {
      --g_be_profiles[g_be_current_profile_index].pending_samples;
    }
    ++g_be_time_sample_failures;
  }
  g_be_current_launch_uses_window = 0;
  g_be_current_launch_blocks = 0;
  g_be_current_predicted_ns = 0;
  g_be_current_profile_index = -1;
  g_be_current_prediction_unknown = 0;
  g_be_current_timing_sample = 0;
  pthread_mutex_unlock(&g_be_event_lock);
}

__attribute__((destructor)) static void print_driver_launch_summary() {
  if (lilt_event_policy_enabled() && !lilt_event_passthrough_enabled() &&
      g_driver_launch_calls != 0) {
    fprintf(stderr,
            "[EVENT][LP] launches=%llu policy=%s lp_count=%u window=%u "
            "block_budget=%llu window_launches=%llu "
            "window_full_waits=%llu block_budget_waits=%llu "
            "completion_waits=%llu completion_wait_ms=%.3f "
            "max_outstanding=%u max_outstanding_blocks=%llu hp_waits=%llu "
            "hp_wait_ms=%.3f create_failures=%llu record_failures=%llu "
            "sync_failures=%llu gap_decisions=%llu gap_late=%llu "
            "effective_budget_min=%llu effective_budget_avg=%.1f "
            "effective_budget_max=%llu long_kernel_singletons=%llu "
            "time_budget_ns=%llu time_budget_waits=%llu "
            "max_outstanding_predicted_ns=%llu time_profiles=%llu "
            "time_samples=%llu time_sample_failures=%llu "
            "time_profile_table_full=%llu unknown_predictions=%llu "
            "unknown_singleton_launches=%llu\n",
            g_driver_launch_calls, event_be_policy_name(g_be_policy),
            g_be_count, g_be_window, (unsigned long long)g_be_block_budget,
            g_be_window_launches, g_be_window_full_waits,
            g_be_block_budget_waits, g_be_completion_waits,
            (double)g_be_completion_wait_ns / 1000000.0,
            g_be_max_outstanding,
            (unsigned long long)g_be_max_outstanding_blocks,
            g_be_hp_wait_episodes,
            (double)g_be_hp_wait_ns / 1000000.0,
            g_be_event_create_failures, g_be_event_record_failures,
            g_be_event_sync_failures, g_be_gap_budget_decisions,
            g_be_gap_late_decisions,
            (unsigned long long)(g_be_effective_budget_min == UINT64_MAX
                                     ? 0
                                     : g_be_effective_budget_min),
            g_be_gap_budget_decisions == 0
                ? 0.0
                : (double)g_be_effective_budget_sum /
                      (double)g_be_gap_budget_decisions,
            (unsigned long long)g_be_effective_budget_max,
            g_be_long_kernel_singletons,
            (unsigned long long)g_be_time_budget_ns,
            g_be_time_budget_waits,
            (unsigned long long)g_be_max_outstanding_predicted_ns,
            g_be_time_profiles_created, g_be_time_samples,
            g_be_time_sample_failures, g_be_time_profile_table_full,
            g_be_unknown_predictions, g_be_unknown_singleton_launches);
    if (g_be_policy == EVENT_BE_POLICY_SURVIVAL) {
      fprintf(stderr,
              "[EVENT][LP][survival] admissions=%llu promotions=%llu "
              "rejections=%llu cold_admissions=%llu "
              "histogram_decisions=%llu window_waits=%llu\n",
              g_be_survival_admissions, g_be_survival_promotions,
              g_be_survival_rejections, g_be_survival_cold_admissions,
              g_be_survival_histogram_decisions,
              g_be_survival_window_waits);
    }
  }
  if (g_tgs_capture_bypassed_launches != 0) {
    fprintf(stderr,
            "[GRAPH][LP] capture_kernel_bypass=%llu\n",
            g_tgs_capture_bypassed_launches);
  }
  if (g_driver_launch_calls != 0) {
    fprintf(stderr,
            "[%s proxy][LP] Driver kernel launches=%llu blocks=%llu\n",
            lilt_event_passthrough_enabled()
                ? "PASSTHROUGH"
                : (lilt_event_policy_enabled() ? "EVENT" : "TGS"),
            g_driver_launch_calls, g_driver_launch_blocks);
  }
}

static const char *cuda_error(CUresult, const char **);

/*
 * memory transfer
 */

void init_list() {
  pthread_mutex_lock(&g_map_mutex);
  list_head = (struct MemRange*)malloc(sizeof(struct MemRange));
  list_head->count = 0;
  list_head->devPtr = -1;
  list_head->device = 0;
  list_head->precursor = NULL;
  list_head->successor = NULL;
  pthread_mutex_unlock(&g_map_mutex);
  LOGGER(4, "list_head: %p\n", list_head);
}


void list_insert(struct MemRange *pos, struct MemRange *item) {
  item->successor = pos->successor;
  item->precursor = pos;
  if (pos->successor)
    pos->successor->precursor = item;
  pos->successor = item;
  ++list_size;
}


void list_delete(struct MemRange *item) {
  if (item->precursor)
    item->precursor->successor = item->successor;
  if (item->successor)
    item->successor->precursor = item->precursor;
  item->precursor = NULL;
  item->successor = NULL;
  --list_size;
}


void allocate_mem(CUdeviceptr devPtr, size_t count, CUdevice device) {
  struct MemRange *item = (struct MemRange *)malloc(sizeof(struct MemRange));
  item->devPtr = devPtr;
  item->count = count;
  item->device = device;
  item->precursor = item->successor = NULL;

  if (list_head == NULL)
    init_list();

  pthread_mutex_lock(&g_map_mutex);
  g_used_memory += count;
  list_insert(list_head, item);
  pthread_mutex_unlock(&g_map_mutex);
}


void delete_mem(CUdeviceptr devPtr) {
  if (list_head == NULL)
    init_list();

  int ptr_find = 0;

  pthread_mutex_lock(&g_map_mutex);

  for (struct MemRange *it = list_head->successor; it; it = it->successor) {
    if (it->devPtr == devPtr) {
      ptr_find = 1;
      g_used_memory -= it->count;
      list_delete(it);
      break;
    }
  }
  pthread_mutex_unlock(&g_map_mutex);

  if (ptr_find == 0) {
  }
}


const char *cuda_error(CUresult code, const char **p) {
  CUDA_ENTRY_CALL(cuda_library_entry, cuGetErrorString, code, p);
  return *p;
}

static ssize_t rio_readn(int fd, void *usrbuf, size_t n) {
  size_t nleft = n;
  ssize_t nread;
  char *bufp = usrbuf;

  while (nleft > 0) {
	  if ((nread = read(fd, bufp, nleft)) < 0) {
	    if (errno == EINTR)
		    nread = 0;
	    else
		    return -1;
	  }
	  else if (nread == 0)
	    break;
	  nleft -= nread;
	  bufp += nread;
  }
    return (n - nleft);
}


static int open_listenfd(CUdevice device) {
  char SOCKET_PATH[108];
  const int LISTENQ = 8;
  struct sockaddr_un name;
  int ret;
  int listenfd;

  /* Create local socket. */

  listenfd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (listenfd == -1) {
    fprintf(stderr, "socket failed: %s\n", strerror(errno));
  }

  /*
  * For portability clear the whole structure, since some
  * implementations have additional (nonstandard) fields in
  * the structure.
  */

  memset(&name, 0, sizeof(name));

  /* Bind socket to socket name. */

  char uuid_str[37] = {};
  for (int i = 0; i < 16; ++i) {
    unsigned char byte = g_uuid[device].bytes[i];
    uuid_str[2*i] = byte / 16;
    uuid_str[2*i+1] = byte % 16;
  }
  for (int i = 0; i < 32; ++i)
    uuid_str[i] = uuid_str[i] >= 10 ? uuid_str[i] - 10 + 'a' : uuid_str[i] + '0';
  uuid_str[32] = 0;

  const char *socket_dir = lilt_getenv("TGS_SOCKET_DIR");
  if (socket_dir == NULL || socket_dir[0] == '\0')
    socket_dir = "/etc/gsharing";
  snprintf(SOCKET_PATH, sizeof(SOCKET_PATH), "%s/rate_%s.sock", socket_dir, uuid_str);

  name.sun_family = AF_UNIX;
  strncpy(name.sun_path, SOCKET_PATH, sizeof(name.sun_path));

  ret = unlink(SOCKET_PATH);
  if (ret == -1) {
    if (!access(SOCKET_PATH, F_OK))
      fprintf(stderr, "unlink failed: %s\n", strerror(errno));
  }

  ret = bind(listenfd, (const struct sockaddr *) &name, sizeof(name));
  if (ret == -1) {
    fprintf(stderr, "bind failed: %s\n", strerror(errno));
  }

  /*
  * Prepare for accepting connections. The backlog size is set
  * to LISTENQ. So while one request is being processed other requests
  * can be waiting.
  */

  ret = listen(listenfd, LISTENQ);
  if (ret == -1) {
    fprintf(stderr, "listen failed: %s\n", strerror(errno));
  }

  return listenfd;
}

static void init_rate_limit(long long initial_value, volatile long long *p_rate_limit, int *p_state) {
  *p_rate_limit = initial_value;
  *p_state = TGS_SLOW_START;
}

static inline long long min(long long a, long long b) {
  return a < b ? a : b;
}

static inline long long max(long long a, long long b) {
  return a > b ? a : b;
}

static const long long update_rate_limit(int *p_state, CUdevice device,
                                         double recv_rate, double max_rate,
                                         double *p_max_rate,
                                         double threshold) {
  const static long long UPPER_LIMIT = 100000000000000LL;
  static int sign = 0;
  const double denominator = fabs(max_rate) < 1e-12 ? 1. : fabs(max_rate);
  double delta = fabs(recv_rate - max_rate) / denominator;

  long long rate_limit = g_rate_limit[device];

  switch (*p_state)
  {
  case TGS_SLOW_START:
    if (delta <= threshold) {
      rate_limit = min(rate_limit * 1.5 + 1, min(UPPER_LIMIT, max(3 * g_current_rate[device], (long long)(1ll << 40))));
    }
    else {
      rate_limit = rate_limit / 1.5;
      sign = -1;
      *p_state = TGS_CONGESTION_AVOIDANCE;
    }
    break;

  case TGS_CONGESTION_AVOIDANCE:
    if ((sign == -1 && delta <= threshold) || (sign == 1 && delta < threshold)) {
      rate_limit += max(max_rate * 0.00025, RATE_MIN);
      rate_limit = min(rate_limit, min(UPPER_LIMIT, max(3 * g_current_rate[device], (long long)65536LL * 65536LL)));
      sign = 1;
    }
    else {
      rate_limit -= max(rate_limit * 0.08, RATE_MIN);
      rate_limit = min(rate_limit, min(UPPER_LIMIT, max(3 * g_current_rate[device], (long long)65536LL * 65536LL)));
      sign = -1;
    }

    if (delta >= 3. * threshold) {
      *p_state = TGS_SLOW_START;
      rate_limit /= 10;
    }
    break;
  }

  static int max_diff_counter = 0;
  if (delta >= 4. * threshold) {
    ++max_diff_counter;
    if (max_diff_counter >= 20) {
      *p_max_rate *= 0.8;
      max_diff_counter = 0;
      *p_state = TGS_SLOW_START;
      rate_limit /= 2;
    }
  }
  else {
    max_diff_counter = 0;
  }

  rate_limit = (rate_limit <= 0) ? 0 : rate_limit;

  g_rate_limit[device] = rate_limit;
  return rate_limit;
}


static void *memory_transfer_routine(CUdevice device) {
  if (list_head == NULL)
    init_list();
  CUresult ret;
  const char *cuda_err_string = NULL;

  CUcontext cuContext;
  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuDevicePrimaryCtxRetain, &cuContext, device);
  if (unlikely(ret)) {
    LOGGER(FATAL, "cuDevicePrimaryCtxRetain error %s",
           cuda_error((CUresult)ret, &cuda_err_string));
  }

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxSetCurrent, cuContext);
  if (unlikely(ret)) {
    LOGGER(FATAL, "cuCtxSetCurrent error %s",
           cuda_error((CUresult)ret, &cuda_err_string));
  }

  pthread_mutex_lock(&g_map_mutex);

  for (struct MemRange *it = list_head->successor; it; it = it->successor)
    if (it->device == device) {
      ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemPrefetchAsync, it->devPtr, it->count, it->device, NULL);
      if (ret != CUDA_SUCCESS) {
        const char *error_name = NULL, *error_reason = NULL;
        CUDA_ENTRY_CALL(cuda_library_entry, cuGetErrorName, ret, &error_name);
        CUDA_ENTRY_CALL(cuda_library_entry, cuGetErrorString, ret, &error_reason);
        continue;
      }
    }

  pthread_mutex_unlock(&g_map_mutex);

  return NULL;
}


void activate_memory_transfer_routine(CUdevice device) {
  memory_transfer_routine(device);
}

inline double shift_window(double rate_window[], const int WINDOW_SIZE, double recv_rate) {
  double max_window_rate = 0;

  for (int i = WINDOW_SIZE-1; i > 0; --i) {
    double mean_rate = (rate_window[i] + rate_window[i-1]) / 2;
    max_window_rate = max_window_rate > mean_rate ? max_window_rate : mean_rate;
    rate_window[i] = rate_window[i-1];
  }
  rate_window[0] = recv_rate;

  return max_window_rate;
}

static inline double relative_delta(double value, double baseline) {
  if (fabs(baseline) < 1e-12) {
    if (fabs(value) < 1e-12)
      return 0.;
    return value > 0. ? INFINITY : -INFINITY;
  }
  return (value - baseline) / baseline;
}

static inline double wall_time_seconds(void) {
  struct timespec now;
  clock_gettime(CLOCK_REALTIME, &now);
  return (double)now.tv_sec + (double)now.tv_nsec / 1000000000.;
}

static void *limit_manager(void *v_device) {
  const CUdevice device = (uintptr_t)v_device;
  const int MAXLINE = 4096;
  const static long long UPPER_LIMIT = 100000000000000LL;
  const double alpha = 0.0;
  int listenfd, connfd, ret;
  socklen_t clientlen;
  struct sockaddr_storage clientaddr;
  char client_hostname[MAXLINE], client_port[MAXLINE];

  listenfd = open_listenfd(device);
  clientlen = sizeof(struct sockaddr_storage);


  while (1) {
    if ((connfd = accept(listenfd, (struct sockaddr *)&clientaddr, &clientlen)) < 0)
      LOGGER(FATAL, "accept error\n");
    if ((ret = getnameinfo((const struct sockaddr *)&clientaddr, clientlen, client_hostname, MAXLINE,
                           client_port, MAXLINE, 0)) != 0)
      LOGGER(FATAL, "getnameinfo error: %s\n", gai_strerror(ret));

    double max_rate = -1;
    if (rio_readn(connfd, (void *)&max_rate, sizeof(double)) != sizeof(double)) {
      continue;
    }

    g_rate_limit[device] = 0;
    g_rate_control_flag[device] = 1;

    double recv_rate = 1.;
    long long cnt = 0;
    int state = -1;

    const int WINDOW_SIZE = 5;
    const int PREWARM_TIME = 0;
    const int PROFILE_TIME = 5;
    const long long limit_initializer = (long long)event_be_parse_u64(
        "TGS_LIMIT_INITIALIZER", (uint64_t)LIMIT_INITIALIZER, 1ULL,
        100000000000000ULL);
    const double reprofile_threshold = (double)event_be_parse_u64(
        "TGS_REPROFILE_THRESHOLD_PCT", 20ULL, 1ULL, 10000ULL) / 100.0;
    const double rate_threshold = (double)event_be_parse_u64(
        "TGS_RATE_THRESHOLD_PCT", 3ULL, 1ULL, 100ULL) / 100.0;
    const char *control_audit_env = lilt_getenv("TGS_CONTROL_AUDIT");
    const int control_audit = control_audit_env != NULL &&
                              strcmp(control_audit_env, "0") != 0;
    double rate_window[WINDOW_SIZE];
    int profile_round = 0;

profile:
    /* Keep profiling bounded: five samples at the HP's 5-second period. */
    ++profile_round;
    const double profile_start_s = wall_time_seconds();
    fprintf(stderr,
            "[TGS][LP] PROFILE_START round=%d epoch_s=%.6f samples=%d\n",
            profile_round, profile_start_s, PROFILE_TIME);
    memset(rate_window, 0, sizeof(rate_window));
    /* Upstream TGS only clears the BE rate when the connection is first
     * established. Preserve the learned allowance during later re-profiles;
     * clearing it here made each Poisson-induced re-profile a 25 s blackout. */
    if (profile_round == 1) {
      g_rate_limit[device] = 0;
    }
    cnt = 0;
    state = -1;
    for (int t = 1; t <= PREWARM_TIME + PROFILE_TIME; ++t) {
      double recv_counter = -1;
      ssize_t n = rio_readn(connfd, (void *)&recv_counter, sizeof(double));
      if (n != sizeof(double)) {
        break;
      }
      if (t <= PREWARM_TIME) {
        continue;
      }

      recv_counter = recv_counter >= 1. ? recv_counter : 1.;
      recv_rate = alpha * recv_rate + (1 - alpha) * recv_counter;
      double max_window_rate = shift_window(rate_window, WINDOW_SIZE, recv_rate);
      double max_delta = relative_delta(max_window_rate, max_rate);

      if (max_delta >= -0.05 && max_delta <= 0.05) {
        max_rate = max_rate > max_window_rate ? max_rate : max_window_rate;
      }
      else {
        if (max_delta > 0.05) {
          double new_max_rate = max_window_rate * 0.975;
          max_rate = max_rate > new_max_rate ? max_rate : new_max_rate;
        }
        else {
          double new_max_rate = max_window_rate * 1.025;
          max_rate = max_rate < new_max_rate ? max_rate : new_max_rate;
        }
      }
    }
    max_rate = max_rate >= 1. ? max_rate : 1.;
    const double profile_end_s = wall_time_seconds();
    fprintf(stderr,
            "[TGS][LP] PROFILE_END round=%d epoch_s=%.6f duration_s=%.6f "
            "max_rate=%.6f\n",
            profile_round, profile_end_s, profile_end_s - profile_start_s,
            max_rate);
    fprintf(stderr, "profile max rate: %.6f\n", max_rate);

    while (1) {
      double recv_counter = -1;
      ssize_t n = rio_readn(connfd, (void *)&recv_counter, sizeof(double));
      if (n != sizeof(double)) {
        if (n)
          LOGGER(4, "readn error: receive %d byte\n", (int)n);
        break;
      }

      recv_counter = recv_counter >= 1. ? recv_counter : 1.;
      recv_rate = alpha * recv_rate + (1 - alpha) * recv_counter;
      double max_window_rate = shift_window(rate_window, WINDOW_SIZE, recv_rate);
      double max_delta = relative_delta(max_window_rate, max_rate);

      if (max_delta >= -0.1 && max_delta <= 0.1)
        max_rate = max_rate > max_window_rate ? max_rate : max_window_rate;
      else if (max_delta > reprofile_threshold ||
               max_delta < -reprofile_threshold) {
        if (max_delta > 0.2) {
          double new_max_rate = max_window_rate * 0.975;
          max_rate = max_rate > new_max_rate ? max_rate : new_max_rate;
        }
        else {
          double new_max_rate = max_window_rate * 1.025;
          max_rate = max_rate < new_max_rate ? max_rate : new_max_rate;
        }
        fprintf(stderr,
                "[TGS][LP] REPROFILE_TRIGGER epoch_s=%.6f observed_rate=%.6f "
                "max_rate=%.6f relative_delta=%.6f\n",
                wall_time_seconds(), max_window_rate, max_rate, max_delta);
        fprintf(stderr, "change max rate: %lf\n", max_rate);
        goto profile;
      }

      ++cnt;
      if (cnt == 1) {
        init_rate_limit(limit_initializer, &g_rate_limit[device], &state);
        if (control_audit) {
          fprintf(stderr,
                  "[TGS][LP] CONTROL epoch_s=%.6f sample=%lld recv_rate=%.6f "
                  "max_rate=%.6f relative_delta=%.6f num_zero=-1 state=%d "
                  "rate_limit=%lld current_lp_rate=%lld\n",
                  wall_time_seconds(), cnt, recv_rate, max_rate, max_delta,
                  state, g_rate_limit[device], g_current_rate[device]);
        }
        continue;
      }

      int num_zero = 0;
      for(int i = 0; i < WINDOW_SIZE; ++i){
        if(rate_window[i] < 1000)
          ++num_zero;
      }

      long long rate_limit;
      if(num_zero <= WINDOW_SIZE / 5 * 2 || cnt < 15){
        if(num_zero == 2 && cnt > 15){
          rate_limit = limit_initializer;
          init_rate_limit(rate_limit, &g_rate_limit[device], &state);
        }
        else
          rate_limit = update_rate_limit(
              &state, device, recv_rate, (double)max_rate, &max_rate,
              rate_threshold);
      }
      else{
        rate_limit = min(UPPER_LIMIT, max(3 * g_current_rate[device], (long long)65536LL * 65536LL));
        init_rate_limit(rate_limit, &g_rate_limit[device], &state);
      }

      if (control_audit) {
        fprintf(stderr,
                "[TGS][LP] CONTROL epoch_s=%.6f sample=%lld recv_rate=%.6f "
                "max_rate=%.6f relative_delta=%.6f num_zero=%d state=%d "
                "rate_limit=%lld current_lp_rate=%lld\n",
                wall_time_seconds(), cnt, recv_rate, max_rate, max_delta,
                num_zero, state, rate_limit, g_current_rate[device]);
      }

    }
    if ((ret = close(connfd)) < 0)
      LOGGER(FATAL, "close error\n");

    g_rate_limit[device] = 0;
    const char *disable_disconnect_prefetch =
        lilt_getenv("TGS_DISABLE_DISCONNECT_PREFETCH");
    if (disable_disconnect_prefetch == NULL ||
        strcmp(disable_disconnect_prefetch, "1") != 0) {
      activate_memory_transfer_routine(device);
    }
    g_rate_control_flag[device] = 0;
  }
}

static int lilt_set_cpu_affinity(pthread_t thread_id, int core_id) {
    int ret;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    ret = pthread_setaffinity_np(thread_id, sizeof(cpuset), &cpuset);
    if (ret != 0) {
        fprintf(stderr, "failed to set cpu affinity, core_id=%d", core_id);
        return -1;
    } else {
        ret = pthread_getaffinity_np(thread_id, sizeof(cpuset), &cpuset);
        if (ret != 0) {
            fprintf(stderr, "failed to get cpu affinity");
            return -1;
        } else {
            fprintf(stderr, "set returned by pthread_getaffinity_np() contained:");
            int cnt = 0, cpu_in_set = 0;
            for (int i = 0; i < CPU_SETSIZE; i++) {
                if (CPU_ISSET(i, &cpuset)) {
                    cnt++;
                    cpu_in_set = i;
                    fprintf(stderr, "  cpu=%d", i);
                }
            }
            // this should not happen though
            if (cnt != 1 || cpu_in_set != core_id) {
                fprintf(stderr, "failed to set cpu affinity with cpu=%d", core_id);
                return -1;
            }
        }
    }
    return 0;
}

static void activate_limit_manager(CUdevice device) {
  pthread_t tid;

  pthread_create(&tid, NULL, limit_manager, (void *)(uintptr_t)device);
  lilt_set_cpu_affinity(tid, g_gpu_id[device]);

#ifdef __APPLE__
  pthread_setname_np("limit_manager");
#else
  pthread_setname_np(tid, "limit_manager");
#endif
}


static inline int launch_test(const long long kernel_size, const CUdevice device) {
  return g_rate_control_flag[device] == 1 && g_rate_counter[device] > g_rate_limit[device];
}


static inline void rate_limiter(const long long kernel_size) {
  CUdevice device = 0;
  const CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &device);
  if (ret != CUDA_SUCCESS) {
    fprintf(stderr, "cuCtxGetDevice error\n");
  }

  if (!g_active_gpu[device])
    initialization(device);

  while (launch_test(kernel_size, device))
    nanosleep(&g_cycle, NULL);
  __sync_add_and_fetch_8(&g_rate_counter[device], kernel_size);
}


static void *rate_watcher(void *v_device) {
  const CUdevice device = (uintptr_t)v_device;
  const unsigned long duration = 50;
  const struct timespec unit_time = {
    .tv_sec = duration / 1000,
    .tv_nsec = duration % 1000 * MILLISEC,
  };
  g_rate_counter[device] = 0;
  while (1) {
    nanosleep(&unit_time, NULL);

    long long current_rate = g_rate_counter[device];
    g_rate_counter[device] = 0;
    g_current_rate[device] = current_rate;
  }
  return NULL;
}


static void activate_rate_watcher(CUdevice device) {
  pthread_t tid;

  pthread_create(&tid, NULL, rate_watcher, (void *)(uintptr_t)device);
  lilt_set_cpu_affinity(tid, g_gpu_id[device]);

#ifdef __APPLE__
  pthread_setname_np("rate_watcher");
#else
  pthread_setname_np(tid, "rate_watcher");
#endif
}


static inline void initialization(const CUdevice device) {
  g_active_gpu[device] = 1;

  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuDeviceGetUuid, &g_uuid[device], device);
  if (ret != CUDA_SUCCESS) {
    LOGGER(FATAL, "cuDeviceGetUuid error\n");
  }

  int gpu_id = 0;
  for (int i = 0; i < 16; ++i) {
    gpu_id += (int)g_uuid[device].bytes[i];
  }
  gpu_id = (gpu_id % 8 + 8) % 8;
  g_gpu_id[device] = gpu_id;

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxResetPersistingL2Cache);
  if (ret != CUDA_SUCCESS) {
    fprintf(stderr, "cuCtxResetPersistingL2Cache error\n");
  }
  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxSetLimit, CU_LIMIT_PERSISTING_L2_CACHE_SIZE, 0);
  if (ret != CUDA_SUCCESS) {
    fprintf(stderr, "cuCtxSetLimit error, ret=%d\n", (int)ret);
  }

  activate_rate_watcher(device);
  activate_limit_manager(device);
}

/** hijack entrypoint */
CUresult cuDriverGetVersion(int *driverVersion) {
  CUresult ret;

  load_necessary_data();

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuDriverGetVersion, driverVersion);
  return ret;
}

CUresult cuInit(unsigned int flag) {
  CUresult ret;

  load_necessary_data();

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuInit, flag);
  return ret;
}

CUresult cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize,
                           unsigned int flags) {
  CUresult ret;

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAllocManaged, dptr, bytesize,
                        flags);

  CUdevice device;
  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &device);
  if (ret != CUDA_SUCCESS) {
    return ret;
  }

  if (ret == CUDA_SUCCESS)
    allocate_mem(*dptr, bytesize, device);

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAdvise, *dptr, bytesize, CU_MEM_ADVISE_SET_ACCESSED_BY,
                         device);

  if (ret != CUDA_SUCCESS) {
    return ret;
  }

  return ret;
}

CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize) {
  CUresult ret;
  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAllocManaged, dptr, bytesize,
                        1);

  CUdevice device;
  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &device);
  if (ret != CUDA_SUCCESS) {
    return ret;
  }

  if (ret == CUDA_SUCCESS)
    allocate_mem(*dptr, bytesize, device);

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAdvise, *dptr, bytesize, CU_MEM_ADVISE_SET_ACCESSED_BY,
                         device);

  if (ret != CUDA_SUCCESS) {
    return ret;
  }

  return ret;
}

CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
  CUresult ret;

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAllocManaged, dptr, bytesize,
                        1);

  CUdevice device;
  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &device);
  if (ret != CUDA_SUCCESS) {
    return ret;
  }

  if (ret == CUDA_SUCCESS)
    allocate_mem(*dptr, bytesize, device);

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAdvise, *dptr, bytesize, CU_MEM_ADVISE_SET_ACCESSED_BY,
                         device);


  if (ret != CUDA_SUCCESS) {
    return ret;
  }

  return ret;
}

CUresult cuMemAllocPitch_v2(CUdeviceptr *dptr, size_t *pPitch,
                            size_t WidthInBytes, size_t Height,
                            unsigned int ElementSizeBytes) {
  *pPitch = ROUND_UP(WidthInBytes, 128);
  size_t bytesize = ROUND_UP(*pPitch * Height, ElementSizeBytes);
  CUresult ret;


  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAllocManaged, dptr, bytesize,
                        1);


  CUdevice device;
  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &device);
  if (ret != CUDA_SUCCESS) {
    return ret;
  }

  if (ret == CUDA_SUCCESS)
    allocate_mem(*dptr, bytesize, device);

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAdvise, *dptr, bytesize, CU_MEM_ADVISE_SET_ACCESSED_BY,
                         device);


  if (ret != CUDA_SUCCESS) {
    return ret;
  }

  return ret;
}

CUresult cuMemAllocPitch(CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes,
                         size_t Height, unsigned int ElementSizeBytes) {
  *pPitch = ROUND_UP(WidthInBytes, 128);
  size_t bytesize = ROUND_UP(*pPitch * Height, ElementSizeBytes);
  CUresult ret;

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAllocManaged, dptr, bytesize,
                        1);

  CUdevice device;
  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &device);
  if (ret != CUDA_SUCCESS) {
    return ret;
  }

  if (ret == CUDA_SUCCESS)
    allocate_mem(*dptr, bytesize, device);

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAdvise, *dptr, bytesize, CU_MEM_ADVISE_SET_ACCESSED_BY,
                         device);


  if (ret != CUDA_SUCCESS) {
    return ret;
  }

  return ret;
}


CUresult cuMemFree_v2(CUdeviceptr dptr) {
  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemFree_v2, dptr);
  if (ret == CUDA_SUCCESS)
    delete_mem(dptr);
  return ret;
}


CUresult cuMemFree(CUdeviceptr dptr) {
  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemFree, dptr);
  if (ret == CUDA_SUCCESS)
    delete_mem(dptr);
  return ret;
}


CUresult cuArrayCreate_v2(CUarray *pHandle,
                          const CUDA_ARRAY_DESCRIPTOR *pAllocateArray) {
  CUresult ret;

  LOGGER(FATAL, "call cuArrayCreate_v2");

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuArrayCreate_v2, pHandle,
                        pAllocateArray);
  return ret;
}

CUresult cuArrayCreate(CUarray *pHandle,
                       const CUDA_ARRAY_DESCRIPTOR *pAllocateArray) {
  CUresult ret;

  LOGGER(FATAL, "call cuArrayCreate");

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuArrayCreate, pHandle,
                        pAllocateArray);
  return ret;
}

CUresult cuArray3DCreate_v2(CUarray *pHandle,
                            const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray) {
  CUresult ret;

  LOGGER(FATAL, "call cuArray3DCreate_v2");

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuArray3DCreate_v2, pHandle,
                        pAllocateArray);
  return ret;
}

CUresult cuArray3DCreate(CUarray *pHandle,
                         const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray) {
  CUresult ret;

  LOGGER(FATAL, "call cuArray3DCreate");

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuArray3DCreate, pHandle,
                        pAllocateArray);
  return ret;
}

CUresult cuMipmappedArrayCreate(
    CUmipmappedArray *pHandle,
    const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc,
    unsigned int numMipmapLevels) {
  CUresult ret;

  LOGGER(FATAL, "call cuMipmappedArrayCreate");

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMipmappedArrayCreate, pHandle,
                        pMipmappedArrayDesc, numMipmapLevels);
  return ret;
}

CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev) {
  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuDeviceTotalMem_v2, bytes, dev);
  if (ret != CUDA_SUCCESS)
    return ret;
  *bytes = (*bytes > g_spare_memory) ? (*bytes - g_spare_memory) : 0;
  return ret;
}

CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuDeviceTotalMem, bytes, dev);
  if (ret != CUDA_SUCCESS)
    return ret;
  *bytes = (*bytes > g_spare_memory) ? (*bytes - g_spare_memory) : 0;
  return ret;
}

CUresult cuMemGetInfo_v2(size_t *free, size_t *total) {
  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemGetInfo_v2, free, total);
  if (ret != CUDA_SUCCESS)
    return ret;
  *total = (*total > g_spare_memory) ? (*total - g_spare_memory) : 0;
  *free = (*total > g_spare_memory + g_used_memory) ? (*total - g_spare_memory - g_used_memory) : 0;
  return ret;
}

CUresult cuMemGetInfo(size_t *free, size_t *total) {
  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemGetInfo, free, total);
  if (ret != CUDA_SUCCESS)
    return ret;
  *total = (*total > g_spare_memory) ? (*total - g_spare_memory) : 0;
  *free = (*total > g_spare_memory + g_used_memory) ? (*total - g_spare_memory - g_used_memory) : 0;
  return ret;
}

CUresult cuStreamCreate(CUstream *phStream, unsigned int Flags) {
  int leastPriority, greatestPriority;
  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetStreamPriorityRange, &leastPriority, &greatestPriority);
  if (ret == CUDA_SUCCESS) {
    ret = CUDA_ENTRY_CALL(cuda_library_entry, cuStreamCreateWithPriority, phStream, Flags, leastPriority);
    if (ret == CUDA_SUCCESS) {
      return ret;
    }
    else {
      fprintf(stderr, "\n[ERROR] cuStreamCreateWithPriority failed\n");
    }
  }
  else {
    fprintf(stderr, "\n[ERROR] cuCtxGetStreamPriorityRange failed\n");
  }

  return CUDA_ENTRY_CALL(cuda_library_entry, cuStreamCreate, phStream, Flags);
}

CUresult cuStreamCreateWithPriority(CUstream *phStream, unsigned int flags,
                                    int priority) {
  int leastPriority, greatestPriority;
  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetStreamPriorityRange, &leastPriority, &greatestPriority);
  if (ret == CUDA_SUCCESS) {
    greatestPriority = (leastPriority + greatestPriority) / 2;
    if (priority < greatestPriority) {
      priority = greatestPriority;
    }
  }
  else {
    fprintf(stderr, "\n[ERROR] cuCtxGetStreamPriorityRange failed\n");
  }
  return CUDA_ENTRY_CALL(cuda_library_entry, cuStreamCreateWithPriority,
                         phStream, flags, priority);
}

CUresult cuLaunchKernel_ptsz(CUfunction f, unsigned int gridDimX,
                             unsigned int gridDimY, unsigned int gridDimZ,
                             unsigned int blockDimX, unsigned int blockDimY,
                             unsigned int blockDimZ,
                             unsigned int sharedMemBytes, CUstream hStream,
                             void **kernelParams, void **extra) {
  lilt_before_kernel_launch_ex(
      f, (uint64_t)gridDimX * gridDimY * gridDimZ, hStream, 1);
  CUresult result = CUDA_ENTRY_CALL(
      cuda_library_entry, cuLaunchKernel_ptsz, f, gridDimX, gridDimY, gridDimZ,
      blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams,
      extra);
  lilt_after_kernel_launch(hStream, 1, result);
  return result;
}

CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX,
                        unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY,
                        unsigned int blockDimZ, unsigned int sharedMemBytes,
                        CUstream hStream, void **kernelParams, void **extra) {
  lilt_before_kernel_launch_ex(
      f, (uint64_t)gridDimX * gridDimY * gridDimZ, hStream, 0);
  CUresult result = CUDA_ENTRY_CALL(
      cuda_library_entry, cuLaunchKernel, f, gridDimX, gridDimY, gridDimZ,
      blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams,
      extra);
  lilt_after_kernel_launch(hStream, 0, result);
  return result;
}

CUresult cuLaunch(CUfunction f) {
  lilt_before_kernel_launch_ex(f, 1, NULL, 0);
  CUresult result = CUDA_ENTRY_CALL(cuda_library_entry, cuLaunch, f);
  lilt_after_kernel_launch(NULL, 0, result);
  return result;
}

CUresult cuLaunchCooperativeKernel_ptsz(
    CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
    void **kernelParams) {
  lilt_before_kernel_launch_ex(
      f, (uint64_t)gridDimX * gridDimY * gridDimZ, hStream, 1);
  CUresult result = CUDA_ENTRY_CALL(
      cuda_library_entry, cuLaunchCooperativeKernel_ptsz, f, gridDimX,
      gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes,
      hStream, kernelParams);
  lilt_after_kernel_launch(hStream, 1, result);
  return result;
}

CUresult cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX,
                                   unsigned int gridDimY, unsigned int gridDimZ,
                                   unsigned int blockDimX,
                                   unsigned int blockDimY,
                                   unsigned int blockDimZ,
                                   unsigned int sharedMemBytes,
                                   CUstream hStream, void **kernelParams) {
  lilt_before_kernel_launch_ex(
      f, (uint64_t)gridDimX * gridDimY * gridDimZ, hStream, 0);
  CUresult result = CUDA_ENTRY_CALL(
      cuda_library_entry, cuLaunchCooperativeKernel, f, gridDimX, gridDimY,
      gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream,
      kernelParams);
  lilt_after_kernel_launch(hStream, 0, result);
  return result;
}

CUresult cuLaunchGrid(CUfunction f, int grid_width, int grid_height) {
  lilt_before_kernel_launch_ex(f, (uint64_t)grid_width * grid_height, NULL, 0);
  CUresult result = CUDA_ENTRY_CALL(cuda_library_entry, cuLaunchGrid, f,
                                    grid_width, grid_height);
  lilt_after_kernel_launch(NULL, 0, result);
  return result;
}

CUresult cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height,
                           CUstream hStream) {
  lilt_before_kernel_launch_ex(f, (uint64_t)grid_width * grid_height, hStream,
                              0);
  CUresult result = CUDA_ENTRY_CALL(cuda_library_entry, cuLaunchGridAsync, f,
                                    grid_width, grid_height, hStream);
  lilt_after_kernel_launch(hStream, 0, result);
  return result;
}

CUresult cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z) {
  while (!CAS(&g_block_locker, 0, 1));

  g_block_x = x;
  g_block_y = y;
  g_block_z = z;

  LOGGER(5, "Set block shape: %d, %d, %d", x, y, z);

  while (!CAS(&g_block_locker, 1, 0));
  return CUDA_ENTRY_CALL(cuda_library_entry, cuFuncSetBlockShape, hfunc, x, y,
                         z);
}

CUresult cuMemcpy_ptds(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpy_ptds, dst, src,
                         ByteCount);
}

CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpy, dst, src, ByteCount);
}

CUresult cuMemcpyAsync_ptsz(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount,
                            CUstream hStream) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyAsync_ptsz, dst, src,
                         ByteCount, hStream);
}

CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount,
                       CUstream hStream) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyAsync, dst, src, ByteCount,
                         hStream);
}

CUresult cuMemcpyPeer_ptds(CUdeviceptr dstDevice, CUcontext dstContext,
                           CUdeviceptr srcDevice, CUcontext srcContext,
                           size_t ByteCount) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyPeer_ptds, dstDevice,
                         dstContext, srcDevice, srcContext, ByteCount);
}

CUresult cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext,
                      CUdeviceptr srcDevice, CUcontext srcContext,
                      size_t ByteCount) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyPeer, dstDevice,
                         dstContext, srcDevice, srcContext, ByteCount);
}

CUresult cuMemcpyPeerAsync_ptsz(CUdeviceptr dstDevice, CUcontext dstContext,
                                CUdeviceptr srcDevice, CUcontext srcContext,
                                size_t ByteCount, CUstream hStream) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyPeerAsync_ptsz, dstDevice,
                         dstContext, srcDevice, srcContext, ByteCount, hStream);
}

CUresult cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext,
                           CUdeviceptr srcDevice, CUcontext srcContext,
                           size_t ByteCount, CUstream hStream) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyPeerAsync, dstDevice,
                         dstContext, srcDevice, srcContext, ByteCount, hStream);
}

CUresult cuMemcpyHtoD_v2_ptds(CUdeviceptr dstDevice, const void *srcHost,
                              size_t ByteCount) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyHtoD_v2_ptds, dstDevice,
                         srcHost, ByteCount);
}

CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost,
                         size_t ByteCount) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyHtoD_v2, dstDevice,
                         srcHost, ByteCount);
}

CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost,
                      size_t ByteCount) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyHtoD, dstDevice, srcHost,
                         ByteCount);
}

CUresult cuMemcpyHtoDAsync_v2_ptsz(CUdeviceptr dstDevice, const void *srcHost,
                                   size_t ByteCount, CUstream hStream) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyHtoDAsync_v2_ptsz,
                         dstDevice, srcHost, ByteCount, hStream);
}

CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void *srcHost,
                              size_t ByteCount, CUstream hStream) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyHtoDAsync_v2, dstDevice,
                         srcHost, ByteCount, hStream);
}

CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void *srcHost,
                           size_t ByteCount, CUstream hStream) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyHtoDAsync, dstDevice,
                         srcHost, ByteCount, hStream);
}

CUresult cuMemcpyDtoH_v2_ptds(void *dstHost, CUdeviceptr srcDevice,
                              size_t ByteCount) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyDtoH_v2_ptds, dstHost,
                         srcDevice, ByteCount);
}

CUresult cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice,
                         size_t ByteCount) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyDtoH_v2, dstHost,
                         srcDevice, ByteCount);
}

CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyDtoH, dstHost, srcDevice,
                         ByteCount);
}

CUresult cuMemcpyDtoHAsync_v2_ptsz(void *dstHost, CUdeviceptr srcDevice,
                                   size_t ByteCount, CUstream hStream) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyDtoHAsync_v2_ptsz, dstHost,
                         srcDevice, ByteCount, hStream);
}

CUresult cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice,
                              size_t ByteCount, CUstream hStream) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyDtoHAsync_v2, dstHost,
                         srcDevice, ByteCount, hStream);
}

CUresult cuMemcpyDtoHAsync(void *dstHost, CUdeviceptr srcDevice,
                           size_t ByteCount, CUstream hStream) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyDtoHAsync, dstHost,
                         srcDevice, ByteCount, hStream);
}

CUresult cuMemcpyDtoD_v2_ptds(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                              size_t ByteCount) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyDtoD_v2_ptds, dstDevice,
                         srcDevice, ByteCount);
}

CUresult cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                         size_t ByteCount) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyDtoD_v2, dstDevice,
                         srcDevice, ByteCount);
}

CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                      size_t ByteCount) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyDtoD, dstDevice, srcDevice,
                         ByteCount);
}

CUresult cuMemcpyDtoDAsync_v2_ptsz(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                                   size_t ByteCount, CUstream hStream) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyDtoDAsync_v2_ptsz,
                         dstDevice, srcDevice, ByteCount, hStream);
}

CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                              size_t ByteCount, CUstream hStream) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyDtoDAsync_v2, dstDevice,
                         srcDevice, ByteCount, hStream);
}

CUresult cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                           size_t ByteCount, CUstream hStream) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyDtoDAsync, dstDevice,
                         srcDevice, ByteCount, hStream);
}

CUresult cuMemcpy2DUnaligned_v2_ptds(const CUDA_MEMCPY2D *pCopy) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpy2DUnaligned_v2_ptds,
                         pCopy);
}

CUresult cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D *pCopy) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpy2DUnaligned_v2, pCopy);
}

CUresult cuMemcpy2DUnaligned(const CUDA_MEMCPY2D *pCopy) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpy2DUnaligned, pCopy);
}

CUresult cuMemcpy2DAsync_v2_ptsz(const CUDA_MEMCPY2D *pCopy, CUstream hStream) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpy2DAsync_v2_ptsz, pCopy,
                         hStream);
}

CUresult cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D *pCopy, CUstream hStream) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpy2DAsync_v2, pCopy,
                         hStream);
}

CUresult cuMemcpy2DAsync(const CUDA_MEMCPY2D *pCopy, CUstream hStream) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpy2DAsync, pCopy, hStream);
}

CUresult cuMemcpy3D_v2_ptds(const CUDA_MEMCPY3D *pCopy) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpy3D_v2_ptds, pCopy);
}

CUresult cuMemcpy3D_v2(const CUDA_MEMCPY3D *pCopy) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpy3D_v2, pCopy);
}

CUresult cuMemcpy3D(const CUDA_MEMCPY3D *pCopy) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpy3D, pCopy);
}

CUresult cuMemcpy3DAsync_v2_ptsz(const CUDA_MEMCPY3D *pCopy, CUstream hStream) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpy3DAsync_v2_ptsz, pCopy,
                         hStream);
}

CUresult cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D *pCopy, CUstream hStream) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpy3DAsync_v2, pCopy,
                         hStream);
}

CUresult cuMemcpy3DAsync(const CUDA_MEMCPY3D *pCopy, CUstream hStream) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpy3DAsync, pCopy, hStream);
}

CUresult cuMemcpy3DPeer_ptds(const CUDA_MEMCPY3D_PEER *pCopy) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpy3DPeer_ptds, pCopy);
}

CUresult cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER *pCopy) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpy3DPeer, pCopy);
}

CUresult cuMemcpy3DPeerAsync_ptsz(const CUDA_MEMCPY3D_PEER *pCopy,
                                  CUstream hStream) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpy3DPeerAsync_ptsz, pCopy,
                         hStream);
}

CUresult cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER *pCopy,
                             CUstream hStream) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpy3DPeerAsync, pCopy,
                         hStream);
}

/*
 *  Context Management
 */

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev) {
  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxCreate_v2, pctx, flags, dev);
  return ret;
}

CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev) {
  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxCreate, pctx, flags, dev);
  return ret;
}

CUresult cuCtxSetCurrent(CUcontext ctx) {
  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxSetCurrent, ctx);
  return ret;
}

CUresult cuCtxPushCurrent_v2(CUcontext ctx) {
  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxPushCurrent_v2, ctx);
  return ret;
}

CUresult cuCtxPushCurrent(CUcontext ctx) {
  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxPushCurrent, ctx);
  return ret;
}

CUresult cuCtxDestroy_v2(CUcontext ctx) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuCtxDestroy_v2, ctx);
}

CUresult cuCtxDestroy(CUcontext ctx) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuCtxDestroy, ctx);
}

CUresult cuCtxPopCurrent_v2(CUcontext *pctx) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuCtxPopCurrent_v2, pctx);
}

CUresult cuCtxPopCurrent(CUcontext *pctx) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuCtxPopCurrent, pctx);
}

/*
 *  Primary Context Management
 */

CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuDevicePrimaryCtxRetain, pctx,
                         dev);
}

CUresult cuDevicePrimaryCtxRelease(CUdevice dev) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuDevicePrimaryCtxRelease, dev);
}
