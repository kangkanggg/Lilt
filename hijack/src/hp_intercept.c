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
#include <math.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netdb.h>
#include <sys/un.h>
#include <stdint.h>
#include <sys/syscall.h>
#include <sched.h>

#include "../include/cuda-helper.h"
#include "../include/activity_tracker.h"
#include "../include/hijack.h"
#include "../include/nvml-helper.h"

extern entry_t cuda_library_entry[];
extern entry_t nvml_library_entry[];
extern char pid_path[];

typedef void (*atomic_fn_ptr)(int, void *);

#define GPU_MAX_NUM 8

static int g_block_x = 1, g_block_y = 1, g_block_z = 1;

static long long g_current_rate[GPU_MAX_NUM] = {};
static long long g_rate_counter[GPU_MAX_NUM] = {};
static int g_active_gpu[GPU_MAX_NUM] = {};
static CUuuid g_uuid[GPU_MAX_NUM];
static int g_gpu_id[GPU_MAX_NUM];

const size_t g_spare_memory = 1ull << 30;

static void activate_rate_watcher();
static void *rate_watcher(void *);
static void rate_estimator(const long long);
static uint32_t g_block_locker = 0;
static unsigned long long g_driver_launch_calls;
static unsigned long long g_driver_launch_blocks;

#define EVENT_MAX_STREAMS 64
#define EVENT_BLOCKING_SYNC 1U
#define EVENT_DISABLE_TIMING 2U
#define EVENT_QUERY_INTERVAL_NS (20 * 1000)
#define EVENT_BLOCKING_QUIET_NS (20 * 1000)
#define EVENT_INTERVAL_NS_MAX (1000 * 1000 * 1000LL)

typedef enum {
  EVENT_COMPLETION_BLOCKING = 0,
  EVENT_COMPLETION_QUERY = 1,
  EVENT_COMPLETION_DEFERRED = 2,
} event_completion_mode_t;

typedef struct {
  CUstream stream;
  CUevent event;
  int active;
  int per_thread_default_stream;
  int deferred_record_pending;
  uint64_t generation;
} hp_event_slot_t;

typedef struct {
  int slot_index;
  CUstream stream;
  CUevent event;
  int per_thread_default_stream;
  int deferred_record_pending;
  uint64_t generation;
  CUresult result;
} hp_event_wait_snapshot_t;

static hp_event_slot_t g_hp_event_slots[EVENT_MAX_STREAMS];
static pthread_mutex_t g_hp_event_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_hp_event_cond = PTHREAD_COND_INITIALIZER;
static pthread_t g_hp_event_thread;
static CUcontext g_hp_event_context;
static int g_hp_event_thread_started;
static int g_hp_event_stop;
static int g_hp_state_active;
static unsigned int g_hp_launches_in_progress;
static unsigned long long g_hp_events_recorded;
static unsigned long long g_hp_event_queries;
static unsigned long long g_hp_event_record_failures;
static unsigned long long g_hp_active_publications;
static unsigned long long g_hp_idle_publications;
static unsigned long long g_hp_event_waits;
static unsigned long long g_hp_event_wait_completions;
static unsigned long long g_hp_event_wait_stale;
static unsigned long long g_hp_event_wait_failures;
static unsigned long long g_hp_event_settle_checks;
static uint64_t g_hp_event_record_generation;
static long long g_hp_event_query_interval_ns = EVENT_QUERY_INTERVAL_NS;
static long long g_hp_event_blocking_quiet_ns = EVENT_BLOCKING_QUIET_NS;
static event_completion_mode_t g_hp_event_completion_mode =
    EVENT_COMPLETION_DEFERRED;

static event_completion_mode_t event_completion_mode(void) {
  const char *configured = lilt_getenv("LILT_EVENT_COMPLETION_MODE");
  if (configured == NULL || configured[0] == '\0' ||
      strcmp(configured, "deferred") == 0) {
    return EVENT_COMPLETION_DEFERRED;
  }
  if (strcmp(configured, "blocking") == 0) {
    return EVENT_COMPLETION_BLOCKING;
  }
  if (strcmp(configured, "query") == 0) {
    return EVENT_COMPLETION_QUERY;
  }
  fprintf(stderr,
          "[EVENT][HP] invalid TGS_EVENT_COMPLETION_MODE=%s; "
          "using deferred\n",
          configured);
  return EVENT_COMPLETION_DEFERRED;
}

static const char *event_completion_mode_name(event_completion_mode_t mode) {
  if (mode == EVENT_COMPLETION_QUERY) {
    return "query";
  }
  return mode == EVENT_COMPLETION_DEFERRED ? "deferred" : "blocking";
}

static long long event_query_interval_ns(void) {
  const char *configured = lilt_getenv("LILT_EVENT_QUERY_INTERVAL_NS");
  if (configured == NULL || configured[0] == '\0') {
    return EVENT_QUERY_INTERVAL_NS;
  }

  char *end = NULL;
  errno = 0;
  long long value = strtoll(configured, &end, 10);
  if (errno != 0 || end == configured || *end != '\0' || value < 0 ||
      value > EVENT_INTERVAL_NS_MAX) {
    fprintf(stderr,
            "[EVENT][HP] invalid TGS_EVENT_QUERY_INTERVAL_NS=%s; "
            "using default %d ns\n",
            configured, EVENT_QUERY_INTERVAL_NS);
    return EVENT_QUERY_INTERVAL_NS;
  }
  return value;
}

static long long event_blocking_quiet_ns(void) {
  const char *configured = lilt_getenv("LILT_EVENT_BLOCKING_QUIET_NS");
  if (configured == NULL || configured[0] == '\0') {
    return EVENT_BLOCKING_QUIET_NS;
  }

  char *end = NULL;
  errno = 0;
  long long value = strtoll(configured, &end, 10);
  if (errno != 0 || end == configured || *end != '\0' || value < 0 ||
      value > EVENT_INTERVAL_NS_MAX) {
    fprintf(stderr,
            "[EVENT][HP] invalid TGS_EVENT_BLOCKING_QUIET_NS=%s; "
            "using default %d ns\n",
            configured, EVENT_BLOCKING_QUIET_NS);
    return EVENT_BLOCKING_QUIET_NS;
  }
  return value;
}

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

static void initialization(CUdevice);
static void event_hp_before_launch(void);
static void event_hp_after_launch(CUstream, int, CUresult);

void lilt_before_kernel_launch(uint64_t blocks) {
  __sync_fetch_and_add(&g_driver_launch_calls, 1);
  __sync_fetch_and_add(&g_driver_launch_blocks, blocks);
  if (lilt_event_passthrough_enabled()) {
    return;
  }
  if (lilt_event_policy_enabled()) {
    event_hp_before_launch();
  } else {
    rate_estimator((long long)blocks);
  }
}

void lilt_before_kernel_launch_ex(CUfunction function, uint64_t blocks,
                                 CUstream stream,
                                 int per_thread_default_stream) {
  (void)function;
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
    event_hp_before_launch();
  } else {
    rate_estimator((long long)blocks);
  }
}

void lilt_after_kernel_launch(CUstream stream, int per_thread_default_stream,
                             CUresult launch_result) {
  if (g_tgs_capture_bypass_after) {
    g_tgs_capture_bypass_after = 0;
    return;
  }
  if (lilt_event_policy_enabled() && !lilt_event_passthrough_enabled()) {
    event_hp_after_launch(stream, per_thread_default_stream, launch_result);
  }
}

void lilt_graphlet_boundary(void) {
  /* HP Graphs are tracked as one asynchronous launch and are never split. */
}

static int hp_any_active_event_locked(void) {
  for (int i = 0; i < EVENT_MAX_STREAMS; ++i) {
    if (g_hp_event_slots[i].active) {
      return 1;
    }
  }
  return 0;
}

static int hp_event_slot_locked(CUstream stream) {
  int empty = -1;
  for (int i = 0; i < EVENT_MAX_STREAMS; ++i) {
    if (g_hp_event_slots[i].event != NULL &&
        g_hp_event_slots[i].stream == stream) {
      return i;
    }
    if (empty < 0 && g_hp_event_slots[i].event == NULL) {
      empty = i;
    }
  }
  if (empty < 0) {
    return -1;
  }
  CUresult result = CUDA_ENTRY_CALL(cuda_library_entry, cuEventCreate,
                                    &g_hp_event_slots[empty].event,
                                    EVENT_BLOCKING_SYNC |
                                        EVENT_DISABLE_TIMING);
  if (result != CUDA_SUCCESS) {
    g_hp_event_slots[empty].event = NULL;
    return -1;
  }
  g_hp_event_slots[empty].stream = stream;
  return empty;
}

static void hp_publish_idle_locked(void) {
  if (g_hp_state_active && g_hp_launches_in_progress == 0 &&
      !hp_any_active_event_locked()) {
    lilt_event_hp_mark_idle();
    g_hp_state_active = 0;
    ++g_hp_idle_publications;
  }
}

static int hp_snapshot_active_events_locked(
    hp_event_wait_snapshot_t snapshots[EVENT_MAX_STREAMS]) {
  int count = 0;
  for (int i = 0; i < EVENT_MAX_STREAMS; ++i) {
    hp_event_slot_t *slot = &g_hp_event_slots[i];
    if (!slot->active) {
      continue;
    }
    snapshots[count].slot_index = i;
    snapshots[count].stream = slot->stream;
    snapshots[count].event = slot->event;
    snapshots[count].per_thread_default_stream =
        slot->per_thread_default_stream;
    snapshots[count].deferred_record_pending =
        slot->deferred_record_pending;
    snapshots[count].generation = slot->generation;
    snapshots[count].result = CUDA_SUCCESS;
    ++count;
  }
  return count;
}

static int hp_wait_for_stable_frontier(uint64_t *stable_generation) {
  uint64_t observed = __atomic_load_n(&g_hp_event_record_generation,
                                       __ATOMIC_ACQUIRE);
  if (g_hp_event_blocking_quiet_ns == 0) {
    *stable_generation = observed;
    return 1;
  }

  const struct timespec quiet_interval = {
      .tv_sec = g_hp_event_blocking_quiet_ns / (1000 * 1000 * 1000LL),
      .tv_nsec = g_hp_event_blocking_quiet_ns % (1000 * 1000 * 1000LL)};
  while (!__atomic_load_n(&g_hp_event_stop, __ATOMIC_ACQUIRE)) {
    struct timespec remaining = quiet_interval;
    while (nanosleep(&remaining, &remaining) != 0 && errno == EINTR) {
    }
    ++g_hp_event_settle_checks;
    uint64_t current = __atomic_load_n(&g_hp_event_record_generation,
                                        __ATOMIC_ACQUIRE);
    if (current == observed) {
      *stable_generation = current;
      return 1;
    }
    observed = current;
  }
  return 0;
}

static void hp_event_blocking_watcher_loop(int deferred_record) {
  hp_event_wait_snapshot_t snapshots[EVENT_MAX_STREAMS];

  while (1) {
    pthread_mutex_lock(&g_hp_event_lock);
    while (!g_hp_event_stop && !hp_any_active_event_locked()) {
      pthread_cond_wait(&g_hp_event_cond, &g_hp_event_lock);
    }
    if (g_hp_event_stop) {
      pthread_mutex_unlock(&g_hp_event_lock);
      break;
    }
    pthread_mutex_unlock(&g_hp_event_lock);

    uint64_t stable_generation;
    if (!hp_wait_for_stable_frontier(&stable_generation)) {
      break;
    }

    pthread_mutex_lock(&g_hp_event_lock);
    if (g_hp_event_stop) {
      pthread_mutex_unlock(&g_hp_event_lock);
      break;
    }
    if (__atomic_load_n(&g_hp_event_record_generation, __ATOMIC_ACQUIRE) !=
            stable_generation ||
        g_hp_launches_in_progress != 0) {
      pthread_mutex_unlock(&g_hp_event_lock);
      continue;
    }
    int snapshot_count = hp_snapshot_active_events_locked(snapshots);
    pthread_mutex_unlock(&g_hp_event_lock);

    /* Never wait in CUDA while holding g_hp_event_lock. A later launch may
     * re-record the same event while this thread is asleep; generation
     * validation below prevents that older completion from publishing IDLE. */
    for (int i = 0; i < snapshot_count; ++i) {
      hp_event_wait_snapshot_t *snapshot = &snapshots[i];
      int deferred_record_failed = 0;
      if (deferred_record && snapshot->deferred_record_pending) {
        if (snapshot->per_thread_default_stream) {
          snapshot->result = CUDA_ENTRY_CALL(cuda_library_entry,
                                              cuEventRecord_ptsz,
                                              snapshot->event,
                                              snapshot->stream);
        } else {
          snapshot->result = CUDA_ENTRY_CALL(cuda_library_entry,
                                              cuEventRecord,
                                              snapshot->event,
                                              snapshot->stream);
        }
        if (snapshot->result == CUDA_SUCCESS) {
          __sync_fetch_and_add(&g_hp_events_recorded, 1);
        } else {
          __sync_fetch_and_add(&g_hp_event_record_failures, 1);
          deferred_record_failed = 1;
        }
      } else {
        snapshot->result = CUDA_SUCCESS;
      }
      if (snapshot->result == CUDA_SUCCESS) {
        snapshot->result = CUDA_ENTRY_CALL(cuda_library_entry,
                                           cuEventSynchronize,
                                           snapshot->event);
        ++g_hp_event_waits;
      }

      if (snapshot->result == CUDA_SUCCESS) {
        continue;
      }

      if (deferred_record_failed) {
        fprintf(stderr,
                "[EVENT][HP] deferred cuEventRecord failed: %d; "
                "falling back to stream synchronization\n",
                (int)snapshot->result);
      } else {
        ++g_hp_event_wait_failures;
        fprintf(stderr,
                "[EVENT][HP] cuEventSynchronize failed: %d; "
                "falling back to stream synchronization\n",
                (int)snapshot->result);
      }
      if (snapshot->per_thread_default_stream) {
        snapshot->result = CUDA_ENTRY_CALL(cuda_library_entry,
                                            cuStreamSynchronize_ptsz,
                                            snapshot->stream);
      } else {
        snapshot->result = CUDA_ENTRY_CALL(cuda_library_entry,
                                            cuStreamSynchronize,
                                            snapshot->stream);
      }
    }

    pthread_mutex_lock(&g_hp_event_lock);
    for (int i = 0; i < snapshot_count; ++i) {
      hp_event_wait_snapshot_t *snapshot = &snapshots[i];
      hp_event_slot_t *slot = &g_hp_event_slots[snapshot->slot_index];
      if (snapshot->result != CUDA_SUCCESS) {
        continue;
      }
      if (slot->active && slot->event == snapshot->event &&
          slot->generation == snapshot->generation) {
        slot->active = 0;
        slot->deferred_record_pending = 0;
        ++g_hp_event_wait_completions;
      } else {
        ++g_hp_event_wait_stale;
      }
    }
    hp_publish_idle_locked();
    pthread_mutex_unlock(&g_hp_event_lock);
  }
}

static void hp_event_query_watcher_loop(void) {
  const struct timespec poll_interval = {
      .tv_sec = g_hp_event_query_interval_ns / (1000 * 1000 * 1000LL),
      .tv_nsec = g_hp_event_query_interval_ns % (1000 * 1000 * 1000LL)};

  while (1) {
    pthread_mutex_lock(&g_hp_event_lock);
    while (!g_hp_event_stop && !hp_any_active_event_locked()) {
      pthread_cond_wait(&g_hp_event_cond, &g_hp_event_lock);
    }
    if (g_hp_event_stop) {
      pthread_mutex_unlock(&g_hp_event_lock);
      break;
    }

    for (int i = 0; i < EVENT_MAX_STREAMS; ++i) {
      hp_event_slot_t *slot = &g_hp_event_slots[i];
      if (!slot->active) {
        continue;
      }
      CUresult result = CUDA_ENTRY_CALL(cuda_library_entry, cuEventQuery,
                                        slot->event);
      ++g_hp_event_queries;
      if (result == CUDA_SUCCESS) {
        slot->active = 0;
      } else if (result != CUDA_ERROR_NOT_READY) {
        fprintf(stderr, "[EVENT][HP] cuEventQuery failed: %d\n", (int)result);
        result = CUDA_ENTRY_CALL(cuda_library_entry, cuEventSynchronize,
                                 slot->event);
        if (result == CUDA_SUCCESS) {
          slot->active = 0;
        }
      }
    }
    int still_active = hp_any_active_event_locked();
    if (!still_active) {
      hp_publish_idle_locked();
    }
    pthread_mutex_unlock(&g_hp_event_lock);

    if (still_active) {
      nanosleep(&poll_interval, NULL);
    }
  }
}

static void *hp_event_watcher(void *unused) {
  (void)unused;
  int context_pushed = 0;
  if (g_hp_event_context != NULL) {
    CUresult result = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxPushCurrent_v2,
                                      g_hp_event_context);
    context_pushed = result == CUDA_SUCCESS;
    if (!context_pushed) {
      fprintf(stderr, "[EVENT][HP] failed to push CUDA context: %d\n",
              (int)result);
    }
  }

  if (g_hp_event_completion_mode == EVENT_COMPLETION_QUERY) {
    hp_event_query_watcher_loop();
  } else {
    hp_event_blocking_watcher_loop(
        g_hp_event_completion_mode == EVENT_COMPLETION_DEFERRED);
  }

  if (context_pushed) {
    CUcontext popped = NULL;
    CUDA_ENTRY_CALL(cuda_library_entry, cuCtxPopCurrent_v2, &popped);
  }
  return NULL;
}

static void event_hp_start_watcher_locked(void) {
  if (g_hp_event_thread_started) {
    return;
  }
  CUresult result = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetCurrent,
                                    &g_hp_event_context);
  if (result != CUDA_SUCCESS || g_hp_event_context == NULL) {
    fprintf(stderr, "[EVENT][HP] cuCtxGetCurrent failed: %d\n", (int)result);
    return;
  }
  int error = pthread_create(&g_hp_event_thread, NULL, hp_event_watcher, NULL);
  if (error != 0) {
    fprintf(stderr, "[EVENT][HP] pthread_create failed: %s\n",
            strerror(error));
    return;
  }
  g_hp_event_thread_started = 1;
}

static void event_hp_before_launch(void) {
  pthread_mutex_lock(&g_hp_event_lock);
  event_hp_start_watcher_locked();
  ++g_hp_launches_in_progress;
  if (!g_hp_state_active) {
    lilt_event_hp_mark_active();
    g_hp_state_active = 1;
    ++g_hp_active_publications;
  }
  pthread_mutex_unlock(&g_hp_event_lock);
}

static void event_hp_after_launch(CUstream stream,
                                  int per_thread_default_stream,
                                  CUresult launch_result) {
  pthread_mutex_lock(&g_hp_event_lock);
  int had_active_event = hp_any_active_event_locked();
  int should_signal_watcher = 0;
  if (launch_result == CUDA_SUCCESS) {
    int slot_index =
        g_hp_event_thread_started ? hp_event_slot_locked(stream) : -1;
    if (slot_index >= 0) {
      hp_event_slot_t *slot = &g_hp_event_slots[slot_index];
      int per_thread_special_stream =
          per_thread_default_stream &&
          (stream == NULL || (uintptr_t)stream == (uintptr_t)2U);
      CUresult result = CUDA_SUCCESS;
      if (g_hp_event_completion_mode != EVENT_COMPLETION_DEFERRED ||
          per_thread_special_stream) {
        if (per_thread_default_stream) {
          result = CUDA_ENTRY_CALL(cuda_library_entry, cuEventRecord_ptsz,
                                   slot->event, stream);
        } else {
          result = CUDA_ENTRY_CALL(cuda_library_entry, cuEventRecord,
                                   slot->event, stream);
        }
      }
      if (result == CUDA_SUCCESS) {
        slot->active = 1;
        slot->per_thread_default_stream = per_thread_default_stream;
        slot->deferred_record_pending =
            g_hp_event_completion_mode == EVENT_COMPLETION_DEFERRED &&
            !per_thread_special_stream;
        ++slot->generation;
        __atomic_add_fetch(&g_hp_event_record_generation, 1,
                           __ATOMIC_RELEASE);
        should_signal_watcher = !had_active_event;
        if (!slot->deferred_record_pending) {
          __sync_fetch_and_add(&g_hp_events_recorded, 1);
        }
      } else {
        __sync_fetch_and_add(&g_hp_event_record_failures, 1);
        fprintf(stderr, "[EVENT][HP] cuEventRecord failed: %d\n",
                (int)result);
        if (per_thread_default_stream) {
          CUDA_ENTRY_CALL(cuda_library_entry, cuStreamSynchronize_ptsz,
                          stream);
        } else {
          CUDA_ENTRY_CALL(cuda_library_entry, cuStreamSynchronize, stream);
        }
      }
    } else {
      __sync_fetch_and_add(&g_hp_event_record_failures, 1);
      if (per_thread_default_stream) {
        CUDA_ENTRY_CALL(cuda_library_entry, cuStreamSynchronize_ptsz, stream);
      } else {
        CUDA_ENTRY_CALL(cuda_library_entry, cuStreamSynchronize, stream);
      }
    }
  }
  if (g_hp_launches_in_progress > 0) {
    --g_hp_launches_in_progress;
  }
  hp_publish_idle_locked();
  if (should_signal_watcher) {
    pthread_cond_signal(&g_hp_event_cond);
  }
  pthread_mutex_unlock(&g_hp_event_lock);
}

__attribute__((constructor)) static void event_hp_process_start(void) {
  if (lilt_event_policy_enabled() && !lilt_event_passthrough_enabled()) {
    g_hp_event_completion_mode = event_completion_mode();
    g_hp_event_query_interval_ns = event_query_interval_ns();
    g_hp_event_blocking_quiet_ns = event_blocking_quiet_ns();
    lilt_event_hp_register();
    fprintf(stderr,
            "[EVENT][HP] completion_mode=%s query_interval_ns=%lld "
            "blocking_quiet_ns=%lld\n",
            event_completion_mode_name(g_hp_event_completion_mode),
            g_hp_event_query_interval_ns, g_hp_event_blocking_quiet_ns);
  }
}

__attribute__((destructor)) static void print_driver_launch_summary() {
  if (lilt_event_policy_enabled() && !lilt_event_passthrough_enabled()) {
    pthread_mutex_lock(&g_hp_event_lock);
    __atomic_store_n(&g_hp_event_stop, 1, __ATOMIC_RELEASE);
    pthread_cond_signal(&g_hp_event_cond);
    pthread_mutex_unlock(&g_hp_event_lock);
    if (g_hp_event_thread_started) {
      pthread_join(g_hp_event_thread, NULL);
    }
    lilt_event_hp_unregister();
    if (g_driver_launch_calls != 0) {
      fprintf(stderr,
              "[EVENT][HP] launches=%llu events=%llu queries=%llu waits=%llu "
              "wait_completions=%llu wait_stale=%llu wait_failures=%llu "
              "settle_checks=%llu record_failures=%llu active=%llu "
              "idle=%llu\n",
              g_driver_launch_calls, g_hp_events_recorded, g_hp_event_queries,
              g_hp_event_waits, g_hp_event_wait_completions,
              g_hp_event_wait_stale, g_hp_event_wait_failures,
              g_hp_event_settle_checks, g_hp_event_record_failures,
              g_hp_active_publications, g_hp_idle_publications);
    }
  }
  if (g_tgs_capture_bypassed_launches != 0) {
    fprintf(stderr,
            "[GRAPH][HP] capture_kernel_bypass=%llu\n",
            g_tgs_capture_bypassed_launches);
  }
  if (g_driver_launch_calls != 0) {
    fprintf(stderr,
            "[%s proxy][HP] Driver kernel launches=%llu blocks=%llu\n",
            lilt_event_passthrough_enabled()
                ? "PASSTHROUGH"
                : (lilt_event_policy_enabled() ? "EVENT" : "TGS"),
            g_driver_launch_calls, g_driver_launch_blocks);
  }
}

/** dynamic rate control */

const char *cuda_error(CUresult code, const char **p) {
  CUDA_ENTRY_CALL(cuda_library_entry, cuGetErrorString, code, p);

  return *p;
}

static ssize_t rio_writen(int fd, void *usrbuf, size_t n) {
  size_t nleft = n;
  ssize_t nwritten;
  char *bufp = usrbuf;

  while (nleft > 0) {
	  if ((nwritten = write(fd, bufp, nleft)) <= 0) {
	    if (errno == EINTR)
		    nwritten = 0;
	    else
		    return -1;
	  }
	  nleft -= nwritten;
	  bufp += nwritten;
  }
  return n;
}


static int open_clientfd(CUdevice device) {
  char SOCKET_PATH[108];
  struct sockaddr_un addr;
  int ret;
  int clientfd;

  /* Create local socket. */

  clientfd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (clientfd == -1) {
    LOGGER(FATAL, "socket failed: %s\n", strerror(errno));
  }

  /*
  * For portability clear the whole structure, since some
  * implementations have additional (nonstandard) fields in
  * the structure.
  */

  memset(&addr, 0, sizeof(addr));

  /* Connect socket to socket address. */

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
  if (access(SOCKET_PATH, 0) != 0)
    return -1;

  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path));

  ret = connect(clientfd, (const struct sockaddr *) &addr, sizeof(addr));
  if (ret == -1) {
    LOGGER(4, "connect failed: %s\n", strerror(errno));
    if (close(clientfd) < 0)
      LOGGER(FATAL, "open_clientfd: close failed: %s\n", strerror(errno));
    return -1;
  }
  return clientfd;
}


static inline void rate_estimator(const long long kernel_size) {
  CUdevice device = 0;
  const CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &device);
  if (ret != CUDA_SUCCESS) {
    fprintf(stderr, "cuCtxGetDevice error\n");
  }

  if (!g_active_gpu[device])
    initialization(device);

  __sync_add_and_fetch_8(&g_rate_counter[device], kernel_size);
}


static void *rate_monitor(void *v_device) {
  const CUdevice device = (uintptr_t)v_device;
  const unsigned long duration = 5000;
  const struct timespec unit_time = {
    .tv_sec = duration / 1000,
    .tv_nsec = duration % 1000 * MILLISEC,
  };
  struct timespec req = unit_time, rem;

  LOGGER(4, "[%d] rate_monitor start\n", device);

  g_rate_counter[device] = 0;
  while (g_active_gpu[device] > 0) {
    int ret = nanosleep(&req, &rem);
    if (ret < 0) {
      if (errno == EINTR) {
        req = rem;
        continue;
      }
      else LOGGER(FATAL, "nanosleep error: %s\n", strerror(errno));
    }
    else
      req = unit_time;

    g_current_rate[device] = g_rate_counter[device];

    g_rate_counter[device] = 0;
  }
  return NULL;
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


static void *rate_watcher(void *v_device) {
  const CUdevice device = (uintptr_t)v_device;
  const unsigned long duration = 5000;

  const struct timespec unit_time = {
    .tv_sec = duration / 1000,
    .tv_nsec = duration % 1000 * MILLISEC,
  };
  const struct timespec listen_time = {
    .tv_nsec = 100 * MILLISEC,
  };
  struct timespec req = listen_time, rem;
  const int WINDOW_SIZE = 5;
  double rate_window[WINDOW_SIZE];
  memset(rate_window, 0, sizeof(rate_window));
  double max_rate = 0;

  while (g_active_gpu[device] > 0) {

    int clientfd;
    int loop_cnt = 0;
    while ((clientfd = open_clientfd(device)) < 0) {
      if (g_active_gpu[device] <= 0) {
        return NULL;
      }

      nanosleep(&listen_time, &rem);

      if (loop_cnt < 20) {
        ++loop_cnt;
        continue;
      }
      else
        loop_cnt = 0;

      double rate_counter =  g_current_rate[device];
      double max_window_rate = shift_window(rate_window, WINDOW_SIZE, (double)rate_counter);

      double max_delta = relative_delta(max_window_rate, max_rate);

      if (max_delta >= -0.2 && max_delta <= 0.2)
        max_rate = max_rate > max_window_rate ? max_rate : max_window_rate;
      else
        max_rate = max_window_rate;
    }

    if (rio_writen(clientfd, (void *)&max_rate, sizeof(double)) != sizeof(double)) {
      LOGGER(4, "rio_writen error\n");
      continue;
    }

    int ret = 0;
    req = unit_time;
    while (g_active_gpu[device] > 0) {

      ret = nanosleep(&req, &rem);
      if (ret < 0) {
        if (errno == EINTR) {
          req = rem;
          continue;
        }
        else LOGGER(FATAL, "nanosleep error: %s\n", strerror(errno));
      }
      else
        req = unit_time;
      double rate_counter = g_current_rate[device];
      double max_window_rate = shift_window(rate_window, WINDOW_SIZE, (double)rate_counter);
      double max_delta = relative_delta(max_window_rate, max_rate);

      if (max_delta >= -0.2 && max_delta <= 0.2)
        max_rate = max_rate > max_window_rate ? max_rate : max_window_rate;
      else
        max_rate = max_window_rate;

      if (rio_writen(clientfd, (void *)&rate_counter, sizeof(double)) != sizeof(double)) {
        LOGGER(4, "rio_writen error\n");
        break;
      }
    }

    close(clientfd);

    max_rate *= 0.8;
  }
  return NULL;
}


static int lilt_set_cpu_affinity(pthread_t thread_id, int core_id) {
    int ret;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    ret = pthread_setaffinity_np(thread_id, sizeof(cpuset), &cpuset);
    if (ret != 0) {
        fprintf(stderr, "failed to set cpu affinity");
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


static void activate_rate_monitor(CUdevice device) {
  pthread_t tid;

  pthread_create(&tid, NULL, rate_monitor, (void *)(uintptr_t)device);
  lilt_set_cpu_affinity(tid, g_gpu_id[device]);

#ifdef __APPLE__
  pthread_setname_np("rate_monitor");
#else
  pthread_setname_np(tid, "rate_monitor");
#endif
}

static inline void initialization(CUdevice device) {
  g_active_gpu[device] = 1;

  fprintf(stderr, "initialize device %d\n", device);

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

  activate_rate_watcher(device);
  activate_rate_monitor(device);
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

  fprintf(stderr, "[cuInit] thread %lu\n", (unsigned long)syscall(SYS_gettid));

  load_necessary_data();

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuInit, flag);
  return ret;
}

CUresult cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize,
                           unsigned int flags) {
  CUresult ret;

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAllocManaged, dptr, bytesize,
                        flags);
  if (ret != CUDA_SUCCESS)
    return ret;

  CUdevice device;
  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &device);
  if (ret != CUDA_SUCCESS)
    return ret;

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAdvise, *dptr, bytesize, CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
                         device);
  return ret;
}

CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize) {
  CUresult ret;
  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAllocManaged, dptr, bytesize,
                        1);
  if (ret != CUDA_SUCCESS)
    return ret;
  CUdevice device;
  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &device);
  if (ret != CUDA_SUCCESS)
    return ret;
  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAdvise, *dptr, bytesize, CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
                         device);
  return ret;
}

CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
  CUresult ret;

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAllocManaged, dptr, bytesize,
                        1);

  if (ret != CUDA_SUCCESS)
    return ret;
  CUdevice device;
  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &device);
  if (ret != CUDA_SUCCESS)
    return ret;
  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAdvise, *dptr, bytesize, CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
                         device);
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

  if (ret != CUDA_SUCCESS)
    return ret;
  CUdevice device;
  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &device);
  if (ret != CUDA_SUCCESS)
    return ret;
  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAdvise, *dptr, bytesize, CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
                         device);
  return ret;
}

CUresult cuMemAllocPitch(CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes,
                         size_t Height, unsigned int ElementSizeBytes) {
  *pPitch = ROUND_UP(WidthInBytes, 128);
  size_t bytesize = ROUND_UP(*pPitch * Height, ElementSizeBytes);
  CUresult ret;

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAllocManaged, dptr, bytesize,
                        1);

  if (ret != CUDA_SUCCESS)
    return ret;
  CUdevice device;
  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &device);
  if (ret != CUDA_SUCCESS)
    return ret;
  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemAdvise, *dptr, bytesize, CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
                         device);
  return ret;
}


CUresult cuMemFree_v2(CUdeviceptr dptr) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemFree_v2, dptr);
}


CUresult cuMemFree(CUdeviceptr dptr) {
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemFree, dptr);
}

CUresult cuArrayCreate_v2(CUarray *pHandle,
                          const CUDA_ARRAY_DESCRIPTOR *pAllocateArray) {
  CUresult ret;

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuArrayCreate_v2, pHandle,
                        pAllocateArray);
  return ret;
}

CUresult cuArrayCreate(CUarray *pHandle,
                       const CUDA_ARRAY_DESCRIPTOR *pAllocateArray) {
  CUresult ret;


  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuArrayCreate, pHandle,
                        pAllocateArray);
  return ret;
}

CUresult cuArray3DCreate_v2(CUarray *pHandle,
                            const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray) {
  CUresult ret;


  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuArray3DCreate_v2, pHandle,
                        pAllocateArray);
  return ret;
}

CUresult cuArray3DCreate(CUarray *pHandle,
                         const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray) {
  CUresult ret;

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuArray3DCreate, pHandle,
                        pAllocateArray);
  return ret;
}

CUresult cuMipmappedArrayCreate(
    CUmipmappedArray *pHandle,
    const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc,
    unsigned int numMipmapLevels) {
  CUresult ret;

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMipmappedArrayCreate, pHandle,
                        pMipmappedArrayDesc, numMipmapLevels);
  return ret;
}

CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev) {
  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuDeviceTotalMem_v2, bytes, dev);
  if (ret != CUDA_SUCCESS)
    return ret;
  *bytes -= g_spare_memory;
  return ret;
}

CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuDeviceTotalMem, bytes, dev);
  if (ret != CUDA_SUCCESS)
    return ret;
  *bytes -= g_spare_memory;
  return ret;
}

CUresult cuMemGetInfo_v2(size_t *free, size_t *total) {
  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemGetInfo_v2, free, total);
  if (ret != CUDA_SUCCESS)
    return ret;
  *total = (*total > g_spare_memory) ? (*total - g_spare_memory) : 0;
  *free = (*free > g_spare_memory) ? (*free - g_spare_memory) : 0;
  return ret;
}

CUresult cuMemGetInfo(size_t *free, size_t *total) {
  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuMemGetInfo, free, total);
  if (ret != CUDA_SUCCESS)
    return ret;
  *total = (*total > g_spare_memory) ? (*total - g_spare_memory) : 0;
  *free = (*free > g_spare_memory) ? (*free - g_spare_memory) : 0;
  return ret;
}

CUresult cuStreamCreate(CUstream *phStream, unsigned int Flags) {
  int leastPriority, greatestPriority;
  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetStreamPriorityRange, &leastPriority, &greatestPriority);
  if (ret == CUDA_SUCCESS) {
    int priority = (leastPriority + greatestPriority - 1) / 2;
    ret = CUDA_ENTRY_CALL(cuda_library_entry, cuStreamCreateWithPriority, phStream, Flags, priority);
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
    leastPriority = (leastPriority + greatestPriority - 1) / 2;
    if (priority > leastPriority) {
      priority = leastPriority;
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
  lilt_before_kernel_launch((uint64_t)gridDimX * gridDimY * gridDimZ);
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
  lilt_before_kernel_launch((uint64_t)gridDimX * gridDimY * gridDimZ);
  CUresult result = CUDA_ENTRY_CALL(
      cuda_library_entry, cuLaunchKernel, f, gridDimX, gridDimY, gridDimZ,
      blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams,
      extra);
  lilt_after_kernel_launch(hStream, 0, result);
  return result;
}

CUresult cuLaunch(CUfunction f) {
  lilt_before_kernel_launch(1);
  CUresult result = CUDA_ENTRY_CALL(cuda_library_entry, cuLaunch, f);
  lilt_after_kernel_launch(NULL, 0, result);
  return result;
}

CUresult cuLaunchCooperativeKernel_ptsz(
    CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
    void **kernelParams) {
  lilt_before_kernel_launch((uint64_t)gridDimX * gridDimY * gridDimZ);
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
  lilt_before_kernel_launch((uint64_t)gridDimX * gridDimY * gridDimZ);
  CUresult result = CUDA_ENTRY_CALL(
      cuda_library_entry, cuLaunchCooperativeKernel, f, gridDimX, gridDimY,
      gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream,
      kernelParams);
  lilt_after_kernel_launch(hStream, 0, result);
  return result;
}

CUresult cuLaunchGrid(CUfunction f, int grid_width, int grid_height) {
  lilt_before_kernel_launch((uint64_t)grid_width * grid_height);
  CUresult result = CUDA_ENTRY_CALL(cuda_library_entry, cuLaunchGrid, f,
                                    grid_width, grid_height);
  lilt_after_kernel_launch(NULL, 0, result);
  return result;
}

CUresult cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height,
                           CUstream hStream) {
  lilt_before_kernel_launch((uint64_t)grid_width * grid_height);
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
  CUdevice device;
  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &device);
  if (ret != CUDA_SUCCESS) {
  }
  else fprintf(stderr, "[%d] cuCtxDestroy_v2!\n", device);

  return CUDA_ENTRY_CALL(cuda_library_entry, cuCtxDestroy_v2, ctx);
}

CUresult cuCtxDestroy(CUcontext ctx) {
  CUdevice device;
  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &device);
  if (ret != CUDA_SUCCESS) {
    LOGGER(FATAL, "cuCtxDestroy error\n");
  }
  else fprintf(stderr, "[%d] cuCtxDestroy!\n", device);

  return CUDA_ENTRY_CALL(cuda_library_entry, cuCtxDestroy, ctx);
}

CUresult cuCtxPopCurrent_v2(CUcontext *pctx) {
  CUdevice device;
  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &device);
  if (ret != CUDA_SUCCESS) {
    LOGGER(FATAL, "cuCtxPopCurrent_v2 error\n");
  }
  else fprintf(stderr, "[%d] cuCtxPopCurrent_v2!\n", device);

  return CUDA_ENTRY_CALL(cuda_library_entry, cuCtxPopCurrent_v2, pctx);
}

CUresult cuCtxPopCurrent(CUcontext *pctx) {
  CUdevice device;
  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &device);
  if (ret != CUDA_SUCCESS) {
    LOGGER(FATAL, "cuCtxPopCurrent error\n");
  }
  else fprintf(stderr, "[%d] cuCtxPopCurrent!\n", device);

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
  CUdevice device;
  CUresult ret = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetDevice, &device);
  if (ret != CUDA_SUCCESS) {
    LOGGER(FATAL, "cuCtxGetDevice error\n");
  }
  else fprintf(stderr, "[%d] cuDevicePrimaryCtxRelease!\n", device);

  return CUDA_ENTRY_CALL(cuda_library_entry, cuDevicePrimaryCtxRelease, dev);
}
