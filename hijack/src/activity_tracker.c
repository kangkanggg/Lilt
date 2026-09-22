/* Modified for Lilt (2026): local proxy and HP/BE scheduling. See LICENSE. */
#include "../include/lilt_config.h"
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <linux/futex.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include "../include/activity_tracker.h"

#define LILT_EVENT_MAGIC 0x54475345U
#define LILT_EVENT_VERSION 4U

static const uint64_t g_idle_survival_thresholds_ns[
    LILT_EVENT_IDLE_SURVIVAL_BUCKETS] = {
    20ULL * 1000,    50ULL * 1000,    75ULL * 1000,
    100ULL * 1000,   125ULL * 1000,   141ULL * 1000,
    161ULL * 1000,   180ULL * 1000,   200ULL * 1000,
    250ULL * 1000,   300ULL * 1000,   400ULL * 1000,
    500ULL * 1000,   750ULL * 1000,   1000ULL * 1000,
    1500ULL * 1000,  2500ULL * 1000,  5000ULL * 1000,
    10000ULL * 1000, 20000ULL * 1000,
};

typedef struct {
  uint32_t magic;
  uint32_t version;
  uint32_t hp_state;
  uint32_t hp_owner_pid;
  uint32_t hp_registration_count;
  uint32_t hp_idle_samples;
  uint32_t hp_active_samples;
  uint64_t hp_generation;
  uint64_t hp_last_active_ns;
  uint64_t hp_last_idle_ns;
  uint64_t hp_idle_ewma_ns;
  uint64_t hp_active_ewma_ns;
  uint64_t hp_idle_survival_counts[LILT_EVENT_IDLE_SURVIVAL_BUCKETS];
  uint64_t reserved[2];
} lilt_event_shared_state_t;

uint64_t lilt_event_idle_survival_threshold_ns(uint32_t index) {
  if (index >= LILT_EVENT_IDLE_SURVIVAL_BUCKETS) {
    return 0;
  }
  return g_idle_survival_thresholds_ns[index];
}

static pthread_mutex_t g_map_lock = PTHREAD_MUTEX_INITIALIZER;
static lilt_event_shared_state_t *g_shared;
static char g_shm_name[128];

static uint64_t monotonic_ns(void) {
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  return (uint64_t)now.tv_sec * 1000000000ULL + (uint64_t)now.tv_nsec;
}

static void update_ewma_ns(uint64_t *address, uint64_t sample_ns) {
  uint64_t previous = __atomic_load_n(address, __ATOMIC_ACQUIRE);
  uint64_t updated;
  if (previous == 0) {
    updated = sample_ns;
  } else if (sample_ns >= previous) {
    updated = previous + (sample_ns - previous) / 8U;
  } else {
    updated = previous - (previous - sample_ns) / 8U;
  }
  __atomic_store_n(address, updated, __ATOMIC_RELEASE);
}

int lilt_event_policy_enabled(void) {
  static int initialized;
  static int enabled;
  if (!__atomic_load_n(&initialized, __ATOMIC_ACQUIRE)) {
    const char *policy = lilt_getenv("LILT_POLICY");
    enabled = policy != NULL &&
              (strcmp(policy, "event") == 0 || strcmp(policy, "lilt") == 0);
    __atomic_store_n(&initialized, 1, __ATOMIC_RELEASE);
  }
  return enabled;
}

int lilt_event_passthrough_enabled(void) {
  static int initialized;
  static int enabled;
  if (!__atomic_load_n(&initialized, __ATOMIC_ACQUIRE)) {
    const char *configured = lilt_getenv("LILT_EVENT_PASSTHROUGH");
    enabled = configured != NULL &&
              (strcmp(configured, "1") == 0 ||
               strcmp(configured, "true") == 0 ||
               strcmp(configured, "yes") == 0);
    __atomic_store_n(&initialized, 1, __ATOMIC_RELEASE);
  }
  return enabled;
}

static const char *event_shm_name(void) {
  const char *configured = lilt_getenv("LILT_EVENT_SHM_NAME");
  if (configured != NULL && configured[0] == '/') {
    return configured;
  }
  snprintf(g_shm_name, sizeof(g_shm_name), "/tgs-event-%u",
           (unsigned int)getuid());
  return g_shm_name;
}

static lilt_event_shared_state_t *event_shared_get(void) {
  if (g_shared != NULL) {
    return g_shared;
  }

  pthread_mutex_lock(&g_map_lock);
  if (g_shared == NULL) {
    const char *name = event_shm_name();
    int fd = shm_open(name, O_RDWR | O_CREAT, 0600);
    if (fd < 0) {
      fprintf(stderr, "[EVENT] shm_open(%s) failed: %s\n", name,
              strerror(errno));
      pthread_mutex_unlock(&g_map_lock);
      return NULL;
    }
    if (ftruncate(fd, (off_t)sizeof(lilt_event_shared_state_t)) != 0) {
      fprintf(stderr, "[EVENT] ftruncate failed: %s\n", strerror(errno));
      close(fd);
      pthread_mutex_unlock(&g_map_lock);
      return NULL;
    }
    void *mapping = mmap(NULL, sizeof(lilt_event_shared_state_t),
                         PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapping == MAP_FAILED) {
      fprintf(stderr, "[EVENT] mmap failed: %s\n", strerror(errno));
      close(fd);
      pthread_mutex_unlock(&g_map_lock);
      return NULL;
    }

    lilt_event_shared_state_t *state = mapping;
    if (flock(fd, LOCK_EX) != 0) {
      fprintf(stderr, "[EVENT] flock failed: %s\n", strerror(errno));
      munmap(mapping, sizeof(*state));
      close(fd);
      pthread_mutex_unlock(&g_map_lock);
      return NULL;
    }
    if (state->magic != LILT_EVENT_MAGIC ||
        state->version != LILT_EVENT_VERSION) {
      memset(state, 0, sizeof(*state));
      state->version = LILT_EVENT_VERSION;
      __atomic_thread_fence(__ATOMIC_RELEASE);
      state->magic = LILT_EVENT_MAGIC;
    }
    flock(fd, LOCK_UN);
    close(fd);
    g_shared = state;
  }
  pthread_mutex_unlock(&g_map_lock);
  return g_shared;
}

static int futex_wait_active(uint32_t *address) {
  const struct timespec timeout = {.tv_sec = 0, .tv_nsec = 10 * 1000 * 1000};
  return (int)syscall(SYS_futex, address, FUTEX_WAIT,
                      LILT_EVENT_HP_ACTIVE, &timeout, NULL, 0);
}

static void futex_wake_all(uint32_t *address) {
  syscall(SYS_futex, address, FUTEX_WAKE, INT_MAX, NULL, NULL, 0);
}

static int owner_is_alive(pid_t owner) {
  if (owner <= 0) {
    return 0;
  }
  if (kill(owner, 0) == 0) {
    return 1;
  }
  return errno == EPERM;
}

int lilt_event_hp_register(void) {
  lilt_event_shared_state_t *state = event_shared_get();
  if (state == NULL) {
    return -1;
  }
  uint32_t self = (uint32_t)getpid();
  uint32_t owner =
      __atomic_load_n(&state->hp_owner_pid, __ATOMIC_ACQUIRE);
  if (owner == self) {
    __atomic_add_fetch(&state->hp_registration_count, 1,
                       __ATOMIC_ACQ_REL);
    return 0;
  }
  __atomic_store_n(&state->hp_registration_count, 1U, __ATOMIC_RELEASE);
  __atomic_store_n(&state->hp_owner_pid, (uint32_t)getpid(),
                   __ATOMIC_RELEASE);
  __atomic_add_fetch(&state->hp_generation, 1, __ATOMIC_ACQ_REL);
  __atomic_store_n(&state->hp_idle_samples, 0U, __ATOMIC_RELEASE);
  __atomic_store_n(&state->hp_active_samples, 0U, __ATOMIC_RELEASE);
  __atomic_store_n(&state->hp_last_active_ns, 0U, __ATOMIC_RELEASE);
  __atomic_store_n(&state->hp_last_idle_ns, monotonic_ns(),
                   __ATOMIC_RELEASE);
  __atomic_store_n(&state->hp_idle_ewma_ns, 0U, __ATOMIC_RELEASE);
  __atomic_store_n(&state->hp_active_ewma_ns, 0U, __ATOMIC_RELEASE);
  for (uint32_t i = 0; i < LILT_EVENT_IDLE_SURVIVAL_BUCKETS; ++i) {
    __atomic_store_n(&state->hp_idle_survival_counts[i], 0U,
                     __ATOMIC_RELEASE);
  }
  __atomic_store_n(&state->hp_state, LILT_EVENT_HP_IDLE, __ATOMIC_RELEASE);
  futex_wake_all(&state->hp_state);
  fprintf(stderr, "[EVENT][HP] registered pid=%d shm=%s\n", (int)getpid(),
          event_shm_name());
  return 0;
}

void lilt_event_hp_unregister(void) {
  lilt_event_shared_state_t *state = event_shared_get();
  if (state == NULL) {
    return;
  }
  uint32_t self = (uint32_t)getpid();
  if (__atomic_load_n(&state->hp_owner_pid, __ATOMIC_ACQUIRE) != self) {
    return;
  }
  uint32_t registrations = __atomic_load_n(&state->hp_registration_count,
                                            __ATOMIC_ACQUIRE);
  while (registrations > 1) {
    if (__atomic_compare_exchange_n(&state->hp_registration_count,
                                    &registrations, registrations - 1, 0,
                                    __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
      return;
    }
  }
  __atomic_store_n(&state->hp_registration_count, 0U, __ATOMIC_RELEASE);
  __atomic_store_n(&state->hp_state, LILT_EVENT_HP_IDLE, __ATOMIC_RELEASE);
  __atomic_store_n(&state->hp_owner_pid, 0U, __ATOMIC_RELEASE);
  futex_wake_all(&state->hp_state);
  shm_unlink(event_shm_name());
}

uint64_t lilt_event_hp_mark_active(void) {
  lilt_event_shared_state_t *state = event_shared_get();
  if (state == NULL) {
    return 0;
  }
  uint64_t now = monotonic_ns();
  uint64_t last_idle =
      __atomic_load_n(&state->hp_last_idle_ns, __ATOMIC_ACQUIRE);
  if (last_idle != 0 && now > last_idle) {
    uint64_t idle_duration_ns = now - last_idle;
    update_ewma_ns(&state->hp_idle_ewma_ns, idle_duration_ns);
    for (uint32_t i = 0; i < LILT_EVENT_IDLE_SURVIVAL_BUCKETS; ++i) {
      if (idle_duration_ns > g_idle_survival_thresholds_ns[i]) {
        __atomic_add_fetch(&state->hp_idle_survival_counts[i], 1U,
                           __ATOMIC_ACQ_REL);
      }
    }
    __atomic_add_fetch(&state->hp_idle_samples, 1U, __ATOMIC_ACQ_REL);
  }
  __atomic_store_n(&state->hp_last_active_ns, now, __ATOMIC_RELEASE);
  uint64_t generation =
      __atomic_add_fetch(&state->hp_generation, 1, __ATOMIC_ACQ_REL);
  __atomic_store_n(&state->hp_state, LILT_EVENT_HP_ACTIVE, __ATOMIC_RELEASE);
  return generation;
}

void lilt_event_hp_mark_idle(void) {
  lilt_event_shared_state_t *state = event_shared_get();
  if (state == NULL) {
    return;
  }
  uint64_t now = monotonic_ns();
  uint64_t last_active =
      __atomic_load_n(&state->hp_last_active_ns, __ATOMIC_ACQUIRE);
  if (last_active != 0 && now > last_active) {
    update_ewma_ns(&state->hp_active_ewma_ns, now - last_active);
    __atomic_add_fetch(&state->hp_active_samples, 1U, __ATOMIC_ACQ_REL);
  }
  __atomic_store_n(&state->hp_last_idle_ns, now, __ATOMIC_RELEASE);
  __atomic_store_n(&state->hp_state, LILT_EVENT_HP_IDLE, __ATOMIC_RELEASE);
  futex_wake_all(&state->hp_state);
}

int lilt_event_hp_timing_snapshot(lilt_event_hp_timing_t *snapshot) {
  if (snapshot == NULL) {
    return -1;
  }
  lilt_event_shared_state_t *state = event_shared_get();
  if (state == NULL) {
    return -1;
  }
  snapshot->hp_state = __atomic_load_n(&state->hp_state, __ATOMIC_ACQUIRE);
  snapshot->idle_samples =
      __atomic_load_n(&state->hp_idle_samples, __ATOMIC_ACQUIRE);
  snapshot->active_samples =
      __atomic_load_n(&state->hp_active_samples, __ATOMIC_ACQUIRE);
  snapshot->generation =
      __atomic_load_n(&state->hp_generation, __ATOMIC_ACQUIRE);
  snapshot->last_active_ns =
      __atomic_load_n(&state->hp_last_active_ns, __ATOMIC_ACQUIRE);
  snapshot->last_idle_ns =
      __atomic_load_n(&state->hp_last_idle_ns, __ATOMIC_ACQUIRE);
  snapshot->idle_ewma_ns =
      __atomic_load_n(&state->hp_idle_ewma_ns, __ATOMIC_ACQUIRE);
  snapshot->active_ewma_ns =
      __atomic_load_n(&state->hp_active_ewma_ns, __ATOMIC_ACQUIRE);
  for (uint32_t i = 0; i < LILT_EVENT_IDLE_SURVIVAL_BUCKETS; ++i) {
    snapshot->idle_survival_counts[i] = __atomic_load_n(
        &state->hp_idle_survival_counts[i], __ATOMIC_ACQUIRE);
  }
  return 0;
}

pid_t lilt_event_hp_owner(void) {
  lilt_event_shared_state_t *state = event_shared_get();
  if (state == NULL) {
    return 0;
  }
  uint32_t owner =
      __atomic_load_n(&state->hp_owner_pid, __ATOMIC_ACQUIRE);
  if (owner != 0U && !owner_is_alive((pid_t)owner)) {
    uint32_t expected = owner;
    if (__atomic_compare_exchange_n(&state->hp_owner_pid, &expected, 0U, 0,
                                    __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
      __atomic_store_n(&state->hp_registration_count, 0U, __ATOMIC_RELEASE);
      __atomic_store_n(&state->hp_state, LILT_EVENT_HP_IDLE, __ATOMIC_RELEASE);
      futex_wake_all(&state->hp_state);
      return 0;
    }
    owner = expected;
  }
  return (pid_t)owner;
}

int lilt_event_be_wait_until_idle(uint64_t *wait_ns) {
  lilt_event_shared_state_t *state = event_shared_get();
  if (state == NULL) {
    return -1;
  }

  uint64_t start = monotonic_ns();
  int waited = 0;
  while (__atomic_load_n(&state->hp_state, __ATOMIC_ACQUIRE) ==
         LILT_EVENT_HP_ACTIVE) {
    waited = 1;
    pid_t owner =
        (pid_t)__atomic_load_n(&state->hp_owner_pid, __ATOMIC_ACQUIRE);
    if (!owner_is_alive(owner)) {
      uint32_t expected = LILT_EVENT_HP_ACTIVE;
      if (__atomic_compare_exchange_n(&state->hp_state, &expected,
                                      LILT_EVENT_HP_IDLE, 0,
                                      __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
        __atomic_store_n(&state->hp_owner_pid, 0U, __ATOMIC_RELEASE);
        futex_wake_all(&state->hp_state);
      }
      break;
    }
    futex_wait_active(&state->hp_state);
  }
  if (wait_ns != NULL && waited) {
    *wait_ns += monotonic_ns() - start;
  }
  return waited;
}
