#ifndef LILT_EVENT_POLICY_H
#define LILT_EVENT_POLICY_H

#include <stdint.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LILT_EVENT_HP_IDLE 0U
#define LILT_EVENT_HP_ACTIVE 1U
#define LILT_EVENT_IDLE_SURVIVAL_BUCKETS 20U

typedef struct {
  uint32_t hp_state;
  uint32_t idle_samples;
  uint32_t active_samples;
  uint64_t generation;
  uint64_t last_active_ns;
  uint64_t last_idle_ns;
  uint64_t idle_ewma_ns;
  uint64_t active_ewma_ns;
  uint64_t idle_survival_counts[LILT_EVENT_IDLE_SURVIVAL_BUCKETS];
} lilt_event_hp_timing_t;

/** Return the fixed lower-bound threshold represented by one histogram bin. */
uint64_t lilt_event_idle_survival_threshold_ns(uint32_t index);

/** True when this proxy process was launched with TGS_POLICY=event. */
int lilt_event_policy_enabled(void);

/** True when interception remains enabled but all scheduler work is bypassed. */
int lilt_event_passthrough_enabled(void);

/** Register/unregister the single HP process for this experiment run. */
int lilt_event_hp_register(void);
void lilt_event_hp_unregister(void);

/** Publish HP activity transitions to BE processes. */
uint64_t lilt_event_hp_mark_active(void);
void lilt_event_hp_mark_idle(void);

/** Inspect the registered HP owner. Zero means no HP process is registered. */
pid_t lilt_event_hp_owner(void);

/**
 * Return a best-effort snapshot of HP ACTIVE/IDLE timing. The scheduling
 * correctness path must still use lilt_event_be_wait_until_idle(); this
 * snapshot is only an advisory input for BE admission sizing.
 */
int lilt_event_hp_timing_snapshot(lilt_event_hp_timing_t *snapshot);

/**
 * Sleep until HP is idle. The HP-side event watcher wakes this call directly.
 * The measured wait duration is added to wait_ns when it is non-NULL.
 */
int lilt_event_be_wait_until_idle(uint64_t *wait_ns);

#ifdef __cplusplus
}
#endif

#endif
