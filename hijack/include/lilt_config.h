/* Lilt configuration names, with read-only compatibility for old experiments. */
#ifndef LILT_CONFIG_H
#define LILT_CONFIG_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Never change the caller's environment. Canonical LILT_* values win. */
static inline char *lilt_getenv(const char *name) {
  char canonical[256], legacy[256];
  const char *suffix;
  if (strncmp(name, "LILT_", 5) == 0) {
    suffix = name + 5;
  } else if (strncmp(name, "TGS_EVENT_", 10) == 0 ||
             strcmp(name, "TGS_POLICY") == 0 ||
             strcmp(name, "TGS_REAL_LIBCUDA") == 0) {
    suffix = name + 4;
  } else {
    /* Actual TGS baseline settings are deliberately not renamed. */
    return getenv(name);
  }
  if (strncmp(suffix, "EVENT_LP_", 9) == 0)
    snprintf(canonical, sizeof(canonical), "LILT_EVENT_BE_%s", suffix + 9);
  else
    snprintf(canonical, sizeof(canonical), "LILT_%s", suffix);
  char *value = getenv(canonical);
  if (value != NULL) return value;
  suffix = canonical + 5;
  if (strncmp(suffix, "EVENT_BE_", 9) == 0)
    snprintf(legacy, sizeof(legacy), "TGS_EVENT_LP_%s", suffix + 9);
  else
    snprintf(legacy, sizeof(legacy), "TGS_%s", suffix);
  return getenv(legacy);
}
#endif
