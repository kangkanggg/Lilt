#include <assert.h>
#include <string.h>
#include "lilt_config.h"

int main(void) {
  unsetenv("LILT_EVENT_BE_WINDOW");
  unsetenv("TGS_EVENT_LP_WINDOW");
  assert(lilt_getenv("LILT_EVENT_BE_WINDOW") == NULL);
  setenv("TGS_EVENT_LP_WINDOW", "3", 1);
  assert(strcmp(lilt_getenv("LILT_EVENT_BE_WINDOW"), "3") == 0);
  setenv("LILT_EVENT_BE_WINDOW", "5", 1);
  assert(strcmp(lilt_getenv("LILT_EVENT_BE_WINDOW"), "5") == 0);
  assert(strcmp(lilt_getenv("TGS_EVENT_LP_WINDOW"), "5") == 0);
  assert(strcmp(getenv("TGS_EVENT_LP_WINDOW"), "3") == 0);
  setenv("TGS_POLICY", "event", 1);
  unsetenv("LILT_POLICY");
  assert(strcmp(lilt_getenv("LILT_POLICY"), "event") == 0);
  setenv("LILT_POLICY", "lilt", 1);
  assert(strcmp(lilt_getenv("LILT_POLICY"), "lilt") == 0);
  setenv("TGS_RATE_THRESHOLD_PCT", "5", 1);
  setenv("LILT_RATE_THRESHOLD_PCT", "99", 1);
  assert(strcmp(lilt_getenv("TGS_RATE_THRESHOLD_PCT"), "5") == 0);
  puts("Lilt configuration compatibility: PASS");
  return 0;
}
