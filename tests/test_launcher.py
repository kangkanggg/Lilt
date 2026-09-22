"""CPU-only checks; no CUDA library is loaded and no GPU work is submitted."""
import importlib.util
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("lilt_launcher", ROOT / "scripts/run_lilt.py")
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="lilt test ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for role in ("hp", "be"):
            path = self.root / "hijack" / f"{role}-lib/libcuda.so.1"
            path.parent.mkdir(parents=True)
            path.touch()
        self.driver = self.root / "real-driver.so"
        self.driver.touch()
        self.args = SimpleNamespace(role="be", session="test_run", gpu="0", be_count=2,
                                    window=64, time_budget_ns=80000, quiet_ns=20000,
                                    unknown_kernel_ns=20000, graph_mode="whole",
                                    graphlet_nodes=8, keep_library_path=False)
        self.env = {"LILT_REAL_LIBCUDA": str(self.driver), "PATH": os.environ["PATH"]}
        patcher = patch.object(launcher, "ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_defaults_and_old_ablation_cleared(self):
        self.env.update(TGS_POLICY="tgs", TGS_EVENT_LP_POLICY="fixed",
                        LILT_EVENT_PASSTHROUGH="1", LILT_EVENT_COMPLETION_MODE="query")
        env = launcher.prepare_environment(self.args, self.env)
        self.assertEqual(env["LILT_POLICY"], "lilt")
        self.assertEqual(env["LILT_EVENT_BE_COUNT"], "2")
        self.assertEqual(env["LILT_EVENT_BE_POLICY"], "time-sum")
        self.assertEqual(env["LILT_EVENT_COMPLETION_MODE"], "deferred")
        self.assertNotIn("LILT_EVENT_PASSTHROUGH", env)
        self.assertFalse(any(key.startswith("TGS_") for key in env))

    def test_roles_share_state_but_not_proxy(self):
        be = launcher.prepare_environment(self.args, self.env)
        self.args.role = "hp"
        hp = launcher.prepare_environment(self.args, self.env)
        self.assertEqual(hp["LILT_EVENT_SHM_NAME"], be["LILT_EVENT_SHM_NAME"])
        self.assertNotEqual(hp["LD_PRELOAD"], be["LD_PRELOAD"])

    def test_invalid_session(self):
        self.args.session = "../../other"
        with self.assertRaises(ValueError):
            launcher.prepare_environment(self.args, self.env)

    def test_group_window_bound(self):
        self.args.be_count = 3
        self.args.window = 2
        with self.assertRaises(ValueError):
            launcher.prepare_environment(self.args, self.env)

    def test_stacked_proxy_refused(self):
        self.env["LD_PRELOAD"] = "some-library.so"
        with self.assertRaises(ValueError):
            launcher.prepare_environment(self.args, self.env)

    def test_framework_library_search_preserved(self):
        self.args.keep_library_path = True
        self.env["LD_LIBRARY_PATH"] = "/framework/cuda"
        env = launcher.prepare_environment(self.args, self.env)
        self.assertEqual(env["LD_LIBRARY_PATH"], "/framework/cuda")

    def test_legacy_real_driver_override(self):
        self.env["TGS_REAL_LIBCUDA"] = self.env.pop("LILT_REAL_LIBCUDA")
        env = launcher.prepare_environment(self.args, self.env)
        self.assertEqual(env["LILT_REAL_LIBCUDA"], str(self.driver))

    def test_real_driver_cannot_be_proxy(self):
        self.env["LILT_REAL_LIBCUDA"] = str(self.root / "hijack/be-lib/libcuda.so.1")
        with self.assertRaises(ValueError):
            launcher.prepare_environment(self.args, self.env)


if __name__ == "__main__":
    unittest.main()
