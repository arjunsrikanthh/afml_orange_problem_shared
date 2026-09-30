from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_job.py"
JOBS = ROOT / "results" / "jobs"


def invoke(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=ROOT, text=True, capture_output=True, timeout=15)


def state(job_id: str) -> dict:
    return json.loads((JOBS / job_id / "state.json").read_text())


def py(code: str) -> list[str]:
    return [sys.executable, "-c", code]


class RunJobTests(unittest.TestCase):
    def test_success_records_state_and_log(self):
        job_id = f"test-success-{os.getpid()}"
        result = invoke("run", "--id", job_id, "--timeout-seconds", "5", "--", *py("print('hello')"))
        self.assertEqual(result.returncode, 0, result.stderr)
        saved = state(job_id)
        self.assertEqual(saved["status"], "succeeded")
        self.assertEqual(saved["exitcode"], 0)
        self.assertEqual(saved["argv"], py("print('hello')"))
        self.assertTrue(saved["end_utc"])
        self.assertIn("hello", (JOBS / job_id / "job.log").read_text())

    def test_failed_command_records_nonzero_exit(self):
        job_id = f"test-failed-{os.getpid()}"
        result = invoke("run", "--id", job_id, "--timeout-seconds", "5", "--", *py("raise SystemExit(7)"))
        self.assertEqual(result.returncode, 1)
        saved = state(job_id)
        self.assertEqual(saved["status"], "failed")
        self.assertEqual(saved["exitcode"], 7)

    def test_timeout_kills_child_descendant(self):
        job_id = f"test-timeout-{os.getpid()}"
        marker = JOBS / job_id / "descendant-marker"
        descendant = "import pathlib,time; time.sleep(1); pathlib.Path(%r).write_text('alive')" % str(marker)
        code = "import subprocess,sys,time; subprocess.Popen([sys.executable, '-c', %r]); time.sleep(30)" % descendant
        result = invoke("run", "--id", job_id, "--timeout-seconds", "0.2", "--", *py(code))
        self.assertEqual(result.returncode, 1)
        saved = state(job_id)
        self.assertEqual(saved["status"], "timeout")
        self.assertIsNotNone(saved["exitcode"])
        time.sleep(0.1)
        self.assertFalse(marker.exists())

    def test_timeout_kills_sigterm_ignoring_descendant(self):
        job_id = f"test-timeout-ignore-term-{os.getpid()}"
        marker = JOBS / job_id / "descendant-marker"
        descendant = (
            "import pathlib,signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "time.sleep(1); pathlib.Path(%r).write_text('alive')"
        ) % str(marker)
        code = "import subprocess,sys,time; subprocess.Popen([sys.executable, '-c', %r]); time.sleep(30)" % descendant
        result = invoke("run", "--id", job_id, "--timeout-seconds", "0.2", "--", *py(code))
        self.assertEqual(result.returncode, 1)
        self.assertEqual(state(job_id)["status"], "timeout")
        time.sleep(1.2)
        self.assertFalse(marker.exists())

    def test_duplicate_and_path_like_ids_are_rejected(self):
        job_id = f"test-duplicate-{os.getpid()}"
        first = invoke("run", "--id", job_id, "--timeout-seconds", "5", "--", *py("pass"))
        second = invoke("run", "--id", job_id, "--timeout-seconds", "5", "--", *py("pass"))
        traversal = invoke("run", "--id", "../outside", "--timeout-seconds", "5", "--", *py("pass"))
        self.assertEqual(first.returncode, 0)
        self.assertEqual(second.returncode, 2)
        self.assertEqual(traversal.returncode, 2)
        self.assertFalse((ROOT / "results" / "outside").exists())

    def test_concurrent_jobs_are_serialized_by_process_wide_lock(self):
        first_id = f"test-lock-first-{os.getpid()}"
        second_id = f"test-lock-second-{os.getpid()}"
        first = subprocess.Popen(
            [sys.executable, str(SCRIPT), "run", "--id", first_id, "--timeout-seconds", "5", "--", *py("import time; time.sleep(1.2)")],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            deadline = time.time() + 5
            while not (JOBS / first_id / "state.json").exists() and time.time() < deadline:
                time.sleep(0.02)
            self.assertTrue((JOBS / first_id / "state.json").exists())
            second = invoke("run", "--id", second_id, "--timeout-seconds", "5", "--", *py("pass"))
            self.assertEqual(second.returncode, 2, second.stderr)
            self.assertIn("another job is already running", second.stderr)
            self.assertFalse((JOBS / second_id).exists())
        finally:
            first.communicate(timeout=5)

    def test_status_does_not_trust_reused_or_unverifiable_pid(self):
        job_id = f"test-stale-state-{os.getpid()}"
        directory = JOBS / job_id
        directory.mkdir(parents=True)
        (directory / "state.json").write_text(json.dumps({
            "argv": py("pass"),
            "cwd": str(ROOT),
            "start_utc": "2026-09-29T00:00:00Z",
            "end_utc": None,
            "pid": os.getpid(),
            "pid_start": "definitely-not-this-process",
            "exitcode": None,
            "status": "running",
        }))
        result = invoke("status", "--id", job_id)
        self.assertEqual(result.returncode, 0)
        reported = json.loads(result.stdout)
        self.assertIsNone(reported["live"])
        self.assertEqual(reported["liveness"], "unknown/stale")
        self.assertEqual(reported["restart"], "manual-only")

    def test_status_does_not_restart_terminal_job(self):
        job_id = f"test-status-{os.getpid()}"
        self.assertEqual(invoke("run", "--id", job_id, "--timeout-seconds", "5", "--", *py("pass")).returncode, 0)
        before = state(job_id)
        result = invoke("status", "--id", job_id)
        after = state(job_id)
        reported = json.loads(result.stdout)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(reported["status"], "succeeded")
        self.assertFalse(reported["live"])
        self.assertEqual(reported["restart"], "manual-only")
        self.assertEqual(after, before)
