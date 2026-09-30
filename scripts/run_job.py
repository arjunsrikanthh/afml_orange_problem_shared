#!/usr/bin/env python3
"""Run one bounded foreground job and record an auditable result."""

from __future__ import annotations

import argparse
import errno
import fcntl
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
JOBS_ROOT = REPO_ROOT / "results" / "jobs"
MAX_TIMEOUT_SECONDS = 7 * 24 * 60 * 60
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def validate_id(value: str) -> str:
    if not ID_RE.fullmatch(value) or value in {".", ".."}:
        raise ValueError("job id must be 1-64 characters of ASCII letters, digits, '.', '_' or '-'")
    return value


def job_dir(job_id: str) -> Path:
    validate_id(job_id)
    return JOBS_ROOT / job_id


def atomic_write_json(path: Path, value: dict) -> None:
    data = json.dumps(value, indent=2, sort_keys=True) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=".state.", suffix=".tmp", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def read_state(job_id: str) -> dict:
    path = job_dir(job_id) / "state.json"
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("state.json is not an object")
    return value


def pid_alive(pid: object) -> bool | None:
    if not isinstance(pid, int) or pid <= 0:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None
    return True


def process_start_marker(pid: object) -> str | None:
    """Return a kernel-reported process start marker when it can be inspected."""
    if not isinstance(pid, int) or pid <= 0:
        return None
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "lstart="],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    marker = result.stdout.strip()
    return marker or None


def acquire_runner_lock(jobs_root: Path):
    """Acquire the process-wide runner lock without waiting for another job."""
    handle = (jobs_root / ".runner.lock").open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        if exc.errno in {errno.EACCES, errno.EAGAIN}:
            return None
        raise
    return handle


def terminate_process(process: subprocess.Popen[bytes], grace_seconds: float = 2.0) -> None:
    """Stop the job, including its POSIX process group where supported."""
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    else:
        # Windows cannot reliably kill arbitrary descendants without a job object.
        process.terminate()
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
    if os.name == "posix":
        # The leader may have exited while descendants remain in this new
        # session. Escalate the group regardless of the leader's poll state.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.kill()
    process.wait()


def run_locked(args: argparse.Namespace) -> int:
    try:
        validate_id(args.id)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not args.command:
        print("error: command is required after --", file=sys.stderr)
        return 2

    jobs_root = JOBS_ROOT
    jobs_root.mkdir(parents=True, exist_ok=True)
    directory = job_dir(args.id)
    try:
        directory.mkdir()
    except FileExistsError:
        print(f"error: job id already exists: {args.id}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: cannot create job directory: {exc}", file=sys.stderr)
        return 2

    log_path = directory / "job.log"
    state_path = directory / "state.json"
    started = utc_now()
    state = {
        "argv": args.command,
        "cwd": str(REPO_ROOT),
        "start_utc": started,
        "end_utc": None,
        "pid": None,
        "pid_start": None,
        "exitcode": None,
        "status": "starting",
    }
    atomic_write_json(state_path, state)

    process: subprocess.Popen[bytes] | None = None
    try:
        with log_path.open("wb") as log:
            kwargs: dict = {"cwd": str(REPO_ROOT), "stdout": log, "stderr": subprocess.STDOUT}
            if os.name == "posix":
                kwargs["start_new_session"] = True
            elif os.name == "nt":
                kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            process = subprocess.Popen(args.command, **kwargs)
            state["pid"] = process.pid
            state["pid_start"] = process_start_marker(process.pid)
            state["status"] = "running"
            atomic_write_json(state_path, state)
            try:
                exitcode = process.wait(timeout=args.timeout_seconds)
                status = "succeeded" if exitcode == 0 else "failed"
            except subprocess.TimeoutExpired:
                terminate_process(process)
                exitcode = process.returncode
                status = "timeout"
            except KeyboardInterrupt:
                terminate_process(process)
                exitcode = process.returncode
                status = "interrupted"
    except KeyboardInterrupt:
        if process is not None:
            terminate_process(process)
        exitcode = process.returncode if process is not None else None
        status = "interrupted"
    except (OSError, ValueError) as exc:
        if process is not None:
            terminate_process(process)
        exitcode = None
        status = "failed"
        with log_path.open("ab") as log:
            log.write(f"runner error: {exc}\n".encode())

    state.update({"end_utc": utc_now(), "exitcode": exitcode, "status": status})
    atomic_write_json(state_path, state)
    print(json.dumps(state, sort_keys=True))
    return 0 if status == "succeeded" else 1


def run(args: argparse.Namespace) -> int:
    jobs_root = JOBS_ROOT
    jobs_root.mkdir(parents=True, exist_ok=True)
    try:
        lock = acquire_runner_lock(jobs_root)
    except OSError as exc:
        print(f"error: cannot acquire runner lock: {exc}", file=sys.stderr)
        return 2
    if lock is None:
        print("error: another job is already running", file=sys.stderr)
        return 2
    try:
        return run_locked(args)
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def status(args: argparse.Namespace) -> int:
    try:
        state = read_state(args.id)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: cannot read state for {args.id}: {exc}", file=sys.stderr)
        return 2
    result = dict(state)
    if state.get("status") in {"succeeded", "failed", "timeout", "interrupted"}:
        result["live"] = False
        result["liveness"] = "not-running"
        result["restart"] = "manual-only"
    else:
        pid = state.get("pid")
        alive = pid_alive(pid)
        marker = process_start_marker(pid)
        if alive is False:
            result["live"] = False
            result["liveness"] = "not-running"
        elif (
            alive is True
            and isinstance(state.get("pid_start"), str)
            and state["pid_start"]
            and marker == state["pid_start"]
        ):
            result["live"] = True
            result["liveness"] = "running"
        else:
            # A live PID is insufficient: it may have been reused, or identity
            # inspection may be unavailable. Never claim that job is running.
            result["live"] = None
            result["liveness"] = "unknown/stale"
        result["restart"] = "manual-only"
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="action", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--id", required=True)
    run_parser.add_argument("--timeout-seconds", required=True, type=float)
    run_parser.add_argument("command", nargs=argparse.REMAINDER)
    status_parser = commands.add_parser("status")
    status_parser.add_argument("--id", required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.action == "run":
        if not math.isfinite(args.timeout_seconds) or not 0 < args.timeout_seconds <= MAX_TIMEOUT_SECONDS:
            print(f"error: timeout must be finite and between 0 and {MAX_TIMEOUT_SECONDS} seconds", file=sys.stderr)
            return 2
        if args.command[:1] == ["--"]:
            args.command = args.command[1:]
        return run(args)
    return status(args)


if __name__ == "__main__":
    raise SystemExit(main())
