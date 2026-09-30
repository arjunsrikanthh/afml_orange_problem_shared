# Bounded job runner

`scripts/run_job.py` runs one command in the foreground and stores its evidence in
`results/jobs/<id>/`. The job ID is deliberately not a filesystem path: it must be
1–64 ASCII letters, digits, `.`, `_`, or `-`, starting with a letter or digit.
The root is fixed to this repository; callers cannot redirect output with an ID.

## Usage

```sh
python scripts/run_job.py run --id smoke-001 --timeout-seconds 3600 -- python train.py --config configs/pilot.yaml
python scripts/run_job.py status --id smoke-001
```

Each job directory contains `job.log` (combined stdout/stderr) and an atomically
replaced `state.json`. State records the exact `argv`, fixed repository `cwd`,
`start_utc`, `end_utc`, child `pid`, the child start marker reported by macOS `ps`
(`pid_start`), `exitcode`, and one of `starting`, `running`,
`succeeded`, `failed`, `timeout`, or `interrupted`. IDs are reserved by exclusive
directory creation, so duplicates fail before another command can run. Timeout
must be finite, positive, and at most seven days.

The runner also takes a nonblocking exclusive lock on `results/jobs/.runner.lock`
before creating a job directory and holds it until the child has exited and its
state is finalized. A second runner invocation therefore exits with status 2 and
does not launch a command, even when it uses a different job ID. The lock is
released by process exit; it does not kill or inspect an unknown process. This is
the runner's single active job guard. It does not enforce the coordinator's
calendar compute budget; the coordinator must enforce the documented one-heavy-job
and two-compute-hour-per-heartbeat policy.

Commands are passed directly to `subprocess.Popen`; no shell is used. On macOS/Linux
the child starts a new session and timeout/interrupt termination signals the whole
process group (TERM, then KILL after a short grace period, even if the leader exits
while a descendant remains). On Windows a new process
group is requested and the direct child is terminated/killed; Python stdlib handling
does not reliably kill arbitrary descendants, so descendant cleanup must be handled
by the Windows command or an external job-object wrapper.

The runner stays in the foreground so the coordinator owns the session. Ctrl-C marks
the job `interrupted` after attempting cleanup. `status` never trusts a PID by itself.
For a non-terminal state it reports `liveness: "running"` and `live: true` only when
`pid` exists and its current `ps` start marker equals the recorded `pid_start`. A
missing PID, a marker mismatch, or unavailable identity inspection is reported as
`live: null` and `liveness: "unknown/stale"`; this includes older state files without
`pid_start`. A PID that is provably absent is reported as `live: false` and
`liveness: "not-running"`. Terminal states are always reported as not running.
`status` never resumes or launches a command. An `unknown/stale` state must be
checked manually from the exact process identity and artifacts before any cleanup or
retry; the runner never kills a guessed PID.

Checkpointing and resume semantics belong to the trainer. A trainer may explicitly
inspect this state and its own checkpoints, choose a new unique job ID, and start a
new command; this runner does not blindly resume a stale or interrupted job.
