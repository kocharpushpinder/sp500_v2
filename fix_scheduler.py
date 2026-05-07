"""
Fix run_scheduler.py: subprocess stdout/stderr must be explicitly
redirected so child processes don't inherit nohup's broken file descriptors.
Run from project root: python fix_scheduler.py
"""
from pathlib import Path

path = Path("/Users/sinder/PycharmProjects/sp500_agent_V2/run_scheduler.py")
with open(path) as f:
    content = f.read()

old = '''def _run(args: list, label: str) -> bool:
    log.info(f"  [{label}] starting ...")
    t0 = time.time()
    try:
        r = subprocess.run([PYTHON] + args,
                           cwd=str(Path(__file__).parent), timeout=3600)
        elapsed = time.time() - t0
        if r.returncode == 0:
            log.info(f"  [{label}] done in {elapsed:.0f}s")
            return True
        log.error(f"  [{label}] FAILED (exit {r.returncode})")
        return False
    except Exception as e:
        log.error(f"  [{label}] ERROR: {e}")
        return False'''

new = '''def _run(args: list, label: str) -> bool:
    log.info(f"  [{label}] starting ...")
    t0 = time.time()
    try:
        # Explicitly open stdout/stderr so child processes don't inherit
        # nohup's redirected (and potentially broken) file descriptors.
        log_path = LOG_DIR / "scheduler.log"
        with open(log_path, "a") as log_fd:
            r = subprocess.run(
                [PYTHON] + args,
                cwd=str(Path(__file__).parent),
                stdout=log_fd,
                stderr=log_fd,
                timeout=3600,
            )
        elapsed = time.time() - t0
        if r.returncode == 0:
            log.info(f"  [{label}] done in {elapsed:.0f}s")
            return True
        log.error(f"  [{label}] FAILED (exit {r.returncode})")
        return False
    except Exception as e:
        log.error(f"  [{label}] ERROR: {e}")
        return False'''

if old in content:
    content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)
    print("run_scheduler.py patched — subprocess now writes directly to log file")
else:
    print("Pattern not found. Showing _run function context:")
    idx = content.find("def _run(")
    print(repr(content[idx:idx+500]))
