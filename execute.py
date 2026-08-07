import os
import shutil
import subprocess
import tempfile

# GNU coreutils 'timeout' (Linux); absent on stock macOS, where we fall back
# to subprocess's own timeout below.
_TIMEOUT_BIN = shutil.which("timeout") or shutil.which("gtimeout")

def execute(cmd, ext, v, timeout=10):
    with tempfile.TemporaryDirectory() as dir:
        fn = f"program.{ext}"
        with open(os.path.join(dir, fn), "w", encoding="utf-8") as f:
            f.write(v)

        if _TIMEOUT_BIN:
            argv = [_TIMEOUT_BIN, "-k", "5s", f"{timeout}s"] + cmd.split() + [fn]
            run_timeout = None
        else:
            argv = cmd.split() + [fn]
            run_timeout = timeout + 5

        try:
            result = subprocess.run(
                argv,
                cwd=dir,
                capture_output=True,
                text=True,
                timeout=run_timeout,
            )
        except subprocess.TimeoutExpired as e:
            def _text(x):
                return x.decode(errors="replace") if isinstance(x, bytes) else (x or "")
            # Mimic GNU timeout: exit code 124 on expiry.
            return {"status": 124, "log": _text(e.stderr) + "\nTimed out.", "out": _text(e.stdout)}

    log = result.stderr
    sys_error_prefix = "sh: line 1:"
    if log.startswith(sys_error_prefix):
        raise RuntimeError(log[len(sys_error_prefix):] + " -- install tool locally")

    return {"status": result.returncode, "log": log, "out": result.stdout}


def get_dafny_errors(program: str, timeout: int = 30) -> str:
    """Run Dafny verify and return stdout/stderr as a string."""
    try:
        cmd = f"dafny verify --verification-time-limit={timeout}"
        result = execute(cmd, "dfy", program, timeout=timeout + 10)
        return f"stdout:\n{result['out']}\n\nstderr:\n{result['log']}"
    except Exception as e:
        return f"Error running Dafny: {e}"
