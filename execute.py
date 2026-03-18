import os
import subprocess
import tempfile

def execute(cmd, ext, v, timeout=10):
    with tempfile.TemporaryDirectory() as dir:
        fn = f"program.{ext}"
        with open(os.path.join(dir, fn), "w", encoding="utf-8") as f:
            f.write(v)

        result = subprocess.run(
            ["timeout", "-k", "5s", f"{timeout}s"] + cmd.split() + [fn],
            cwd=dir,
            capture_output=True,
            text=True,
        )

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
