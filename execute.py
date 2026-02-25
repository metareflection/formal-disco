import hashlib
import os
import subprocess

def execute(cmd, ext, v):
    HOME = os.environ["HOME"]
    TMP_DIR = f"{HOME}/tmp/formal-disco/{ext}/"
    key = hashlib.md5(v.encode("utf-8")).hexdigest()
    dir = f"{TMP_DIR}{key}/"

    os.makedirs(dir, exist_ok=True)

    fn = f"ex.{ext}"
    with open(os.path.join(dir, fn), "w", encoding="utf-8") as f:
        f.write(v)

    result = subprocess.run(
        ["timeout", "-k", "5s", "10s", cmd, fn],
        cwd=dir,
        capture_output=True,
        text=True,
    )

    log = result.stderr
    sys_error_prefix = "sh: line 1:"
    if log.startswith(sys_error_prefix):
        raise RuntimeError(log[len(sys_error_prefix):] + " -- install tool locally")

    return {"status": result.returncode, "log": log, "out": result.stdout}
