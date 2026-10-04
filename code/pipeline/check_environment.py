#!/usr/bin/env python3
"""Print non-sensitive environment facts needed by the pipeline."""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import sys


def command_version(command: str, *args: str) -> str:
    path = shutil.which(command)
    if not path:
        return "missing"
    try:
        result = subprocess.run(
            [path, *args], capture_output=True, text=True, timeout=10, check=False
        )
    except OSError as exc:
        return f"error: {exc.__class__.__name__}"
    line = (result.stdout or result.stderr).strip().splitlines()
    return line[0] if line else f"exit={result.returncode}"


print(f"python={sys.version.split()[0]}")
print(f"platform={platform.platform()}")
print(f"cwd_is_repo={os.path.isfile('README.md')}")
for command, args in (("git", ("--version",)), ("docker", ("--version",)), ("nvidia-smi", ("--query-gpu=name", "--format=csv,noheader"))):
    print(f"{command}={command_version(command, *args)}")
for module in ("numpy", "torch"):
    print(f"python_module_{module}={importlib.util.find_spec(module) is not None}")
