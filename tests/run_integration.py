#!/usr/bin/env python3
"""Launch the integration test suite inside hython on Windows, macOS, or Linux.

Usage:
    python tests/run_integration.py              # whole integration suite
    python tests/run_integration.py -k pyro      # pytest args pass through

Finds the newest installed Houdini (override with the HYTHON environment
variable pointing at the hython executable) and reuses this interpreter's
pytest installation by appending its site-packages to hython's sys.path
(after Houdini's own packages). Requires a Houdini license seat.
"""

from __future__ import annotations

# Built-in
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_SEARCH_PATTERNS = [
    # Windows
    "C:/Program Files/Side Effects Software/Houdini *",
    # macOS
    "/Applications/Houdini/Houdini*",
    # Linux
    "/opt/hfs*",
]

_HYTHON_SUBPATHS = [
    "bin/hython.exe",
    "bin/hython",
    "Frameworks/Houdini.framework/Versions/Current/Resources/bin/hython",
]


def _version_key(path: Path) -> tuple[int, ...]:
    digits = re.findall(r"\d+", path.name)
    return tuple(int(d) for d in digits) if digits else (0,)


def find_hython() -> Path:
    """Return the hython executable: $HYTHON, newest install, or PATH."""
    env_override = os.environ.get("HYTHON")
    if env_override:
        candidate = Path(env_override)
        if candidate.is_file():
            return candidate
        sys.exit(f"HYTHON is set but does not exist: {env_override}")

    installs: list[Path] = []
    for pattern in _SEARCH_PATTERNS:
        root = Path(pattern.split("*")[0]).parent
        glob = pattern.split("/")[-1]
        if root.is_dir():
            installs.extend(p for p in root.glob(glob) if p.is_dir())

    for install in sorted(installs, key=_version_key, reverse=True):
        for subpath in _HYTHON_SUBPATHS:
            hython = install / subpath
            if hython.is_file():
                return hython

    on_path = shutil.which("hython")
    if on_path:
        return Path(on_path)

    sys.exit(
        "No hython executable found. Install Houdini or set the HYTHON "
        "environment variable to the full path of hython."
    )


def main() -> int:
    hython = find_hython()

    try:
        import pytest  # noqa: F401

        site_packages = Path(pytest.__file__).resolve().parent.parent
    except ImportError:
        sys.exit(
            "pytest is not importable from this Python. Install it first: "
            f"{sys.executable} -m pip install pytest"
        )

    env = os.environ.copy()
    # Only the fork's own pure-Python package goes on PYTHONPATH. This
    # interpreter's site-packages must NOT: PYTHONPATH entries precede
    # hython's own site-packages, and a host site-packages can hold
    # compiled packages built for another Python (numpy, lxml,
    # pydantic_core, ...) that shadow Houdini's and break imports inside
    # hython (seen on H22: PDG's `from lxml import etree` failed). The
    # bootstrap below APPENDS it to sys.path, so pytest is found there
    # while Houdini's packages keep priority.
    python_path = [str(REPO_ROOT / "python")]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    bootstrap = (
        "import sys; sys.path.append(sys.argv.pop(1)); "
        "import pytest; sys.exit(pytest.main(sys.argv[1:]))"
    )

    print(f"Using hython: {hython}")
    command = [
        str(hython),
        "-c",
        bootstrap,
        str(site_packages),
        str(REPO_ROOT / "tests" / "integration"),
        "-q",
        "-s",
        "--durations=15",
        *sys.argv[1:],
    ]
    return subprocess.call(command, env=env, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    sys.exit(main())
