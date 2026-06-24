"""Hython-smoke: verify _is_graphical_session() returns False under hython/headless.

Run with:
    "C:/Program Files/Side Effects Software/Houdini 21.0.729/bin/hython.exe" \
        tests/startup_sync_hython_smoke.py

Does NOT call start(), does NOT bind :8100.
"""
import os
import sys

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "houdini", "scripts", "python"),
)

from fxhoudinimcp_server import startup  # noqa: E402

result = startup._is_graphical_session()
assert result is False, (
    "_is_graphical_session() returned {!r} under hython — expected False".format(result)
)
print("PASS: _is_graphical_session() is False under hython")
