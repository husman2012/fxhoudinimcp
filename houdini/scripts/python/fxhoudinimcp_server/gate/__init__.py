"""Gate package: security gate middleware and install_gate() entry point.

Wraps fxhoudinimcp_server.dispatcher.dispatch() with a security chokepoint
backed by the homedini.dcc.mcp_gate pure core.  All 179 registered handlers
are gated by construction — no per-handler opt-in required.

Usage:
    from fxhoudinimcp_server.gate import install_gate
    install_gate()  # call once after handlers are registered, before hwebserver starts
"""

from __future__ import annotations

from fxhoudinimcp_server.gate.middleware import install_gate

__all__ = ["install_gate"]
