"""Live trading must be impossible to switch on by accident."""

from __future__ import annotations

import pytest


def test_importing_the_live_adapter_raises():
    with pytest.raises(Exception) as excinfo:
        import tradebot.execution.live  # noqa: F401
    assert "not implemented" in str(excinfo.value).lower()


def test_the_live_module_is_not_reachable_from_the_package():
    """`from tradebot.execution import *` must not drag it in."""
    import tradebot.execution as ex

    assert "live" not in getattr(ex, "__all__", [])
