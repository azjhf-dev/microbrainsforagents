"""Harness integration shims: deterministic precheck layer for agent loops.

Fork attribution: adapted from drtikov/microbrainsforagents (PolyForm
Noncommercial 1.0.0, Copyright (c) 2026 Dmitry Tikov). See NOTICE.
"""

from .precheck import (
    ALLOW,
    DENY,
    ESCALATE,
    PrecheckBlocked,
    PrecheckGate,
    PrecheckVerdict,
    journal_verdicts,
)
from .pi_coding_agent import PiAgentPrecheck
from .deepseek_harness import DeepSeekHarnessPrecheck

__all__ = [
    "ALLOW",
    "DENY",
    "ESCALATE",
    "PrecheckBlocked",
    "PrecheckGate",
    "PrecheckVerdict",
    "journal_verdicts",
    "PiAgentPrecheck",
    "DeepSeekHarnessPrecheck",
]
