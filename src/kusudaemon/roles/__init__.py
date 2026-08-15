"""Role provider subsystem (ROLE-CALLS-VIA-BACKENDS-PLAN.md).

Formalizes the interface for reasoning and role model calls and provides
routing through CLI agent backends (opencode, claude, codex) and HTTP (gptme).
"""

from __future__ import annotations

from .backend_provider import BackendRoleProvider
from .factory import LazyRoleProvider, make_role_provider
from .json_io import build_json_instruction, extract_last_json_object
from .protocol import RoleProvider, RoleProviderBase

__all__ = [
    "BackendRoleProvider",
    "LazyRoleProvider",
    "RoleProvider",
    "RoleProviderBase",
    "build_json_instruction",
    "extract_last_json_object",
    "make_role_provider",
]
