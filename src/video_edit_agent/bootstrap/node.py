"""Node/npm/Remotion detection (spec section 9). Optional: never blocks
basic operation, since the simple motion engine covers offline motion
without Node -- Remotion is an enhancement, not a requirement.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass


@dataclass
class NodeStatus:
    node_found: bool
    node_version: str | None
    npm_found: bool
    npm_version: str | None
    remotion_ready: bool
    detail: str


def _version(cmd: str) -> str | None:
    try:
        proc = subprocess.run([cmd, "--version"], capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def check_node() -> NodeStatus:
    node_path = shutil.which("node")
    npm_path = shutil.which("npm") or shutil.which("npm.cmd")

    node_version = _version(node_path) if node_path else None
    npm_version = _version(npm_path) if npm_path else None

    remotion_ready = bool(node_path and npm_path and node_version and npm_version)
    if remotion_ready:
        detail = f"node {node_version}, npm {npm_version} -- Remotion motion engine available via npx"
    elif node_path or npm_path:
        detail = "node/npm partially detected but not both healthy -- Remotion unavailable, local motion engine used"
    else:
        detail = "Node/npm not found -- optional; local motion engine covers offline motion without it"

    return NodeStatus(
        node_found=bool(node_path), node_version=node_version,
        npm_found=bool(npm_path), npm_version=npm_version,
        remotion_ready=remotion_ready, detail=detail,
    )
