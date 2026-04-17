"""
YAML-based config loader with inheritance and validation.

Usage:
    from config.loader import load_config

    # Load from YAML (new way)
    cfg = load_config('configs/birl_fwd.yaml')
    cfg = load_config('configs/birl_fwd.yaml', overrides={'reward.fwd_vel': 3.0})

    # Load from legacy Python class (backward compat)
    cfg = load_config('BIRL_Fwd')

Config access:
    cfg.task.cfg          # 'BIRL'
    cfg.command.lin_vel_x_range  # [-0.3, 0.7]
    cfg.reward.fwd_vel    # 2.3

Inheritance:
    _base: base.yaml      # deep-merges parent, child overrides parent
"""

import copy
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


# ---------------------------------------------------------------------------
# AttrDict — config object with dot-access
# ---------------------------------------------------------------------------

class CfgNode:
    """Recursive attribute-access dict for config values."""

    def __init__(self, d: Optional[Dict] = None):
        if d is not None:
            for k, v in d.items():
                if isinstance(v, dict):
                    self.__dict__[k] = CfgNode(v)
                else:
                    self.__dict__[k] = v

    def __getattr__(self, key: str) -> Any:
        # Return None for missing keys so getattr(cfg, 'x', default) patterns work
        return None

    def __setattr__(self, key: str, value: Any) -> None:
        self.__dict__[key] = value

    def __getitem__(self, key: str) -> Any:
        try:
            return self.__dict__[key]
        except KeyError:
            return None

    def __setitem__(self, key: str, value: Any) -> None:
        self.__dict__[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__

    def __repr__(self) -> str:
        return f"CfgNode({self.to_dict()})"

    def to_dict(self) -> Dict:
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                d[k] = v.to_dict()
            else:
                d[k] = v
        return d

    def get(self, key: str, default: Any = None) -> Any:
        return self.__dict__.get(key, default)

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        for k, v in self.__dict__.items():
            yield k, v

    def __iter__(self):
        return iter(self.__dict__)


# ---------------------------------------------------------------------------
# Deep merge
# ---------------------------------------------------------------------------

def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge override into base. Override wins on conflicts."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


# ---------------------------------------------------------------------------
# YAML loader with _base inheritance
# ---------------------------------------------------------------------------

_CONFIGS_DIR = Path(__file__).parent.parent / 'configs'


def _resolve_yaml_path(path: str) -> Path:
    """Resolve a config path. Supports relative to configs/ dir or absolute."""
    p = Path(path)
    if p.is_absolute() and p.exists():
        return p
    # Relative to configs/ dir
    candidate = _CONFIGS_DIR / path
    if candidate.exists():
        return candidate
    # Relative to project root
    candidate = Path(__file__).parent.parent / path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Config not found: {path} (searched {_CONFIGS_DIR}, project root)")


def _load_yaml_recursive(path: str, _visited: Optional[set] = None) -> Dict:
    """Load a YAML config, resolving _base inheritance recursively."""
    if _visited is None:
        _visited = set()

    resolved = _resolve_yaml_path(path)
    resolved_str = str(resolved.resolve())
    if resolved_str in _visited:
        raise ValueError(f"Circular config inheritance: {path}")
    _visited.add(resolved_str)

    with open(resolved, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f) or {}

    base_path = raw.pop('_base', None)
    if base_path is not None:
        base_dict = _load_yaml_recursive(base_path, _visited)
        return _deep_merge(base_dict, raw)
    return raw


# ---------------------------------------------------------------------------
# Dotted-key override parsing
# ---------------------------------------------------------------------------

def _apply_overrides(d: Dict, overrides: Dict[str, Any]) -> Dict:
    """Apply dotted-key overrides like {'reward.fwd_vel': 3.0}."""
    d = copy.deepcopy(d)
    for dotted_key, value in overrides.items():
        keys = dotted_key.split('.')
        target = d
        for k in keys[:-1]:
            if k not in target or not isinstance(target[k], dict):
                target[k] = {}
            target = target[k]
        target[keys[-1]] = value
    return d


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

# Required top-level sections
_KNOWN_SECTIONS = {
    'task', 'runner', 'policy', 'algorithm', 'action', 'pd_gains',
    'init_state', 'domain_rand', 'noise_values', 'command', 'terrain',
    'sim', 'asset', 'viewer', 'reward', 'observation', 'phase',
}


def _validate_config(d: Dict, strict: bool = False) -> List[str]:
    """Validate config dict. Returns list of warnings."""
    warnings = []
    if strict:
        for key in d:
            if key not in _KNOWN_SECTIONS and not key.startswith('_'):
                warnings.append(f"Unknown top-level config section: '{key}'")
    # Required fields
    if 'task' not in d or 'cfg' not in d.get('task', {}):
        warnings.append("Missing required field: task.cfg")
    return warnings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(
    path: str,
    overrides: Optional[Dict[str, Any]] = None,
    strict: bool = False,
) -> CfgNode:
    """
    Load a YAML config file with _base inheritance.

    Args:
        path: YAML file path (e.g. 'configs/birl_fwd.yaml' or absolute path).
        overrides: Optional dotted-key overrides, e.g. {'reward.fwd_vel': 3.0}
        strict: If True, warn on unknown top-level sections.

    Returns:
        CfgNode with attribute access.
    """
    d = _load_yaml_recursive(path)

    if overrides:
        d = _apply_overrides(d, overrides)

    warnings = _validate_config(d, strict=strict)
    for w in warnings:
        print(f"[config] WARNING: {w}")

    return CfgNode(d)


def config_to_dict(cfg: CfgNode) -> Dict:
    """Convert a CfgNode back to a plain dict."""
    return cfg.to_dict()


def save_config(cfg: CfgNode, path: str) -> None:
    """Save a CfgNode to YAML."""
    d = cfg.to_dict()
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(d, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
