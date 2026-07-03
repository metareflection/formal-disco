"""Tiny Python equivalent of `make`.

Targets are registered via the `target` decorator. Run with:
    python -m make <target> [<target> ...]
    python -m make --list
"""

import argparse
import importlib
import pkgutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class Target:
    name: str
    func: Callable
    depends: list[str] = field(default_factory=list)
    generates: list[str] = field(default_factory=list)
    cache: bool = True


TARGETS: dict[str, Target] = {}


def add_target(
    name: str,
    func: Callable,
    depends: list[str] | None = None,
    generates: list[str] | None = None,
    cache: bool = True,
) -> Target:
    if name in TARGETS:
        raise ValueError(f"Duplicate target: {name}")
    t = Target(
        name=name,
        func=func,
        depends=list(depends or []),
        generates=list(generates or []),
        cache=cache,
    )
    TARGETS[name] = t
    return t


def target(
    name: str,
    depends: list[str] | None = None,
    generates: list[str] | None = None,
    cache: bool = True,
):
    def decorator(func: Callable) -> Callable:
        add_target(name, func, depends=depends, generates=generates, cache=cache)
        return func
    return decorator


def _is_up_to_date(t: Target) -> bool:
    # Make-style: skip if every output exists and is at least as new as every
    # output of every dependency target.
    if not t.cache or not t.generates:
        return False
    gen_paths = [Path(p) for p in t.generates]
    if not all(p.exists() for p in gen_paths):
        return False
    oldest_gen = min(p.stat().st_mtime for p in gen_paths)
    for dep_name in t.depends:
        dep = TARGETS[dep_name]
        for dep_path in dep.generates:
            p = Path(dep_path)
            if p.exists() and p.stat().st_mtime > oldest_gen:
                return False
    return True


def _run(t: Target, built: set[str], no_cache: bool = False) -> None:
    if t.name in built:
        return
    for dep_name in t.depends:
        if dep_name not in TARGETS:
            raise ValueError(f"Unknown dependency '{dep_name}' of target '{t.name}'")
        _run(TARGETS[dep_name], built, no_cache=no_cache)
    if not no_cache and _is_up_to_date(t):
        print(f"[make] up-to-date: {t.name}")
    else:
        print(f"[make] building: {t.name}")
        t.func()
    built.add(t.name)


def _import_siblings() -> None:
    pkg_dir = Path(__file__).parent
    for info in pkgutil.iter_modules([str(pkg_dir)]):
        if info.name.startswith("_"):
            continue
        importlib.import_module(f"{__name__}.{info.name}")


def main() -> None:
    _import_siblings()
    parser = argparse.ArgumentParser(prog="python -m make")
    parser.add_argument("targets", nargs="*", help="Target names to build")
    parser.add_argument("--list", action="store_true", help="List available targets")
    parser.add_argument("--no-cache", action="store_true",
                        help="Always rebuild, ignoring the per-target cache check")
    args = parser.parse_args()

    if args.list or not args.targets:
        if not TARGETS:
            print("(no targets registered)")
            return
        width = max(len(n) for n in TARGETS)
        for name in sorted(TARGETS):
            t = TARGETS[name]
            tail = f" -> {', '.join(t.generates)}" if t.generates else ""
            print(f"  {name:<{width}}{tail}")
        return

    built: set[str] = set()
    for name in args.targets:
        if name not in TARGETS:
            sys.exit(f"Unknown target: {name}")
        _run(TARGETS[name], built, no_cache=args.no_cache)
