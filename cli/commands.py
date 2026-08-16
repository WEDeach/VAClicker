from __future__ import annotations

import asyncio
import importlib
import inspect
import sys
import threading
from pathlib import Path
from typing import Awaitable, Callable

from src.core import VACore

CommandHandler = Callable[[list[str]], Awaitable[None]]

_registry: dict[str, CommandHandler] = {}
_scripts: dict[str, Callable] = {}
_stop_event = threading.Event()
_core = VACore(signal=_stop_event)


def command(name: str):
    def decorator(fn: CommandHandler) -> CommandHandler:
        _registry[name] = fn
        return fn

    return decorator


async def load_scripts() -> None:
    scripts_dir = Path(__file__).parents[1] / "scripts"
    _scripts.clear()
    for path in scripts_dir.glob("*.py"):
        if path.stem.startswith("_"):
            continue
        module_name = f"scripts.{path.stem}"
        for key in list(sys.modules.keys()):
            if key == module_name or key.startswith(module_name + "."):
                del sys.modules[key]
        module = importlib.import_module(module_name)
        entry = getattr(module, "entrypoint", None)
        if callable(entry):
            _scripts[path.stem] = entry
    print(f"已載入 {len(_scripts)} 個腳本")


def _invoke_script(entry: Callable, core: VACore, args: list[str]) -> None:
    """Pass CLI arguments only to entrypoints that accept a second positional
    argument.
    """
    parameters = tuple(inspect.signature(entry).parameters.values())
    accepts_args = (
        any(
            parameter.kind is inspect.Parameter.VAR_POSITIONAL
            for parameter in parameters
        )
        or len(
            [
                parameter
                for parameter in parameters
                if parameter.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
        )
        >= 2
    )
    if accepts_args:
        entry(core, args)
    else:
        entry(core)


async def dispatch(loop: asyncio.AbstractEventLoop, line: str) -> None:
    parts = line.strip().split()
    if not parts:
        return
    name, args = parts[0].lower(), parts[1:]
    handler = _registry.get(name)
    if handler:
        await handler(args)
    else:
        if name in _scripts:
            _stop_event.clear()
            _invoke_script(_scripts[name], _core, args)
            exit()
        else:
            print(f"未知指令: {name}")


@command("list")
async def _list(args: list[str]) -> None:
    print("可用的腳本:", ", ".join(_scripts.keys()))


@command("reload")
async def _reload(args: list[str]) -> None:
    print("重載中...")
    await load_scripts()


@command("exit")
async def _exit(args: list[str]) -> None:
    raise SystemExit(0)
