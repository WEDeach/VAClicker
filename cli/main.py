from __future__ import annotations

import asyncio

from . import commands


async def cli_loop():
    loop = asyncio.get_running_loop()
    await commands.load_scripts()
    print("CLI 已啟動，輸入 help 查看指令，exit 結束")
    while True:
        line = input("> ")
        await commands.dispatch(loop, line)


def main():
    asyncio.run(cli_loop())


if __name__ == "__main__":
    main()
