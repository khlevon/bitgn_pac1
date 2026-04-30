"""Entry point for the PAC1 challenge agent (run mode, uses Qwen self-hosted model)."""
import asyncio
import sys

import config
from harness import print_summary, run_benchmark


async def main() -> None:
    task_filter = sys.argv[1:] or None
    results = await run_benchmark(
        task_filter=task_filter,
        program_path=config.OPTIMIZED_AGENT_PATH,
    )
    print_summary(results)


if __name__ == "__main__":
    asyncio.run(main())
