.PHONY: sync run task optimize optimize-task

sync:
	uv sync

# Run all benchmark tasks (Qwen self-hosted, loads optimized agent if present)
run:
	uv run python src/main.py

# Run specific tasks: make run-task TASKS='t01 t03'
run-task:
	@if [ -z "$(TASKS)" ]; then echo "usage: make run-task TASKS='t01 t03'"; exit 1; fi
	uv run python src/main.py $(TASKS)

# Optimize agent prompts (Claude as teacher/prompt-model, Qwen as task-model)
# Saves result to OPTIMIZED_AGENT_PATH (default: optimized_agent.json)
optimize:
	uv run python src/optimize.py

# Optimize on specific tasks only: make optimize-task TASKS='t01 t02 t03'
optimize-task:
	@if [ -z "$(TASKS)" ]; then echo "usage: make optimize-task TASKS='t01 t02 t03'"; exit 1; fi
	uv run python src/optimize.py $(TASKS)
