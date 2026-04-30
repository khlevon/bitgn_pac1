"""PCM runtime tools as plain Python callables for dspy.ReAct."""
import json
import shlex
from datetime import datetime, timezone
from typing import Callable, Literal

from bitgn.vm.pcm_connect import PcmRuntimeClientSync
from bitgn.vm.pcm_pb2 import (
    ContextRequest,
    DeleteRequest,
    FindRequest,
    ListRequest,
    MkDirRequest,
    MoveRequest,
    ReadRequest,
    SearchRequest,
    TreeRequest,
    WriteRequest,
)
from google.protobuf.json_format import MessageToDict


# ---------------------------------------------------------------------------
# Formatters — keep output shell-like so the LLM sees familiar syntax
# ---------------------------------------------------------------------------

def _tree_entry_lines(entry, prefix: str = "", is_last: bool = True) -> list[str]:
    branch = "└── " if is_last else "├── "
    lines = [f"{prefix}{branch}{entry.name}"]
    child_prefix = f"{prefix}{'    ' if is_last else '│   '}"
    children = list(entry.children)
    for idx, child in enumerate(children):
        lines.extend(_tree_entry_lines(child, prefix=child_prefix, is_last=idx == len(children) - 1))
    return lines


def _fmt_tree(result, root_arg: str, level_arg: int) -> str:
    root = result.root
    if not root.name:
        body = "."
    else:
        lines = [root.name]
        children = list(root.children)
        for idx, child in enumerate(children):
            lines.extend(_tree_entry_lines(child, is_last=idx == len(children) - 1))
        body = "\n".join(lines)
    level = f" -L {level_arg}" if level_arg > 0 else ""
    return f"tree{level} {root_arg or '/'}\n{body}"


def _fmt_list(path: str, result) -> str:
    if not result.entries:
        return f"ls {path}\n."
    body = "\n".join(f"{e.name}/" if e.is_dir else e.name for e in result.entries)
    return f"ls {path}\n{body}"


def _fmt_read(path: str, result, start_line: int, end_line: int, number: bool) -> str:
    if start_line > 0 or end_line > 0:
        s = start_line if start_line > 0 else 1
        e = end_line if end_line > 0 else "$"
        cmd = f"sed -n '{s},{e}p' {path}"
    elif number:
        cmd = f"cat -n {path}"
    else:
        cmd = f"cat {path}"
    return f"{cmd}\n{result.content}"


def _fmt_search(pattern: str, root: str, result) -> str:
    body = "\n".join(
        f"{m.path}:{m.line}:{m.line_text}" for m in result.matches
    ) or "(no matches)"
    return f"rg -n --no-heading -e {shlex.quote(pattern)} {shlex.quote(root)}\n{body}"


def _fmt_find(result) -> str:
    return "\n".join(m.path for m in result.matches) if result.matches else "(none)"


# ---------------------------------------------------------------------------
# Tool factory — captures vm client via closure
# ---------------------------------------------------------------------------

_KIND_MAP: dict[str, int] = {"all": 0, "files": 1, "dirs": 2}


def make_tools(vm: PcmRuntimeClientSync) -> list[Callable]:
    """Return PCM runtime tools bound to *vm* for use with dspy.ReAct."""

    def tree(root: str = "/", level: int = 2) -> str:
        """Show directory tree. level=0 means unlimited depth."""
        result = vm.tree(TreeRequest(root=root, level=level))
        return _fmt_tree(result, root, level)

    def list_dir(path: str = "/") -> str:
        """List immediate contents of a directory."""
        result = vm.list(ListRequest(name=path))
        return _fmt_list(path, result)

    def read(path: str, start_line: int = 0, end_line: int = 0, number: bool = False) -> str:
        """Read file content. start_line/end_line are 1-based (0 = full file)."""
        result = vm.read(ReadRequest(path=path, number=number, start_line=start_line, end_line=end_line))
        return _fmt_read(path, result, start_line, end_line, number)

    def write(path: str, content: str, start_line: int = 0, end_line: int = 0) -> str:
        """Write or patch a file. Omit start/end_line to overwrite the whole file."""
        vm.write(WriteRequest(path=path, content=content, start_line=start_line, end_line=end_line))
        return f"written: {path}"

    def delete(path: str) -> str:
        """Delete a file or directory."""
        vm.delete(DeleteRequest(path=path))
        return f"deleted: {path}"

    def mkdir(path: str) -> str:
        """Create a directory (including intermediate directories)."""
        vm.mk_dir(MkDirRequest(path=path))
        return f"created dir: {path}"

    def move(from_name: str, to_name: str) -> str:
        """Move or rename a file or directory."""
        vm.move(MoveRequest(from_name=from_name, to_name=to_name))
        return f"moved: {from_name} -> {to_name}"

    def find(name: str, root: str = "/", kind: Literal["all", "files", "dirs"] = "all", limit: int = 10) -> str:
        """Find files/dirs whose name contains *name* (case-insensitive substring match). kind: all | files | dirs. Increase limit for exhaustive searches."""
        result = vm.find(FindRequest(root=root, name=name, type=_KIND_MAP[kind], limit=limit))  # type: ignore[arg-type]
        return _fmt_find(result)

    def search(pattern: str, root: str = "/", limit: int = 50) -> str:
        """Search file contents by regex pattern (ripgrep semantics). Use this to find names, emails, IDs, or field values across JSON and text files. Increase limit for large directories."""
        result = vm.search(SearchRequest(root=root, pattern=pattern, limit=limit))
        return _fmt_search(pattern, root, result)

    def context() -> str:
        """Get repository-level metadata including the current timestamp (unixTime field). Use this for any date arithmetic — convert unixTime (seconds since epoch) to a calendar date."""
        result = vm.context(ContextRequest())
        return json.dumps(MessageToDict(result), indent=2)

    return [tree, list_dir, read, write, delete, mkdir, move, find, search, context]


def grounding_snapshot(vm: PcmRuntimeClientSync) -> str:
    """Fetch initial grounding context (tree + AGENTS.md + context) before the agent loop."""
    parts: list[str] = []

    tree_res = vm.tree(TreeRequest(root="/", level=2))
    parts.append(_fmt_tree(tree_res, "/", 2))

    try:
        read_res = vm.read(ReadRequest(path="AGENTS.md"))
        parts.append(f"cat AGENTS.md\n{read_res.content}")
    except Exception:
        pass

    ctx_res = vm.context(ContextRequest())
    ctx_dict = MessageToDict(ctx_res)

    # Add human-readable date fields to make date arithmetic easier for the agent
    if "unixTime" in ctx_dict:
        try:
            unix_ts = int(ctx_dict["unixTime"])
            dt = datetime.fromtimestamp(unix_ts, tz=timezone.utc)
            ctx_dict["date_utc"] = dt.strftime("%Y-%m-%d")
            ctx_dict["datetime_utc"] = dt.isoformat()
        except (ValueError, OSError):
            pass

    parts.append(f"context\n{json.dumps(ctx_dict, indent=2)}")

    return "\n\n".join(parts)
