#!/usr/bin/env python3
"""Execute plain Python notebook cells in a fresh process for local rehearsal.

This is not a Jupyter/Kaggle kernel run. Use the bounded job runner, explicit
Team 23 inputs, and a fresh output directory. Executed outputs belong in results/.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback


def rehearse(notebook: Path, executed: Path, report: Path) -> dict:
    if executed.exists() or report.exists():
        raise FileExistsError("rehearsal outputs must be fresh")
    document = json.loads(notebook.read_text(encoding="utf-8"))
    if document.get("nbformat") != 4:
        raise ValueError("expected notebook format 4")
    namespace = {"__name__": "__main__"}
    source_hash = hashlib.sha256(notebook.read_bytes()).hexdigest()
    started = time.monotonic()
    cells = []
    error = None
    for index, cell in enumerate(document["cells"]):
        if cell["cell_type"] != "code":
            continue
        code = "".join(cell["source"])
        cell_start = time.monotonic()
        output = io.StringIO()
        cell["execution_count"] = len(cells) + 1
        try:
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                exec(compile(code, f"{notebook.name}:cell-{index}", "exec"), namespace)
                runtime = namespace.get("runtime_root")
                if runtime is not None:
                    for module_name, module in list(sys.modules.items()):
                        if module_name == "scripts" or module_name.startswith("scripts."):
                            origin = getattr(module, "__file__", None)
                            if origin and not Path(origin).resolve().is_relative_to(Path(runtime).resolve()):
                                raise RuntimeError(f"notebook reused source outside its bundle: {module_name}")
        except BaseException as exc:
            error = {"cell": index, "type": type(exc).__name__, "message": str(exc)}
            traceback.print_exc(file=output)
        cell["outputs"] = [{"output_type": "stream", "name": "stdout", "text": output.getvalue().splitlines(True)}]
        cells.append({"index": index, "sha256": hashlib.sha256(code.encode()).hexdigest(), "elapsed_seconds": time.monotonic() - cell_start})
        if error is not None:
            break
    result = {
        "status": "failed" if error else "completed",
        "execution_engine": "fresh Python process executing notebook cells; not Jupyter or Kaggle",
        "notebook_sha256": source_hash,
        "cells": cells,
        "elapsed_seconds": time.monotonic() - started,
        "error": error,
        "source_bundle_sha256": namespace.get("SOURCE_BUNDLE_SHA256"),
        "runtime_root": str(namespace.get("runtime_root")) if namespace.get("runtime_root") else None,
        "declared_inputs": {key: os.environ.get(key) for key in ("TEAM23_DATA_DIR", "TEAM23_OUTPUT_DIR", "TEAM23_PART1_DIR", "TEAM23_TRANSLATION_DEVICE")},
    }
    executed.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    executed.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    result["executed_notebook_sha256"] = hashlib.sha256(executed.read_bytes()).hexdigest()
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--executed-notebook", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = rehearse(args.notebook, args.executed_notebook, args.report)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
