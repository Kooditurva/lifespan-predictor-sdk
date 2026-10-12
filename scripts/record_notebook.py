#!/usr/bin/env python3
"""Run the sample notebook and keep its outputs, cleaned, in place. Billed.

    python scripts/record_notebook.py examples/marketplace_sample.ipynb

The release runs this after the tests pass, with the tested wheel installed in
this Python, so the notebook's install cell is skipped: the tag it installs does
not exist yet. Credentials, region, role and bucket come from the environment,
which the kernel inherits.

The outputs are then cleaned and checked:

- the account the run used is shown as 111122223333, AWS's documentation
  placeholder;
- what the environment printed is left out: all of stderr, and the SageMaker
  library's `sagemaker.config INFO` lines;
- a cell that failed, an account other than the placeholder and the algorithm's,
  or a path on the machine that ran it, refuses the recording.

A refused recording exits non-zero and leaves the notebook as it was.
"""

import copy
import json
import re
import sys
from pathlib import Path

PLACEHOLDER_ACCOUNT = "111122223333"
# stdout lines the libraries log about the machine, not about the run.
ENVIRONMENT_LINES = ("sagemaker.config INFO",)
ACCOUNT_ID = re.compile(r"(?<!\d)\d{12}(?!\d)")
LOCAL_PATH = re.compile(r"(?:/Users/|/home/|/private/|/tmp/|[A-Z]:\\Users\\)\S*")
ALGORITHM_ARN = re.compile(r"arn:aws:sagemaker:[a-z0-9-]+:(\d{12}):algorithm/")


def _text(value):
    return "".join(value) if isinstance(value, list) else value


def install_cell(notebook):
    """The index of the cell that installs the SDK."""
    return next(
        index
        for index, cell in enumerate(notebook["cells"])
        if "!pip install" in _text(cell["source"])
    )


def replace_account(notebook, account_id):
    """A copy whose text outputs show the placeholder account instead.

    Images are left alone: they are base64, and carry no account.
    """

    def replaced(value):
        if isinstance(value, list):
            return [line.replace(account_id, PLACEHOLDER_ACCOUNT) for line in value]
        return value.replace(account_id, PLACEHOLDER_ACCOUNT)

    scrubbed = copy.deepcopy(notebook)
    for cell in scrubbed["cells"]:
        for output in cell.get("outputs", []):
            for field in ("text", "evalue", "traceback"):
                if field in output:
                    output[field] = replaced(output[field])
            for mime, value in output.get("data", {}).items():
                if mime.startswith("text/"):
                    output["data"][mime] = replaced(value)
    return scrubbed


def without_environment_noise(notebook):
    """A copy without what the environment printed: stderr, and ENVIRONMENT_LINES."""
    kept = copy.deepcopy(notebook)
    for cell in kept["cells"]:
        outputs = []
        for output in cell.get("outputs", []):
            if output.get("name") == "stderr":
                continue
            if output.get("output_type") == "stream":
                lines = _text(output["text"]).splitlines(keepends=True)
                lines = [l for l in lines if not l.startswith(ENVIRONMENT_LINES)]
                if not lines:
                    continue
                output["text"] = lines
            outputs.append(output)
        if "outputs" in cell:
            cell["outputs"] = outputs
    return kept


def output_text(notebook):
    """Everything the outputs show as text. Images are skipped: base64 carries
    no text, and would match a digit pattern by chance."""
    parts = []
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            for field in ("text", "evalue", "traceback"):
                parts.append(_text(output.get(field, "")))
            for mime, value in output.get("data", {}).items():
                if mime.startswith("text/"):
                    parts.append(_text(value))
    return "\n".join(parts)


def problems(notebook):
    """Why the recording must not ship; empty when it may."""
    found = []
    failed = [
        index
        for index, cell in enumerate(notebook["cells"])
        if any(o.get("output_type") == "error" for o in cell.get("outputs", []))
    ]
    if failed:
        found.append(f"cells {failed} failed")
    source = "\n".join(_text(c["source"]) for c in notebook["cells"])
    allowed = {PLACEHOLDER_ACCOUNT} | set(ALGORITHM_ARN.findall(source))
    shown = output_text(notebook)
    for account in sorted(set(ACCOUNT_ID.findall(shown)) - allowed):
        found.append(f"the outputs name account {account}")
    for path in sorted(set(LOCAL_PATH.findall(shown))):
        found.append(f"the outputs name a local path: {path}")
    return found


def cleaned(executed, original, install, account_id):
    """The executed notebook as it ships: the install cell restored, unexecuted,
    and the outputs cleaned."""
    executed = copy.deepcopy(executed)
    executed["cells"][install]["source"] = original["cells"][install]["source"]
    executed["cells"][install]["outputs"] = []
    executed["cells"][install]["execution_count"] = None
    for cell in executed["cells"]:
        cell.get("metadata", {}).pop("execution", None)
    return replace_account(without_environment_noise(executed), account_id)


def record(path, timeout=3600):
    """Execute the notebook at `path` and write it back with its cleaned outputs."""
    import boto3
    import nbformat
    from nbclient import NotebookClient

    path = Path(path)
    original = json.loads(path.read_text())
    install = install_cell(original)
    run = copy.deepcopy(original)
    run["cells"][install]["source"] = ["# Skipped: this Python has the tested wheel.\n"]
    executable = nbformat.reads(json.dumps(run), as_version=4)

    # The kernel is this Python, which has the wheel installed.
    NotebookClient(executable, kernel_name="python3", timeout=timeout).execute()
    executed = json.loads(nbformat.writes(executable))

    account = boto3.client("sts").get_caller_identity()["Account"]
    recorded = cleaned(executed, original, install, account)
    found = problems(recorded)
    if found:
        raise SystemExit("The recording can't ship: " + "; ".join(found) + ".")
    path.write_text(json.dumps(recorded, indent=1, ensure_ascii=False) + "\n")
    print(f"Recorded {path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    record(sys.argv[1])
