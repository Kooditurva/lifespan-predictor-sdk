#!/usr/bin/env python3
"""Tag a tested SDK and create a GitHub draft release for it.

    python scripts/draft_release.py check
    python scripts/draft_release.py draft --algorithm-arn ARN --tested-commit SHA

`check` runs before anything is billed: it stops the run if this version has a
published release. `draft` runs after the local tests, the cloud tests and the
sample notebook have passed. It commits the notebook's recorded outputs on top
of the tested commit, tags that commit `vX.Y.Z`, and creates a draft release
with the wheel in dist/. A draft of the same version, from an earlier run, is
replaced with its tag.

It never publishes. A maintainer does that, in GitHub, after checking the draft.
Publishing is permanent: this repository has immutable releases, so a published
release's tag can never move, and if the release is deleted its tag name can
never be used again. Nothing here deletes a published release.

The version is the one in pyproject.toml: the algorithm version the SDK is
tested against, or a post release of it. Standard library only; it shells out
to git and the GitHub CLI.
"""

import argparse
import datetime
import json
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = "examples/sagemaker_training_demo.ipynb"
# A runner has no git identity, and the commit and the annotated tag both need one.
AS_BOT = [
    "git",
    "-c",
    "user.name=github-actions[bot]",
    "-c",
    "user.email=github-actions[bot]@users.noreply.github.com",
]


def read_version(root=ROOT):
    """The version this tree builds, from pyproject.toml."""
    with open(Path(root) / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["version"]


def next_post_release(version):
    """The post release after `version`: 1.0.3 -> 1.0.3.post1 -> 1.0.3.post2."""
    base, _, post = re.fullmatch(r"(.+?)(\.post(\d+))?", version).groups()
    return f"{base}.post{int(post or 0) + 1}"


def _run(command):
    return subprocess.run(command, capture_output=True, text=True)


def _must(run, command):
    result = run(command)
    if result.returncode:
        raise SystemExit(
            f"Failed: {' '.join(command)}\n{getattr(result, 'stderr', '')}"
        )
    return result


def existing_release(version, run):
    """None, "draft", or "published": the state of v{version}'s release.

    Listed rather than viewed by tag, because a draft is not found by its tag
    name until it is published.
    """
    result = _must(
        run,
        [
            "gh",
            "release",
            "list",
            "--limit",
            "1000",
            "--json",
            "tagName,isDraft,isPrerelease",
        ],
    )
    for release in json.loads(result.stdout):
        if release["tagName"] == f"v{version}":
            return "draft" if release["isDraft"] else "published"
    return None


def check(version, run=_run):
    """Stop if v{version} is published; a draft may be replaced."""
    state = existing_release(version, run)
    if state == "published":
        raise SystemExit(
            f"v{version} is published, and a published release is immutable: its tag "
            "can never move, and if deleted its name can never be used again. Release "
            f"the SDK as {next_post_release(version)} instead (sdk_version in the "
            "private repository's marketplace/listing.yaml), or wait for a new "
            "algorithm version."
        )
    if state == "draft":
        print(f"v{version} is a draft; a passing run replaces it.")
    else:
        print(f"v{version} has no release yet.")
    return state


def release_notes(version, algorithm_arn, tested_commit, root=ROOT):
    """The changelog, and what this run tested."""
    changelog = Path(root) / "CHANGELOG.md"
    text = changelog.read_text() if changelog.exists() else ""
    today = datetime.date.today().isoformat()
    return (
        f"{text}\n"
        f"Tested on {today} against `{algorithm_arn}`: commit {tested_commit} passed "
        "the local tests, the cloud tests and the sample notebook. The tag's commit "
        "adds the notebook's recorded outputs to it.\n"
    )


def draft(version, algorithm_arn, tested_commit, run=_run, root=ROOT):
    """Commit the recorded notebook, tag it v{version}, create a draft release."""
    state = check(version, run)

    tag = f"v{version}"
    _must(run, ["git", "add", NOTEBOOK])
    _must(
        run,
        [
            *AS_BOT,
            "commit",
            "--allow-empty",
            "-m",
            f"Record the sample notebook's run against {algorithm_arn}",
        ],
    )
    if state == "draft":
        # Safe only because no published release ever named this tag.
        _must(run, ["gh", "release", "delete", tag, "--cleanup-tag", "--yes"])
        run(["git", "tag", "-d", tag])  # the checkout fetched it; absent is fine
    _must(run, [*AS_BOT, "tag", "-a", tag, "-m", f"Tested against {algorithm_arn}"])
    _must(run, ["git", "push", "origin", tag])

    notes = Path(tempfile.mkdtemp()) / "notes.md"
    notes.write_text(release_notes(version, algorithm_arn, tested_commit, root))
    wheels = sorted(str(p) for p in (Path(root) / "dist").glob("*.whl"))
    _must(
        run,
        [
            "gh",
            "release",
            "create",
            tag,
            *wheels,
            "--title",
            tag,
            "--notes-file",
            str(notes),
            "--draft",
            "--verify-tag",
        ],
    )
    print(f"Created {tag} as a draft. Check it, then publish it in GitHub.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["check", "draft"])
    parser.add_argument("--algorithm-arn", help="the ARN the run tested against")
    parser.add_argument("--tested-commit", help="the commit the tests ran on")
    args = parser.parse_args(argv)

    version = read_version()
    if args.command == "check":
        check(version)
        return 0
    if not args.algorithm_arn or not args.tested_commit:
        parser.error("draft needs --algorithm-arn and --tested-commit")
    draft(version, args.algorithm_arn, args.tested_commit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
