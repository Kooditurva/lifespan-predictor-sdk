#!/usr/bin/env python3
"""Tag a tested SDK and publish it as a GitHub pre-release.

    python scripts/publish_prerelease.py check
    python scripts/publish_prerelease.py publish --algorithm-arn ARN

`check` runs before anything is billed: it stops the run if this version is
already released, because a released version never moves. `publish` runs after
the local tests, the cloud tests and the sample notebook have passed. It commits
the notebook's recorded outputs on top of the tested commit, tags that commit
`vX.Y.Z`, and publishes a pre-release with the wheel in dist/. A pre-release of
the same version, from an earlier run, is replaced.

It never publishes a full release. A maintainer does that, in GitHub, by
unticking "Set as a pre-release".

The version is the one in pyproject.toml: the algorithm version the SDK is
tested against. Standard library only; it shells out to git and the GitHub CLI.
"""

import argparse
import datetime
import json
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = "examples/marketplace_sample.ipynb"


def read_version(root=ROOT):
    """The version this tree builds, from pyproject.toml."""
    with open(Path(root) / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["version"]


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
    """None, or whether v{version}'s release is a pre-release."""
    result = run(["gh", "release", "view", f"v{version}", "--json", "isPrerelease"])
    if result.returncode:
        return None
    return json.loads(result.stdout)["isPrerelease"]


def check(version, run=_run):
    """Stop if v{version} is already released; a pre-release may be replaced."""
    prerelease = existing_release(version, run)
    if prerelease is False:
        raise SystemExit(
            f"v{version} is already released, and a released version never moves. "
            "Release a new algorithm version, and an SDK for it, instead."
        )
    if prerelease:
        print(f"v{version} is a pre-release; a passing run replaces it.")
    else:
        print(f"v{version} has no release yet.")


def release_notes(version, algorithm_arn, tested_commit, root=ROOT):
    """The changelog, and what this run tested."""
    changelog = Path(root) / "CHANGELOG.md"
    text = changelog.read_text() if changelog.exists() else ""
    today = datetime.date.today().isoformat()
    return (
        f"{text}\n"
        f"Tested on {today} against `{algorithm_arn}`: commit {tested_commit} passed "
        "the local tests, the cloud tests and the sample notebook. The tag's commit "
        "adds the notebook's recorded outputs to it.\n\n"
        "This is a pre-release until a maintainer releases it.\n"
    )


def publish(version, algorithm_arn, tested_commit, run=_run, root=ROOT):
    """Commit the recorded notebook, tag it v{version}, publish a pre-release."""
    prerelease = existing_release(version, run)
    if prerelease is False:
        check(version, run)

    tag = f"v{version}"
    _must(run, ["git", "add", NOTEBOOK])
    _must(
        run,
        [
            "git",
            "-c",
            "user.name=github-actions[bot]",
            "-c",
            "user.email=github-actions[bot]@users.noreply.github.com",
            "commit",
            "--allow-empty",
            "-m",
            f"Record the sample notebook's run against {algorithm_arn}",
        ],
    )
    if prerelease:
        _must(run, ["gh", "release", "delete", tag, "--cleanup-tag", "--yes"])
        run(["git", "tag", "-d", tag])  # the checkout fetched it; absent is fine
    _must(run, ["git", "tag", "-a", tag, "-m", f"Tested against {algorithm_arn}"])
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
            "--prerelease",
            "--verify-tag",
        ],
    )
    print(f"Published {tag} as a pre-release. Release it in GitHub when ready.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["check", "publish"])
    parser.add_argument("--algorithm-arn", help="the ARN the run tested against")
    parser.add_argument("--tested-commit", help="the commit the tests ran on")
    args = parser.parse_args(argv)

    version = read_version()
    if args.command == "check":
        check(version)
        return 0
    if not args.algorithm_arn or not args.tested_commit:
        parser.error("publish needs --algorithm-arn and --tested-commit")
    publish(version, args.algorithm_arn, args.tested_commit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
