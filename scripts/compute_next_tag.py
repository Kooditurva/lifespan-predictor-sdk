#!/usr/bin/env python3
"""Compute the next tag for a release line, and the version an rc promotes to.

The scheme mirrors the algorithm's: a candidate carries `-rcN` and only a
verified build gets the bare `vX.Y.Z`. It matters here more than it does there,
because this SDK is installed straight from a tag —
`pip install git+https://github.com/kooditurva/lifespan-predictor-sdk@v1.49.2` —
so a bare tag is not a label on an artifact, it *is* the artifact. Publishing
one before the cloud tests have run hands out an untested install.

    compute_next_tag.py next rel-1.49 v1.49.1 v1.49.2-rc1  -> v1.49.2-rc2
    compute_next_tag.py promote v1.49.2-rc2                -> v1.49.2

Tags are read from the command line rather than from the repository so the rule
can be tested without building a history for every case.
"""

import re
import sys

TAG = re.compile(r"^v(\d+)\.(\d+)\.(\d+)(?:-rc(\d+))?$")


def parse(tag):
    """(major, minor, patch, rc or None), or None if the tag is not ours."""
    match = TAG.match(tag.strip())
    if not match:
        return None
    major, minor, patch = (int(match.group(i)) for i in (1, 2, 3))
    return (major, minor, patch, int(match.group(4)) if match.group(4) else None)


def next_tag(branch, tags):
    """The tag a push to `branch` should create.

    A release line that has already shipped its highest patch starts the next
    one; a line whose highest patch is still a candidate produces the next
    candidate. Both cases produce an `-rc`: nothing on this path is entitled to
    a bare tag, which is what `promote` is for.
    """
    line = re.match(r"^rel-(\d+)\.(\d+)$", branch)
    if not line:
        raise SystemExit(f"not a release branch: {branch}")
    major, minor = int(line.group(1)), int(line.group(2))

    parsed = [p for p in map(parse, tags) if p and p[0] == major and p[1] == minor]
    if not parsed:
        return f"v{major}.{minor}.1-rc1"

    patch = max(p[2] for p in parsed)
    on_this_patch = [p for p in parsed if p[2] == patch]

    if any(p[3] is None for p in on_this_patch):
        # It shipped. The next change to this line is a new patch.
        return f"v{major}.{minor}.{patch + 1}-rc1"

    return f"v{major}.{minor}.{patch}-rc{max(p[3] for p in on_this_patch) + 1}"


def promote(tag):
    """The bare tag an rc becomes once it has passed the cloud tests."""
    parsed = parse(tag)
    if not parsed or parsed[3] is None:
        raise SystemExit(f"not a release candidate: {tag}")
    return f"v{parsed[0]}.{parsed[1]}.{parsed[2]}"


def main(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__)
    if argv[1] == "next":
        print(next_tag(argv[2], argv[3:]))
    elif argv[1] == "promote":
        print(promote(argv[2]))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
