#!/usr/bin/env python3
"""Run the SDK's tests against a real algorithm: the same run the release makes.

    python scripts/run_cloud_tests.py                 # build, local tests, cloud tests
    python scripts/run_cloud_tests.py -- -k inference # pass arguments to pytest
    python scripts/run_cloud_tests.py --notebook      # then run the sample notebook too
    python scripts/run_cloud_tests.py resolve         # print the ARN and its region

It builds the wheel into dist/, installs it in a fresh virtual environment, runs
the local tests, and then the cloud tests against the installed wheel. A local
failure stops it before anything is billed. The cloud tests start SageMaker
jobs in your account, and those are billed. With --notebook, a passing run then
executes examples/marketplace_sample.ipynb with the same wheel and keeps its
cleaned outputs (scripts/record_notebook.py), as a release does.

What it needs:

    SAGEMAKER_ROLE_ARN       the SageMaker execution role your jobs run as
    AWS credentials          a profile, SSO session or environment, as for the AWS CLI
    SAGEMAKER_ALGORITHM_ARN  optional: your subscription's algorithm ARN, or pass
                             --algorithm-arn. Defaults to the free tier's in eu-west-1.
                             The region is read from it.
    SAGEMAKER_BUCKET         optional: a bucket for the data and models. SageMaker's
                             default bucket otherwise.

The GitHub workflow sdk-cloud-tests.yml runs this script too. Standard library
only, so it runs in a fresh clone.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# The free tier's public ARN, written in when the SDK is published.
DEFAULT_ARN = "arn:aws:sagemaker:eu-west-1:985815980388:algorithm/free-tier-lifespan-predictor-m-227e5356082e3b6ea8f8e5e080c1ef12"
ARN = re.compile(
    r"^arn:aws:sagemaker:([a-z0-9-]+):[0-9]{12}:algorithm/[A-Za-z0-9._-]+$"
)
ROOT = Path(__file__).resolve().parent.parent
CHECK_CREDENTIALS = "import boto3; boto3.client('sts').get_caller_identity()"


def resolve(flag, env, default=DEFAULT_ARN):
    """The algorithm ARN to test against, and its region."""
    arn = flag or env.get("SAGEMAKER_ALGORITHM_ARN") or default
    match = ARN.match(arn)
    if not match:
        raise SystemExit(f"Not a SageMaker algorithm ARN: {arn}")
    return arn, match.group(1)


def write_github_output(arn, region, env):
    """Hand the ARN and region to later workflow steps, when run by GitHub."""
    with open(env["GITHUB_OUTPUT"], "a") as output:
        output.write(f"arn={arn}\nregion={region}\n")


def _bin(venv, name):
    """An executable in the virtual environment, on any platform."""
    if os.name == "nt":
        return venv / "Scripts" / f"{name}.exe"
    return venv / "bin" / name


NOTEBOOK = Path("examples") / "marketplace_sample.ipynb"


def run(
    arn, region, pytest_args, env, root=ROOT, workdir=None, runner=None, notebook=False
):
    """Build, install and test, then optionally record the notebook. Returns the
    exit code of the last step."""
    runner = runner or (
        lambda command, env=None, cwd=None: subprocess.run(command, env=env, cwd=cwd)
    )
    if tuple(sys.version_info[:2]) < (3, 12):
        found = ".".join(map(str, sys.version_info[:3]))
        raise SystemExit(
            f"The SDK needs Python 3.12 or newer; this is {found}. "
            "Run this script with a newer python."
        )
    if not env.get("SAGEMAKER_ROLE_ARN"):
        raise SystemExit(
            "SAGEMAKER_ROLE_ARN is not set. Set it to the SageMaker execution role "
            "your training jobs run as; scripts/setup-sagemaker-role.sh creates one."
        )

    workdir = Path(workdir or tempfile.mkdtemp(prefix="lifespan-cloud-tests-"))
    venv, dist = workdir / "venv", root / "dist"

    def step(command, step_env=env):
        return runner(command, env=step_env, cwd=root).returncode

    def check(command):
        code = step(command)
        if code:
            raise SystemExit(f"Failed ({code}): {' '.join(map(str, command))}")

    print(f"Testing against {arn} ({region})", flush=True)
    check([sys.executable, "-m", "venv", venv])
    shutil.rmtree(dist, ignore_errors=True)
    check([_bin(venv, "pip"), "wheel", "--no-deps", "-w", dist, root])
    check([_bin(venv, "pip"), "install", *sorted(dist.glob("*.whl")), "pytest"])

    # The installed pytest, not `python -m pytest`, which would put the clone
    # first on the path and test the source tree instead of the wheel.
    pytest = [_bin(venv, "pytest"), root / "tests", "-v"]
    code = step([*pytest, "-m", "local", *pytest_args])
    if code:
        print("The local tests failed; the cloud tests were not started.", flush=True)
        return code

    cloud_env = {**env, "SAGEMAKER_ALGORITHM_ARN": arn, "AWS_DEFAULT_REGION": region}
    if step([_bin(venv, "python"), "-c", CHECK_CREDENTIALS], cloud_env):
        raise SystemExit(
            "No usable AWS credentials. Sign in (aws sso login, or set a profile) "
            "and run this again."
        )
    code = step([*pytest, "-m", "cloud", *pytest_args], cloud_env)
    if code or not notebook:
        return code

    check([_bin(venv, "pip"), "install", "nbclient", "ipykernel"])
    return step(
        [
            _bin(venv, "python"),
            root / "scripts" / "record_notebook.py",
            root / NOTEBOOK,
        ],
        cloud_env,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the SDK's local and cloud tests against a real algorithm.",
        epilog="Arguments after -- are passed to pytest.",
    )
    parser.add_argument("command", nargs="?", default="run", choices=["run", "resolve"])
    parser.add_argument("--algorithm-arn", help="your subscription's algorithm ARN")
    parser.add_argument(
        "--notebook",
        action="store_true",
        help="after the tests pass, run the sample notebook and keep its outputs",
    )
    argv = sys.argv[1:] if argv is None else argv
    pytest_args = []
    if "--" in argv:
        pytest_args = argv[argv.index("--") + 1 :]
        argv = argv[: argv.index("--")]
    args = parser.parse_args(argv)

    env = dict(os.environ)
    arn, region = resolve(args.algorithm_arn, env)
    if args.command == "resolve":
        print(f"Algorithm ARN: {arn} (region {region})")
        if env.get("GITHUB_OUTPUT"):
            write_github_output(arn, region, env)
        return 0
    return run(arn, region, pytest_args, env, notebook=args.notebook)


if __name__ == "__main__":
    sys.exit(main())
