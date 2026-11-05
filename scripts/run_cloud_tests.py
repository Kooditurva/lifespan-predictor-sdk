#!/usr/bin/env python3
"""Run the SDK's tests against a real algorithm: the same run the release makes.

    python scripts/run_cloud_tests.py                 # build, local tests, cloud tests
    python scripts/run_cloud_tests.py -- -k inference # pass arguments to pytest
    python scripts/run_cloud_tests.py --notebook      # then run the example notebooks too
    python scripts/run_cloud_tests.py resolve         # print the ARN and its region

It builds the wheel into dist/, installs it in a fresh virtual environment, runs
the local tests, and then the cloud tests against the installed wheel. A local
failure stops it before anything is billed. The cloud tests start SageMaker
jobs in your account, and those are billed. With --notebook, a passing run then
executes the example notebooks in examples/ with the same wheel and keeps their
cleaned outputs (scripts/record_notebook.py), as a release does.

The release runs the same steps as separate jobs, each one a subcommand:

    python scripts/run_cloud_tests.py build             # wheel into dist/, local tests
    python scripts/run_cloud_tests.py cloud             # cloud tests on dist/'s wheel
    python scripts/run_cloud_tests.py record NOTEBOOK   # one notebook with that wheel

`cloud` and `record` build nothing: they install the one wheel `build` left in
dist/, so every job tests the same wheel.

What it needs:

    SAGEMAKER_ROLE_ARN       the SageMaker execution role your jobs run as
    AWS credentials          a profile, SSO session or environment, as for the AWS CLI
    SAGEMAKER_ALGORITHM_ARN  optional: your subscription's algorithm ARN, or pass
                             --algorithm-arn. Defaults to the free tier's in eu-west-1.
                             The region is read from it.
    SAGEMAKER_BUCKET         optional: a bucket for the data and models. SageMaker's
                             default bucket otherwise.

`build` needs none of them. The GitHub workflow sdk-cloud-tests.yml runs this
script too. Standard library only, so it runs in a fresh clone.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# The free tier's public ARN, written in when the SDK is published.
DEFAULT_ARN = "arn:aws:sagemaker:eu-west-1:985815980388:algorithm/free-tier-lifespan-predictor-m-61cb7b65231331099f089cbf75f4410e"
ARN = re.compile(
    r"^arn:aws:sagemaker:([a-z0-9-]+):[0-9]{12}:algorithm/[A-Za-z0-9._-]+$"
)
ROOT = Path(__file__).resolve().parent.parent
CHECK_CREDENTIALS = "import boto3; boto3.client('sts').get_caller_identity()"

# Every example. `run --notebook` records them in this order, and a failure
# stops the rest; the release records each in a job of its own.
NOTEBOOKS = [
    Path("examples") / "sagemaker_training_demo.ipynb",
    Path("examples") / "sagemaker_coxph_demo.ipynb",
    Path("examples") / "sagemaker_coxph_fd003_demo.ipynb",
]


def resolve(flag, env, default=DEFAULT_ARN):
    """The algorithm ARN to test against, and its region."""
    arn = flag or env.get("SAGEMAKER_ALGORITHM_ARN") or default
    match = ARN.match(arn)
    if not match:
        raise SystemExit(f"Not a SageMaker algorithm ARN: {arn}")
    return arn, match.group(1)


def write_github_output(arn, region, env):
    """Hand the ARN, its region and the notebooks to later workflow steps, when
    run by GitHub. The release's notebook matrix is this list."""
    notebooks = json.dumps([notebook.name for notebook in NOTEBOOKS])
    with open(env["GITHUB_OUTPUT"], "a") as output:
        output.write(f"arn={arn}\nregion={region}\nnotebooks={notebooks}\n")


def _bin(venv, name):
    """An executable in the virtual environment, on any platform."""
    if os.name == "nt":
        return venv / "Scripts" / f"{name}.exe"
    return venv / "bin" / name


def _require_python():
    if tuple(sys.version_info[:2]) < (3, 12):
        found = ".".join(map(str, sys.version_info[:3]))
        raise SystemExit(
            f"The SDK needs Python 3.12 or newer; this is {found}. "
            "Run this script with a newer python."
        )


def _require_role(env):
    if not env.get("SAGEMAKER_ROLE_ARN"):
        raise SystemExit(
            "SAGEMAKER_ROLE_ARN is not set. Set it to the SageMaker execution role "
            "your training jobs run as; scripts/setup-sagemaker-role.sh creates one."
        )


class _Venv:
    """A fresh virtual environment, and the steps run in it from the clone."""

    def __init__(self, env, root, workdir, runner):
        self.env, self.root, self.dist = env, root, root / "dist"
        self.runner = runner or (
            lambda command, env=None, cwd=None: subprocess.run(
                command, env=env, cwd=cwd
            )
        )
        workdir = Path(workdir or tempfile.mkdtemp(prefix="lifespan-cloud-tests-"))
        self.path = workdir / "venv"

    def bin(self, name):
        return _bin(self.path, name)

    def step(self, command, env=None):
        return self.runner(command, env=env or self.env, cwd=self.root).returncode

    def check(self, command):
        code = self.step(command)
        if code:
            raise SystemExit(f"Failed ({code}): {' '.join(map(str, command))}")

    def create(self):
        self.check([sys.executable, "-m", "venv", self.path])

    def build_wheel(self):
        shutil.rmtree(self.dist, ignore_errors=True)
        self.check([self.bin("pip"), "wheel", "--no-deps", "-w", self.dist, self.root])

    def install_wheel(self, *extra):
        """The one wheel in dist/, so what runs is what was built and tested."""
        wheels = sorted(self.dist.glob("*.whl"))
        if len(wheels) != 1:
            raise SystemExit(
                f"Expected one wheel in dist/, found {len(wheels)}. "
                "Run `python scripts/run_cloud_tests.py build` first."
            )
        self.check([self.bin("pip"), "install", *wheels, *extra])

    def pytest(self, marker, pytest_args, env=None):
        # The installed pytest, not `python -m pytest`, which would put the clone
        # first on the path and test the source tree instead of the wheel.
        command = [self.bin("pytest"), self.root / "tests", "-v", "-m", marker]
        return self.step([*command, *pytest_args], env)

    def signed_in(self, arn, region):
        """The environment the billed steps run in, once credentials work."""
        cloud_env = {
            **self.env,
            "SAGEMAKER_ALGORITHM_ARN": arn,
            "AWS_DEFAULT_REGION": region,
        }
        if self.step([self.bin("python"), "-c", CHECK_CREDENTIALS], cloud_env):
            raise SystemExit(
                "No usable AWS credentials. Sign in (aws sso login, or set a profile) "
                "and run this again."
            )
        return cloud_env

    def record(self, notebook, cloud_env):
        script = self.root / "scripts" / "record_notebook.py"
        return self.step([self.bin("python"), script, self.root / notebook], cloud_env)


def _local_tests(venv, pytest_args):
    code = venv.pytest("local", pytest_args)
    if code:
        print("The local tests failed; the cloud tests were not started.", flush=True)
    return code


def build(pytest_args, env, root=ROOT, workdir=None, runner=None):
    """Build the wheel into dist/ and run the local tests against it. Not billed."""
    _require_python()
    venv = _Venv(env, root, workdir, runner)
    venv.create()
    venv.build_wheel()
    venv.install_wheel("pytest")
    return _local_tests(venv, pytest_args)


def cloud(arn, region, pytest_args, env, root=ROOT, workdir=None, runner=None):
    """Run the cloud tests against the wheel in dist/. Billed."""
    _require_python()
    _require_role(env)
    venv = _Venv(env, root, workdir, runner)
    venv.create()
    venv.install_wheel("pytest")
    return venv.pytest("cloud", pytest_args, venv.signed_in(arn, region))


def record(arn, region, notebook, env, root=ROOT, workdir=None, runner=None):
    """Record one example notebook with the wheel in dist/. Billed."""
    if Path(notebook) not in NOTEBOOKS:
        raise SystemExit(
            f"Not an example notebook: {notebook}. "
            f"The examples are {', '.join(map(str, NOTEBOOKS))}."
        )
    _require_python()
    _require_role(env)
    venv = _Venv(env, root, workdir, runner)
    venv.create()
    venv.install_wheel("nbclient", "ipykernel")
    return venv.record(notebook, venv.signed_in(arn, region))


def run(
    arn, region, pytest_args, env, root=ROOT, workdir=None, runner=None, notebook=False
):
    """Build, install and test, then optionally record the example notebooks,
    all in one environment. Returns the exit code of the last step."""
    _require_python()
    _require_role(env)
    venv = _Venv(env, root, workdir, runner)

    print(f"Testing against {arn} ({region})", flush=True)
    venv.create()
    venv.build_wheel()
    venv.install_wheel("pytest")
    code = _local_tests(venv, pytest_args)
    if code:
        return code

    cloud_env = venv.signed_in(arn, region)
    code = venv.pytest("cloud", pytest_args, cloud_env)
    if code or not notebook:
        return code

    venv.check([venv.bin("pip"), "install", "nbclient", "ipykernel"])
    for example in NOTEBOOKS:
        code = venv.record(example, cloud_env)
        if code:
            return code
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the SDK's local and cloud tests against a real algorithm.",
        epilog="Arguments after -- are passed to pytest.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=["run", "resolve", "build", "cloud", "record"],
    )
    parser.add_argument("notebook", nargs="?", help="record: the notebook to record")
    parser.add_argument("--algorithm-arn", help="your subscription's algorithm ARN")
    parser.add_argument(
        "--notebook",
        dest="all_notebooks",
        action="store_true",
        help="run: after the tests pass, run the example notebooks and keep their outputs",
    )
    argv = sys.argv[1:] if argv is None else argv
    pytest_args = []
    if "--" in argv:
        pytest_args = argv[argv.index("--") + 1 :]
        argv = argv[: argv.index("--")]
    args = parser.parse_args(argv)
    if (args.command == "record") != (args.notebook is not None):
        parser.error("record takes one notebook, and only record takes one")

    env = dict(os.environ)
    if args.command == "build":
        return build(pytest_args, env)
    arn, region = resolve(args.algorithm_arn, env)
    if args.command == "resolve":
        print(f"Algorithm ARN: {arn} (region {region})")
        if env.get("GITHUB_OUTPUT"):
            write_github_output(arn, region, env)
        return 0
    if args.command == "cloud":
        return cloud(arn, region, pytest_args, env)
    if args.command == "record":
        return record(arn, region, args.notebook, env)
    return run(arn, region, pytest_args, env, notebook=args.all_notebooks)


if __name__ == "__main__":
    sys.exit(main())
