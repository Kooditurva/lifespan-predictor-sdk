import gc
import os
import zipfile

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler

# N-CMAPSS dataset sub-dataset identifiers (failure mode / competing risk per dataset)
NCMAPSS_FILES = [
    "N-CMAPSS_DS01-005.h5",
    "N-CMAPSS_DS02-006.h5",
    "N-CMAPSS_DS03-012.h5",
    "N-CMAPSS_DS04.h5",
    "N-CMAPSS_DS05.h5",
    "N-CMAPSS_DS06.h5",
    "N-CMAPSS_DS07.h5",
    "N-CMAPSS_DS08c-008.h5",
]

# N-CMAPSS ships eight sub-datasets, each one file.
NCMAPSS_SUBSETS = [1, 2, 3, 4, 5, 6, 7, 8]

# The self-hosted downsampled artifact: the same HDF5 keys, collapsed
# to one row per (unit, cycle), so make_ncmapss_dataframe reads it unchanged.
NCMAPSS_MINI_FILES = [name[: -len(".h5")] + "_mini.h5" for name in NCMAPSS_FILES]

# Every N-CMAPSS variant shares one labelling path. They differ only in
# acquisition -- NASA's archive is monolithic, the two self-hosted sets are one
# object per sub-dataset -- and the competing-risk labelling below must not fork
# between them.
_NCMAPSS_VERSIONS = ("ncmapss", "ncmapss-full", "ncmapss-mini")

# The two variants served from the project's own bucket, one object per
# sub-dataset. "ncmapss" is not among them: it comes from NASA, whole.
_NCMAPSS_HOSTED_VERSIONS = ("ncmapss-full", "ncmapss-mini")

# Where the test-split truncation falls. See _truncate_ncmapss_test_split.
_NCMAPSS_TEST_CUT_SEED = 20260906
_NCMAPSS_TEST_KEEP_FRACTION = (0.4, 0.9)


def _ncmapss_files_for(version):
    """The on-disk file list backing an N-CMAPSS variant."""
    return NCMAPSS_MINI_FILES if version == "ncmapss-mini" else NCMAPSS_FILES


unit_col = "unit"
step_col = "cycles"
event_col = "event"
duration_col = "duration"
group_col = "group"
metadata_cols = [unit_col, step_col, event_col, duration_col, group_col]


def get_feature_cols(df):
    return [col for col in df.columns if col not in metadata_cols]


def make_turbofan_dataframe(training_filename):
    # Define your column names as a list
    column_names = [unit_col, step_col, "setting_1", "setting_2", "setting_3"]
    for i in range(1, 22):
        feature_name = f"f{i}"
        column_names.append(feature_name)
    # Read the file using pandas.read_csv, specifying no header and whitespace delimiter
    df = pd.read_csv(
        training_filename, header=None, delimiter=r"\s+", names=column_names
    )
    df = df.set_index([unit_col, step_col])
    # df = impute(df)
    return df.reset_index()


def make_ncmapss_dataframe(h5_file, split="dev"):
    """Parse an N-CMAPSS HDF5 file into a DataFrame with metadata columns."""
    try:
        import h5py
    except ImportError as e:
        raise ImportError(
            "h5py is required for N-CMAPSS datasets: pip install h5py"
        ) from e

    with h5py.File(h5_file, "r") as f:
        W = np.array(f[f"W_{split}"])
        X_s = np.array(f[f"X_s_{split}"])
        Y = np.array(f[f"Y_{split}"])
        A = np.array(f[f"A_{split}"])

        W_cols = [n.decode() if isinstance(n, bytes) else n for n in f["W_var"][:]]
        X_s_cols = [n.decode() if isinstance(n, bytes) else n for n in f["X_s_var"][:]]
        A_cols = [n.decode() if isinstance(n, bytes) else n for n in f["A_var"][:]]

    a_df = pd.DataFrame(A, columns=A_cols)
    w_df = pd.DataFrame(W, columns=W_cols)
    xs_df = pd.DataFrame(X_s, columns=X_s_cols)
    rul = Y.flatten()

    df = pd.concat([a_df, w_df, xs_df], axis=1)

    unit_name = "unit" if "unit" in a_df.columns else A_cols[0]
    cycle_name = "cycle" if "cycle" in a_df.columns else A_cols[1]

    df = df.rename(columns={unit_name: unit_col, cycle_name: step_col})
    df[unit_col] = df[unit_col].astype(int)
    df[step_col] = df[step_col].astype(int)
    df[duration_col] = rul.astype(np.float32)

    return df


def get_nasa_train_dataframe(path, datasets=[1, 2, 3, 4], version="cmapss"):
    """Load NASA train data.

    Parameters
    ----------
    path : str
        Directory containing the dataset files.
    datasets : list of int
        Which sub-datasets to include (1-4 for C-MAPSS, 1-8 for N-CMAPSS).
    version : str
        ``'cmapss'`` (default) for the original C-MAPSS text files, or one of
        three N-CMAPSS variants carrying competing risks:

        - ``'ncmapss'`` -- NASA's archive, a single 15.7 GB object whatever
          ``datasets`` says. Needs no AWS credentials; needs ``bsdtar``.
        - ``'ncmapss-full'`` -- the same files, republished per sub-dataset in
          this project's bucket, so one sub-dataset costs one object.
        - ``'ncmapss-mini'`` -- the downsampled artifact, one row per flight
          cycle. The only one a test can fetch.

        All three label ``event`` with the failure mode (1-based dataset index)
        through the same code path. They differ only in acquisition.
    """
    if version in _NCMAPSS_VERSIONS:
        df = None
        for i, filename in enumerate(_ncmapss_files_for(version)):
            if (i + 1) not in datasets:
                continue
            h5_path = os.path.join(path, filename)
            df2 = make_ncmapss_dataframe(h5_path, split="dev")
            df2[group_col] = i
            df2[event_col] = i + 1  # failure mode = dataset index (competing risks)
            if df is not None:
                df2[unit_col] = df2[unit_col] + df[unit_col].max()
            df = pd.concat([df, df2], ignore_index=True)
            gc.collect()
        return df

    # Original C-MAPSS (text format)
    df = None
    for i, file in enumerate(
        [
            os.path.join(path, "train_FD001.txt"),
            os.path.join(path, "train_FD002.txt"),
            os.path.join(path, "train_FD003.txt"),
            os.path.join(path, "train_FD004.txt"),
        ]
    ):
        if (i + 1) not in datasets:
            continue
        df2 = make_turbofan_dataframe(file)
        df2[group_col] = i
        if df is not None:
            df2[unit_col] = df2[unit_col] + df[unit_col].max()
        df = pd.concat([df, df2])
        gc.collect()

    train_max_cycles = df.groupby(unit_col)[step_col].max().reset_index()
    train_max_cycles.columns = [unit_col, "max_cycles"]
    df = df.merge(train_max_cycles, on=unit_col, how="left")
    df[duration_col] = df["max_cycles"] - df[step_col] + 1
    df[event_col] = 1
    return df.drop(columns=["max_cycles"])


def _truncate_ncmapss_test_split(df, subset_index):
    """Cut each N-CMAPSS test unit short of failure, C-MAPSS style.

    N-CMAPSS ships test units that run **to failure**, so the last observed RUL
    is 0 for every one of them. Labelling from the last cycle therefore gives a
    constant zero duration, and a survival metric has nothing to work with: no
    pair is comparable on time, so the all-cause concordance is undefined and
    returns 0.5 for every model. C-MAPSS never has this problem because NASA
    truncated its test trajectories early and supplied the remaining life
    separately in ``RUL_FD00x.txt``.

    So do here what NASA did there: keep a leading prefix of each unit's cycles
    and let the RUL at the cut be the label. The keep fraction is drawn per unit
    -- a single fixed cut would make every duration a function of sequence
    length alone -- from a generator seeded by the **sub-dataset index**, so the
    cut is reproducible and does not move when a caller widens ``datasets``.

    Only the test split is truncated. Training units should run to failure;
    that is where the shape of degradation is learned.
    """
    rng = np.random.default_rng(_NCMAPSS_TEST_CUT_SEED + subset_index)
    low, high = _NCMAPSS_TEST_KEEP_FRACTION
    kept = []

    for _, rows in df.groupby(unit_col, sort=True):
        rows = rows.sort_values(step_col)
        n_keep = int(round(rng.uniform(low, high) * len(rows)))
        # Keep something to predict from, drop something to predict about.
        n_keep = max(1, min(len(rows) - 1, n_keep))
        kept.append(rows.iloc[:n_keep])

    return pd.concat(kept, ignore_index=True)


def get_nasa_test_dataframe(path, datasets=[1, 2, 3, 4], version="cmapss"):
    """Load NASA test data and labels.

    Parameters
    ----------
    path : str
        Directory containing the dataset files.
    datasets : list of int
        Which sub-datasets to include.
    version : str
        ``'cmapss'`` (default), or ``'ncmapss'`` / ``'ncmapss-full'`` /
        ``'ncmapss-mini'`` -- see :func:`get_nasa_train_dataframe` for how the
        three N-CMAPSS variants differ.

        For every N-CMAPSS variant, each test unit's sequence is truncated
        before failure (see :func:`_truncate_ncmapss_test_split`), the label
        duration is the ``Y_test`` RUL at that cut, and ``event`` encodes the
        failure mode (1-based dataset index), matching what
        :func:`get_nasa_train_dataframe` assigns.

    Returns
    -------
    test_df : DataFrame
        Feature rows (one row per unit/cycle).
    test_labels_df : DataFrame
        One row per unit with ``unit``, ``duration``, ``event`` columns.
    """
    if version in _NCMAPSS_VERSIONS:
        test_df = None
        label_rows = []

        for i, filename in enumerate(_ncmapss_files_for(version)):
            if (i + 1) not in datasets:
                continue
            h5_path = os.path.join(path, filename)
            df2 = make_ncmapss_dataframe(h5_path, split="test")
            df2 = _truncate_ncmapss_test_split(df2, i)
            df2[group_col] = i

            if test_df is not None:
                df2[unit_col] = df2[unit_col] + test_df[unit_col].max()

            # RUL at the cut = label duration; event = failure mode
            last = df2.groupby(unit_col)[duration_col].last().reset_index()
            last[event_col] = i + 1
            label_rows.append(last)

            test_df = pd.concat([test_df, df2], ignore_index=True)
            gc.collect()

        test_labels_df = pd.concat(label_rows, ignore_index=True)

        # The label belongs to test_labels_df, not to the feature frame.
        # make_ncmapss_dataframe sets duration from Y, so it arrives here on
        # both frames; SurvivalModel.predict rejects it as leakage, and the
        # C-MAPSS branch below never produces it. Drop it so the two versions
        # return the same shape.
        test_df = test_df.drop(columns=[duration_col])

        return test_df, test_labels_df

    test_df = None
    all_durations = []

    for i, files in enumerate(
        [
            (os.path.join(path, "test_FD001.txt"), os.path.join(path, "RUL_FD001.txt")),
            (os.path.join(path, "test_FD002.txt"), os.path.join(path, "RUL_FD002.txt")),
            (os.path.join(path, "test_FD003.txt"), os.path.join(path, "RUL_FD003.txt")),
            (os.path.join(path, "test_FD004.txt"), os.path.join(path, "RUL_FD004.txt")),
        ]
    ):
        if (i + 1) not in datasets:
            continue

        # Load test data for this dataset
        df2 = make_turbofan_dataframe(files[0])

        # Load RUL values (one per unit in this dataset)
        rul_values = np.loadtxt(files[1])

        # Store RUL values for later
        all_durations.extend(rul_values)

        # Adjust unit numbers to be globally unique
        if test_df is not None:
            df2[unit_col] = df2[unit_col] + test_df[unit_col].max()

        # Concatenate
        test_df = pd.concat([test_df, df2])

        # Force garbage collection after each file
        gc.collect()

    # Convert durations to array
    durations = np.array(all_durations, dtype=np.float32)

    # All test units eventually fail (event = 1)
    events = np.ones(len(durations), dtype=np.int32)

    # Create test_labels_df - one row per unit with duration and event
    test_labels_df = test_df.groupby("unit").last().reset_index()[["unit"]]
    test_labels_df[duration_col] = durations
    test_labels_df[event_col] = events

    return test_df, test_labels_df


_NCMAPSS_ZIP_URL = "https://phm-datasets.s3.amazonaws.com/NASA/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip"


def download_ncmapss_dataset(save_dir, datasets=[1]):
    """Download N-CMAPSS HDF5 files from the NASA PHM S3 bucket.

    Downloads and extracts the full zip (~16 GB uncompressed) on first call,
    then skips files that are already present.

    Parameters
    ----------
    save_dir : str
        Directory where the HDF5 files will be placed.
    datasets : list of int
        Which sub-datasets to verify are present after extraction (1-8).
        All files in the zip are always extracted; this only controls the
        post-download presence check.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    needed = {NCMAPSS_FILES[i - 1] for i in datasets}
    missing = [
        f
        for f in needed
        if not os.path.exists(os.path.join(save_dir, f))
        or os.path.getsize(os.path.join(save_dir, f)) < 1_000_000
    ]

    if not missing:
        return

    zip_path = os.path.join(save_dir, "_ncmapss.zip")
    if not os.path.exists(zip_path) or not zipfile.is_zipfile(zip_path):
        print("Downloading N-CMAPSS zip (~16 GB uncompressed)...")
        import subprocess

        result = subprocess.run(
            [
                "curl",
                "-L",
                "--progress-bar",
                "--retry",
                "5",
                "--retry-delay",
                "3",
                "--continue-at",
                "-",
                "-o",
                zip_path,
                _NCMAPSS_ZIP_URL,
            ],
            check=False,
        )
        if result.returncode != 0:
            raise Exception(f"curl failed with exit code {result.returncode}")
    else:
        print(f"Found existing zip at {zip_path}, skipping download.")

    import subprocess

    # bsdtar (libarchive) handles nested zips transparently without writing
    # an intermediate file to disk. Install with: brew install libarchive
    print("Extracting .h5 files (requires bsdtar: brew install libarchive)...")
    proc = subprocess.Popen(
        [
            "bsdtar",
            "-xf",
            zip_path,
            "--include=*.h5",
            "-C",
            save_dir,
            "--strip-components=2",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in proc.stdout:
        line = line.strip()
        if line:
            print(f"  {line}")
    proc.wait()
    if proc.returncode != 0:
        raise Exception(f"bsdtar failed with exit code {proc.returncode}")


def download_nasa_dataset(save_dir, datasets=None, version="cmapss"):
    """Download NASA dataset files.

    Routes to :func:`download_ncmapss_dataset` when ``version='ncmapss'``, to
    :func:`_download_ncmapss_hosted` for ``'ncmapss-full'`` and
    ``'ncmapss-mini'``, otherwise downloads the original C-MAPSS text files.

    The two hosted paths are proportional: they fetch one object per requested
    sub-dataset. NASA's archive is monolithic, so under ``version='ncmapss'``
    ``datasets`` scopes the presence check but never the transfer -- which is
    why ``'ncmapss-full'`` exists.

    On return, every file backing ``datasets`` is present in ``save_dir``.
    ``datasets=None`` means all sub-datasets, matching the defaults of
    :func:`get_nasa_train_dataframe` and :func:`get_nasa_test_dataframe`.
    """
    if version in _NCMAPSS_HOSTED_VERSIONS:
        _download_ncmapss_hosted(
            save_dir,
            datasets if datasets is not None else NCMAPSS_SUBSETS,
            version,
        )
        return
    if version == "ncmapss":
        download_ncmapss_dataset(
            save_dir,
            datasets=datasets if datasets is not None else NCMAPSS_SUBSETS,
        )
        return
    _download_cmapss(save_dir, datasets if datasets is not None else CMAPSS_SUBSETS)


# Below these floors a file is a truncated placeholder rather than a dataset.
# The two hosted variants differ by four orders of magnitude in size, so one
# floor cannot serve both: 1 KB would accept a truncated multi-GiB file, and
# 1 MB would reject every reduced one.
_NCMAPSS_MIN_BYTES = {
    "ncmapss-mini": 1_000,
    "ncmapss-full": 1_000_000,
}

# Published by scripts/build_ncmapss_mini.py. Private -- read requires
# credentials for the project account, unlike NASA's own bucket which is
# anonymously readable and is what version="ncmapss" still uses.
#
# A dedicated bucket, not the CI one: the CI bucket expires every object after
# seven days (infra/github-oidc/main.tf), which is right for scratch and would
# silently delete a published dataset. Overridable for a fork or a re-publish
# under a different account.
_NCMAPSS_MINI_BUCKET = "kooditurva-lifespan-predictor-datasets"
_NCMAPSS_PREFIXES = {
    "ncmapss-mini": "ncmapss-mini/v1",
    "ncmapss-full": "ncmapss-full/v1",
}


def _ncmapss_hosted_file(dataset, version):
    """The single file backing one sub-dataset of a hosted variant.

    C-MAPSS needs a train/test/RUL trio per sub-dataset, hence ``_cmapss_files``
    above; N-CMAPSS is one file each, so this stays singular.

    ``ncmapss-full`` keeps NASA's own filenames because its objects *are* NASA's
    files. That is what lets a directory populated by ``version="ncmapss"`` serve
    ``"ncmapss-full"`` unchanged, and stops either from re-fetching what the
    other already has.
    """
    return _ncmapss_files_for(version)[dataset - 1]


def _validate_ncmapss_subsets(datasets):
    """Reject an impossible sub-dataset before any request is spent on it."""
    unknown = sorted(set(datasets) - set(NCMAPSS_SUBSETS))
    if unknown:
        raise ValueError(
            f"Unknown N-CMAPSS sub-dataset(s) {unknown}; valid values are "
            f"{NCMAPSS_SUBSETS}"
        )


def _missing_ncmapss_hosted_datasets(save_dir, datasets, version):
    """Of the sub-datasets requested, those whose file is absent or truncated."""
    _validate_ncmapss_subsets(datasets)
    floor = _NCMAPSS_MIN_BYTES[version]
    missing = []
    for n in datasets:
        path = os.path.join(save_dir, _ncmapss_hosted_file(n, version))
        if not os.path.exists(path) or os.path.getsize(path) <= floor:
            missing.append(n)
    return missing


def _download_ncmapss_hosted(save_dir, datasets, version):
    """Fetch the self-hosted N-CMAPSS files for ``datasets``.

    Unlike NASA's archive -- one 15.7 GB object whatever you ask for -- both
    hosted variants are one object per sub-dataset, so this transfers exactly
    what was requested.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for n in _missing_ncmapss_hosted_datasets(save_dir, datasets, version):
        _fetch_ncmapss_hosted_object(save_dir, n, version)

    still_missing = _missing_ncmapss_hosted_datasets(save_dir, datasets, version)
    if still_missing:
        names = ", ".join(_ncmapss_hosted_file(n, version) for n in still_missing)
        raise Exception(f"{version} download did not produce {names} in {save_dir}")


def _fetch_ncmapss_hosted_object(save_dir, dataset, version):
    """Download one sub-dataset file from the project's S3 bucket."""
    try:
        import boto3
    except ImportError as e:
        raise ImportError(
            f"boto3 is required for the self-hosted {version} dataset: "
            "pip install boto3"
        ) from e

    bucket = os.environ.get("NCMAPSS_MINI_BUCKET") or _NCMAPSS_MINI_BUCKET
    filename = _ncmapss_hosted_file(dataset, version)
    key = f"{_NCMAPSS_PREFIXES[version]}/{filename}"
    print(f"Downloading s3://{bucket}/{key} ...")
    boto3.client("s3").download_file(bucket, key, os.path.join(save_dir, filename))


_CMAPSS_ZIP_URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"

# C-MAPSS ships four sub-datasets, each as a train/test/RUL trio.
CMAPSS_SUBSETS = [1, 2, 3, 4]

# Below this, a file is a truncated placeholder rather than a dataset.
_CMAPSS_MIN_BYTES = 100


def _cmapss_files(datasets):
    """Filenames the requested C-MAPSS sub-datasets need on disk."""
    unknown = sorted(set(datasets) - set(CMAPSS_SUBSETS))
    if unknown:
        raise ValueError(
            f"Unknown C-MAPSS sub-dataset(s) {unknown}; valid values are "
            f"{CMAPSS_SUBSETS}"
        )
    return [
        f"{prefix}_FD00{n}.txt" for n in datasets for prefix in ("train", "test", "RUL")
    ]


def _missing_cmapss_files(save_dir, datasets):
    """Of the files ``datasets`` needs, those absent or implausibly small."""
    return [
        name
        for name in _cmapss_files(datasets)
        if not os.path.exists(os.path.join(save_dir, name))
        or os.path.getsize(os.path.join(save_dir, name)) <= _CMAPSS_MIN_BYTES
    ]


def _download_cmapss(save_dir, datasets):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # The presence of one sub-dataset says nothing about the others: this
    # repository tracks the FD001 trio, so a sentinel on train_FD001.txt is
    # satisfied on every checkout and the download never runs.
    if not _missing_cmapss_files(save_dir, datasets):
        return

    _fetch_cmapss_archive(save_dir)

    still_missing = _missing_cmapss_files(save_dir, datasets)
    if still_missing:
        raise Exception(
            f"C-MAPSS download did not produce {', '.join(still_missing)} "
            f"in {save_dir}"
        )


def _fetch_cmapss_archive(save_dir):
    """Download and extract the C-MAPSS archive, which holds all four subsets."""
    zip_path = os.path.join(save_dir, "_cmapss.zip")
    print("Downloading C-MAPSS zip...")
    with requests.get(_CMAPSS_ZIP_URL, stream=True) as r:
        if r.status_code != 200:
            raise Exception(f"Failed to download C-MAPSS zip: HTTP {r.status_code}")
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                f.write(chunk)

    # The zip contains a folder with a nested CMAPSSData.zip
    inner_zip = os.path.join(save_dir, "_cmapss_inner.zip")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            if member.endswith("CMAPSSData.zip"):
                with zf.open(member) as src, open(inner_zip, "wb") as dst:
                    dst.write(src.read())
                break
    with zipfile.ZipFile(inner_zip, "r") as zf:
        zf.extractall(save_dir)


def load_nasa_data(data_dir):
    # Columns based on dataset description
    cols = ["unit", "cycles", "os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]

    train_path = os.path.join(data_dir, "train_FD001.txt")
    test_path = os.path.join(data_dir, "test_FD001.txt")
    rul_path = os.path.join(data_dir, "RUL_FD001.txt")

    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=cols)
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=cols)
    test_rul = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["rul"])

    return train_df, test_df, test_rul
