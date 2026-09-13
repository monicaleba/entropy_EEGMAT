"""
augment_eegmat.py

Legitimate, leakage-safe data augmentation for the EEGMAT dataset, to
address the class imbalance between "Good" (nonstress, ~26 subjects) and
"Bad" (stress, ~10 subjects) performers.

WHY THIS EXISTS
----------------
Salankar et al. (2021, J. Healthcare Engineering, doi:10.1155/2021/2146369)
balance EEGMAT by "replicating the data channelwise" for the minority
class -- i.e. literal duplication of an existing subject's raw signal,
relabeled as a new "dummy" participant. That does not add information,
and it is dangerous under a subject-grouped train/test split: if the
duplicate is treated as an independent subject/group, it can land in the
test set while a near-identical copy of the same signal sits in the
training set, silently reproducing the exact epoch-level leakage that a
GroupShuffleSplit/GroupKFold is meant to prevent.

This script instead generates genuinely perturbed minority-class signals
(additive noise, amplitude scaling, circular time-shift), and -- crucially
-- encodes the SOURCE subject's ID in every augmented file's name, so that
augmented copies can be (and MUST be) grouped with their source subject
in any downstream group-aware split. An augmented file is never a
substitute for real independent data; it only lets the network see more
varied instances of the same subject's stress/relax pattern.

AUGMENTATION TECHNIQUES (applied on the continuous raw signal, per file)
--------------------------------------------------------------------------
1. Additive Gaussian noise   -- simulates sensor / thermal noise, scaled
                                 per-channel to a fraction of that
                                 channel's own std (SNR-controlled, not a
                                 fixed absolute magnitude).
2. Amplitude scaling          -- simulates electrode contact / impedance
                                 variability; one scale factor per
                                 channel, drawn close to 1.0.
3. Circular time-shift         -- simulates task-onset timing jitter;
                                 whole-signal circular roll, channels
                                 shifted together (so cross-channel
                                 timing relationships are preserved).

All three are applied together for every augmented copy, each with an
independently drawn random perturbation, so successive augmented copies
of the same file are not identical to each other either.

CRITICAL — DOWNSTREAM GROUPING
--------------------------------
Augmented files are named:
    <original_stem>_aug<k><original_suffix>
e.g.  subject07.edf  ->  subject07_aug1.edf, subject07_aug2.edf, ...

When you build the group array for GroupShuffleSplit / GroupKFold in your
training script, derive the group ID from the SOURCE subject (strip the
"_augN" suffix) -- NOT from a fresh per-file counter. A ready-to-use
helper, `get_group_key()`, is provided at the bottom of this file; import
it into your training script:

    from augment_eegmat import get_group_key
    group_key = get_group_key(filename)   # "subject07" for both the
                                           # original and every augmented
                                           # copy of it

If every file (original or augmented) is instead given its own unique
group ID, GroupShuffleSplit can still place a subject's original file in
train and one of its augmented copies in test (or vice versa) -- which
reintroduces leakage just as effectively as duplication does. Grouping by
source subject is what makes this augmentation legitimate.

USAGE
-----
    python augment_eegmat.py \\
        --data_dir data_BBC \\
        --minority_class Bad \\
        --balance \\
        --noise_std_frac 0.05 \\
        --scale_range 0.9 1.1 \\
        --max_shift_frac 0.05 \\
        --seed 42

    # Or a fixed number of augmented copies per source file:
    python augment_eegmat.py --data_dir data_BBC --minority_class Bad --n_aug 3

Requires: mne (>=1.6 recommended for EDF export), numpy, and either the
`edfio` or `pyedflib` package installed as MNE's EDF-writer backend
(`pip install edfio`).
"""

import argparse
import os
import re
import sys

import numpy as np
import mne


# ---------------------------------------------------------------------
# 1. Group-key helper -- import this into your training script
# ---------------------------------------------------------------------
_AUG_SUFFIX_RE = re.compile(r"_aug\d+$")


def get_group_key(filename: str) -> str:
    """
    Return the group key (source-subject identifier) for a filename,
    stripping any trailing "_aug<k>" suffix added by this script and the
    file extension.

    "subject07.edf"       -> "subject07"
    "subject07_aug1.edf"  -> "subject07"
    "subject07_aug12.edf" -> "subject07"

    Use this (not a fresh per-file counter) when building the `groups`
    array for GroupShuffleSplit / GroupKFold, so a source file and every
    augmented copy of it are always kept in the same split.
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    return _AUG_SUFFIX_RE.sub("", stem)


# ---------------------------------------------------------------------
# 2. Augmentation core
# ---------------------------------------------------------------------
def augment_signal(
    data: np.ndarray,
    rng: np.random.Generator,
    noise_std_frac: float,
    scale_range: tuple,
    max_shift_frac: float,
) -> np.ndarray:
    """
    Apply additive noise + amplitude scaling + circular time-shift to a
    (n_channels, n_samples) array. Returns a new array; does not modify
    `data` in place.
    """
    n_channels, n_samples = data.shape
    augmented = data.copy()

    # --- 1. Additive Gaussian noise, scaled per channel to its own std ---
    channel_std = augmented.std(axis=1, keepdims=True)
    noise = rng.normal(
        loc=0.0,
        scale=noise_std_frac * channel_std,
        size=augmented.shape,
    )
    augmented = augmented + noise

    # --- 2. Amplitude scaling, one factor per channel ---
    low, high = scale_range
    scale_factors = rng.uniform(low, high, size=(n_channels, 1))
    augmented = augmented * scale_factors

    # --- 3. Circular time-shift, same shift for all channels ---
    max_shift = int(max_shift_frac * n_samples)
    if max_shift > 0:
        shift = int(rng.integers(-max_shift, max_shift + 1))
        augmented = np.roll(augmented, shift, axis=1)

    return augmented


# ---------------------------------------------------------------------
# 3. EDF I/O
# ---------------------------------------------------------------------
def load_raw(file_path: str) -> mne.io.Raw:
    return mne.io.read_raw_edf(file_path, preload=True, verbose=False)


def save_augmented_edf(raw_original: mne.io.Raw, new_data: np.ndarray, out_path: str) -> None:
    """
    Build a new Raw object from the augmented array (reusing the
    original's channel/measurement info) and export it to EDF.
    """
    raw_aug = mne.io.RawArray(new_data, raw_original.info.copy(), verbose=False)
    raw_aug.set_meas_date(raw_original.info["meas_date"])
    raw_aug.export(out_path, fmt="edf", overwrite=True, verbose=False)


# ---------------------------------------------------------------------
# 4. Main balancing routine
# ---------------------------------------------------------------------
def list_edf_files(class_dir: str):
    return sorted(
        f for f in os.listdir(class_dir) if f.lower().endswith(".edf")
    )


def compute_class_counts(data_dir: str):
    counts = {}
    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            counts[class_name] = len(list_edf_files(class_dir))
    return counts


def augment_class(
    data_dir: str,
    minority_class: str,
    n_aug: int,
    noise_std_frac: float,
    scale_range: tuple,
    max_shift_frac: float,
    seed: int,
) -> None:
    class_dir = os.path.join(data_dir, minority_class)
    if not os.path.isdir(class_dir):
        raise FileNotFoundError(f"Class directory not found: {class_dir}")

    source_files = [
        f for f in list_edf_files(class_dir) if not _AUG_SUFFIX_RE.search(
            os.path.splitext(f)[0]
        )
    ]

    if not source_files:
        print(f"No source (non-augmented) EDF files found in {class_dir}.")
        return

    print(f"Found {len(source_files)} source file(s) in '{minority_class}'.")
    print(f"Generating {n_aug} augmented copy/copies per source file "
          f"({len(source_files) * n_aug} new files total).\n")

    rng = np.random.default_rng(seed)

    for filename in source_files:
        file_path = os.path.join(class_dir, filename)
        stem, ext = os.path.splitext(filename)

        print(f"Processing: {filename}")
        raw = load_raw(file_path)
        data, _ = raw[:, :]

        for k in range(1, n_aug + 1):
            augmented_data = augment_signal(
                data,
                rng=rng,
                noise_std_frac=noise_std_frac,
                scale_range=scale_range,
                max_shift_frac=max_shift_frac,
            )

            out_filename = f"{stem}_aug{k}{ext}"
            out_path = os.path.join(class_dir, out_filename)

            save_augmented_edf(raw, augmented_data, out_path)
            print(f"  -> wrote {out_filename}  "
                  f"(group key: '{get_group_key(out_filename)}')")

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(
        description="Leakage-safe EDF augmentation for balancing EEGMAT."
    )
    parser.add_argument("--data_dir", required=True,
                         help="Root folder containing one subfolder per class "
                              "(e.g. data_BBC/Good, data_BBC/Bad).")
    parser.add_argument("--minority_class", required=True,
                         help="Name of the class subfolder to augment "
                              "(e.g. 'Bad').")

    balance_group = parser.add_mutually_exclusive_group(required=True)
    balance_group.add_argument(
        "--n_aug", type=int,
        help="Fixed number of augmented copies to generate per source file."
    )
    balance_group.add_argument(
        "--balance", action="store_true",
        help="Automatically compute n_aug so the minority class's file "
             "count approximately matches the majority class's file count."
    )

    parser.add_argument("--noise_std_frac", type=float, default=0.05,
                         help="Std of additive noise, as a fraction of "
                              "each channel's own std. Default 0.05.")
    parser.add_argument("--scale_range", type=float, nargs=2,
                         default=(0.9, 1.1), metavar=("LOW", "HIGH"),
                         help="Range for per-channel amplitude scaling. "
                              "Default 0.9 1.1.")
    parser.add_argument("--max_shift_frac", type=float, default=0.05,
                         help="Max circular time-shift, as a fraction of "
                              "signal length. Default 0.05.")
    parser.add_argument("--seed", type=int, default=42,
                         help="Random seed, for reproducibility.")

    args = parser.parse_args()

    if args.balance:
        counts = compute_class_counts(args.data_dir)
        if args.minority_class not in counts:
            print(f"Error: class '{args.minority_class}' not found under "
                  f"{args.data_dir}. Found: {list(counts.keys())}")
            sys.exit(1)

        minority_count = counts[args.minority_class]
        majority_count = max(counts.values())
        print(f"Class counts (source files): {counts}")

        if minority_count == 0:
            print("Error: minority class has zero source files.")
            sys.exit(1)

        # How many augmented copies per source file to roughly reach
        # majority_count total files in the minority class.
        n_aug = max(0, -(-((majority_count - minority_count)) // minority_count))
        # ceil division, so we reach or slightly exceed majority_count
        print(f"Auto-computed n_aug = {n_aug} "
              f"(minority {minority_count} -> "
              f"~{minority_count * (1 + n_aug)} after augmentation, "
              f"target {majority_count})\n")
    else:
        n_aug = args.n_aug

    if n_aug <= 0:
        print("Nothing to do: minority class is already at or above the "
              "majority class's file count (n_aug <= 0).")
        return

    augment_class(
        data_dir=args.data_dir,
        minority_class=args.minority_class,
        n_aug=n_aug,
        noise_std_frac=args.noise_std_frac,
        scale_range=tuple(args.scale_range),
        max_shift_frac=args.max_shift_frac,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
