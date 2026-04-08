from typing import Optional, Sequence, Tuple, List, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


def train_validate_split(valid_size: float, num_units: int, seed: int = 42) -> Tuple[List[int], List[int]]:
    """
    Split unit indices into train and validation subsets.

    The indices are generated as integers from 1 to ``num_units`` (inclusive),
    shuffled with a reproducible RNG, and then split according to ``valid_size``.

    :param valid_size: Fraction of units to allocate to the validation split,
        typically in the range ``[0.0, 1.0]``.
    :type valid_size: float
    :param num_units: Total number of units to split. Indices are generated from
        ``1`` to ``num_units``.
    :type num_units: int
    :param seed: Random seed for reproducibility.
    :type seed: int
    :returns: A tuple ``(train_idx, valid_idx)`` with lists of unit indices.
    :rtype: Tuple[list[int], list[int]]
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(np.arange(1, num_units + 1))
    split = int(np.floor(valid_size * num_units))
    valid_idx = indices[:split].tolist()
    train_idx = indices[split:].tolist()
    return train_idx, valid_idx

def gen_sequence(arr: np.ndarray, seq_length: int) -> np.ndarray:
    """
    Generate sliding windows (fixed-length sequences) from a 2D array.

    Given an input array of shape ``(n_rows, n_features)``, this function returns
    an array of windows with shape ``(n_windows, seq_length, n_features)``,
    where ``n_windows = n_rows - seq_length + 1``. If there are not enough rows
    to form at least one window, an empty array is returned.

    :param arr: Input data matrix of shape ``(n_rows, n_features)``.
    :type arr: numpy.ndarray
    :param seq_length: Window length (number of rows per sequence).
    :type seq_length: int
    :returns: Stacked sliding windows of shape
        ``(n_windows, seq_length, n_features)``.
    :rtype: numpy.ndarray
    """
    n_rows, n_feats = arr.shape
    n_wins = n_rows - seq_length + 1
    if n_wins <= 0:
        return np.empty((0, seq_length, n_feats))
    windows = np.stack([arr[i:i+seq_length] for i in range(n_wins)], axis=0)
    return windows


def concat_and_discretize(
    x_bin: Sequence[np.ndarray],
    x_num: Sequence[np.ndarray],
    y: np.ndarray,
    c: np.ndarray,
    stride: Optional[int],
    edges: Sequence[float],
) -> Tuple[Sequence[np.ndarray], Sequence[np.ndarray], np.ndarray, np.ndarray]:
    """
    Concatenate-aligned inputs and discretize a 1D continuous target using bin edges.

    This helper optionally downsamples ``y`` and ``c`` using ``stride`` (keeping
    every ``stride``-th element). If, after striding, ``y`` and ``c`` are longer
    than ``x_bin`` along the sample axis, they are truncated to match
    ``len(x_bin)``.

    The target ``y`` is discretized into integer bin indices using
    :func:`pandas.cut` with the provided ``edges``.

    IMPORTANT:
    ``y`` and ``c`` must represent 1-dimensional data of length ``n_samples``.
    Accepted input formats are:

    - NumPy array of shape ``(n_samples,)``
    - pandas ``Series`` of length ``n_samples``
    - pandas ``DataFrame`` with exactly one column

    If a 2D array (e.g., shape ``(n_samples, 1)``) or a multi-column
    ``DataFrame`` is provided, it must be flattened or reduced to a single
    column before discretization, since :func:`pandas.cut` requires 1D input.

    Diagnostic information is printed before and after discretization to help
    validate bin boundaries and class assignments.

    :param x_bin: Binary/categorical input matrix of shape
        ``(n_samples, n_features_bin)``.
    :type x_bin: numpy.ndarray
    :param x_num: Numerical input matrix of shape
        ``(n_samples, n_features_num)``.
    :type x_num: numpy.ndarray
    :param y: Continuous time target (1D), e.g., time in seconds.
    :type y: numpy.ndarray | pandas.Series | pandas.DataFrame (single column)
    :param c: Censoring/event indicator aligned with ``y`` (1D).
    :type c: numpy.ndarray | pandas.Series | pandas.DataFrame (single column)
    :param stride: If not ``None``, downsample ``y`` and ``c`` by ``y[::stride]``.
    :type stride: int | None
    :param edges: Monotonic bin edges passed to :func:`pandas.cut`.
    :type edges: Sequence[float]
    :returns: Tuple ``(x_bin, x_num, y_discrete, c)`` where:
        - ``y_discrete`` is a 1D NumPy array of integer bin indices
        - ``x_bin`` and ``x_num`` are returned unchanged
        - ``c`` is aligned with the discretized ``y``
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """
    # Convert y to a 1D array/Series accepted by pd.cut
    if isinstance(y, pd.DataFrame):
        # Expect a single column dataframe
        if y.shape[1] != 1:
            raise ValueError(f"`y` must be 1D (or a single-column DataFrame). Got shape {y.shape}.")
        y_1d = y.iloc[:, 0]
    elif isinstance(y, pd.Series):
        y_1d = y
    else:
        y_1d = np.asarray(y).reshape(-1)

    # Convert c to 1D (for slicing/alignment); keep original type on return if you want,
    # but typically it's safer to return numpy arrays.
    if isinstance(c, pd.DataFrame):
        if c.shape[1] != 1:
            raise ValueError(f"`c` must be 1D (or a single-column DataFrame). Got shape {c.shape}.")
        c_1d = c.iloc[:, 0]
    elif isinstance(c, pd.Series):
        c_1d = c
    else:
        c_1d = np.asarray(c).reshape(-1)

    # Downsample if needed
    if stride is not None:
        y_1d = y_1d[::stride]
        c_1d = c_1d[::stride]
        if len(y_1d) > len(x_bin):
            y_1d, c_1d = y_1d[: len(x_bin)], c_1d[: len(x_bin)]

    # --- DIAGNOSTICS (PRE-DISCRETIZATION) ---
    print("\n--- PRE-DISCRETIZATION DIAGNOSTICS ---")
    print(f"Total samples: {len(y_1d)}")

    # Extract scalars that format safely (works for numpy scalars and pandas scalars)
    min_time = y_1d.min()
    max_time = y_1d.max()
    mean_time = y_1d.mean()

    # Ensure Python scalars for formatting
    if hasattr(min_time, "item"):
        min_time = min_time.item()
        max_time = max_time.item()

    print(f"Min time: {min_time:.2f} s ({min_time / 60:.2f} min)")
    print(f"Max time: {max_time:.2f} s ({max_time / 3600:.2f} hours)")
    print(f"Mean time: {mean_time / 3600:.2f} hours")

    threshold = edges[-1]
    count_above = int((y_1d > threshold).sum())
    count_equal = int((y_1d == threshold).sum())

    print(f"Defined edges: {list(edges)}")
    print(f"Samples STRICTLY greater than {threshold / 3600:.2f}h: {count_above}")
    print(f"Samples EXACTLY equal to {threshold / 3600:.2f}h: {count_equal}")
    # --------------------------------------

    # --- DISCRETIZATION ---
    y_discrete = pd.cut(y_1d, bins=edges, labels=False, include_lowest=True, right=True)

    safe_zone_idx = len(edges) - 1

    unique, counts = np.unique(
        np.nan_to_num(y_discrete, nan=safe_zone_idx), return_counts=True
    )
    print(f"Assigned distribution: {dict(zip(unique.tolist(), counts.tolist()))}")
    print("--------------------------------------\n")

    y_discrete = np.nan_to_num(y_discrete, nan=safe_zone_idx).astype(int)

    # Return c as numpy 1D to keep things consistent downstream
    c_out = np.asarray(c_1d).reshape(-1)

    return x_bin, x_num, y_discrete, c_out


class SurvivalDataset(Dataset):
    """
    Dataset for survival analysis models.

    This dataset receives binary and numerical feature blocks,
    converts them into tensors, and returns per sample:

    - Concatenated feature vector
    - Discretized time bin label (long tensor)
    - Censoring indicator (float tensor)

    :param x_bin: Binary/categorical feature matrix or sequence of arrays.
    :type x_bin: Sequence[numpy.ndarray] | numpy.ndarray
    :param x_num: Numerical feature matrix or sequence of arrays.
    :type x_num: Sequence[numpy.ndarray] | numpy.ndarray
    :param y: Discretized time bin indices.
    :type y: numpy.ndarray
    :param c: Censoring/event indicator per sample.
    :type c: numpy.ndarray
    """

    def __init__(
        self,
        x_bin: Union[Sequence[np.ndarray], np.ndarray],
        x_num: Union[Sequence[np.ndarray], np.ndarray],
        y: np.ndarray,
        c: np.ndarray
    ):
        # OConvert list-of-arrays to a single NumPy array
        if isinstance(x_bin, list):
            if len(x_bin) > 0:
                x_bin = np.stack(x_bin)
            else:
                x_bin = np.array([], dtype=np.float32)
        elif isinstance(x_bin, np.ndarray):
            pass

        if isinstance(x_num, list):
            if len(x_num) > 0:
                x_num = np.stack(x_num)
            else:
                x_num = np.array([], dtype=np.float32)
        elif isinstance(x_num, np.ndarray):
            pass

        self.x_bin = torch.from_numpy(x_bin).float()
        self.x_num = torch.from_numpy(x_num).float()

        self.y = torch.as_tensor(y, dtype=torch.long)
        self.c = torch.as_tensor(c, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y_idx = self.y[idx]
        c_idx = self.c[idx]
        
        x = torch.cat([self.x_bin[idx], self.x_num[idx]], dim=-1)
        
        return x, y_idx, c_idx


def compute_DataLoaders(
    x_train_bin: Sequence[np.ndarray],
    x_train_num: Sequence[np.ndarray],
    y_train: np.ndarray,
    c_train: np.ndarray,
    x_val_bin: Sequence[np.ndarray],
    x_val_num: Sequence[np.ndarray],
    y_val: np.ndarray,
    c_val: np.ndarray,
    x_test_bin: Sequence[np.ndarray],
    x_test_num: Sequence[np.ndarray],
    y_test: np.ndarray,
    c_test: np.ndarray,
    batch_size: int = 16,
    stride: Optional[int] = 30,
    bin_edges: Optional[Sequence[float]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build DataLoaders for multimodal LSTM survival training.

    This function discretizes the continuous time target into bins (using ``bin_edges``),
    optionally downsamples targets/censoring with ``stride``, builds train/val/test
    :class:`SurvivalDataset` objects, and creates corresponding :class:`DataLoader`
    instances. For the training loader, it optionally applies class balancing over
    *event* samples using :class:`WeightedRandomSampler`.

    Notes:
    - If there are no events in the training split (all censored), the train loader
      falls back to ``shuffle=True`` without a sampler.
    - Metadata ``num_time_bins`` and ``bin_edges`` are attached to ``train_loader``
      for downstream training code.

    :param x_train_bin: Binary/categorical features for train split.
    :type x_train_bin[: numpy.ndarray
    :param x_train_num: Numerical features for train split.
    :type x_train_num[: numpy.ndarray
    :param y_train: Continuous time target (e.g., seconds) for train split.
    :type y_train: numpy.ndarray
    :param c_train: Censoring/event indicator for train split.
    :type c_train: numpy.ndarray
    :param x_val_bin: Binary/categorical features for validation split.
    :type x_val_bin: Sequence[numpy.ndarray]
    :param x_val_num: Numerical features for validation split.
    :type x_val_num: Sequence[numpy.ndarray]
    :param y_val: Continuous time target for validation split.
    :type y_val: numpy.ndarray
    :param c_val: Censoring/event indicator for validation split.
    :type c_val: numpy.ndarray
    :param x_test_bin: Binary/categorical features for test split.
    :type x_test_bin: Sequence[numpy.ndarray]
    :param x_test_num: Numerical features for test split.
    :type x_test_num: Sequence[numpy.ndarray]
    :param y_test: Continuous time target for test split.
    :type y_test: numpy.ndarray
    :param c_test: Censoring/event indicator for test split.
    :type c_test: numpy.ndarray
    :param batch_size: Batch size for all loaders.
    :type batch_size: int
    :param stride: If not ``None``, downsample ``y`` and ``c`` by ``y[::stride]``.
    :type stride: int | None
    :param bin_edges: Monotonic bin edges for discretization. If ``None``,
        defaults to ``[0, 1h, 4h, 8h]`` in seconds.
    :type bin_edges: Sequence[float] | None
    :returns: Tuple ``(train_loader, val_loader, test_loader)``.
    :rtype: tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]
    """
    # 1. Define time bins (default)
    if bin_edges is None:
        m = 60.0
        h = 3600.0
        bin_edges = [0.0, 1 * h, 4 * h, 8 * h]

    # Total number of output classes (bins)
    num_time_bins = len(bin_edges) - 1

    # 2. Process train/val/test using the external helper
    Xb_tr, Xn_tr, y_tr, c_tr = concat_and_discretize(
        x_train_bin, x_train_num, y_train, c_train, stride, bin_edges
    )
    Xb_va, Xn_va, y_va, c_va = concat_and_discretize(
        x_val_bin, x_val_num, y_val, c_val, stride, bin_edges
    )
    Xb_te, Xn_te, y_te, c_te = concat_and_discretize(
        x_test_bin, x_test_num, y_test, c_test, stride, bin_edges
    )

    # 3. Build datasets
    train_ds = SurvivalDataset(Xb_tr, Xn_tr, y_tr, c_tr)
    val_ds = SurvivalDataset(Xb_va, Xn_va, y_va, c_va)
    test_ds = SurvivalDataset(Xb_te, Xn_te, y_te, c_te)

    # 4. Class balancing on TRAIN events using WeightedRandomSampler
    y_tr_t = train_ds.y
    c_tr_t = train_ds.c
    mask_not_censor = c_tr_t == 0

    event_idx = torch.where(mask_not_censor)[0]
    if int(event_idx.numel()) == 0:
        sampler = None
        shuffle_train = False
    else:
        y_event = y_tr_t[event_idx]

        # Count per class over events only
        class_counts = torch.bincount(y_event, minlength=num_time_bins).float()
        class_counts = torch.clamp(class_counts, min=1.0)

        # Inverse-frequency weights (normalized)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.mean()

        # Per-event weight by class
        event_weights = class_weights[y_event].double()

        sampler = WeightedRandomSampler(
            weights=event_weights.tolist(),
            num_samples=int(event_idx.numel()),
            replacement=True,
        )
        shuffle_train = False

    # 5. Build loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=sampler,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Attach metadata for the trainer
    train_loader.num_time_bins = num_time_bins
    train_loader.bin_edges = list(bin_edges)

    return train_loader, val_loader, test_loader