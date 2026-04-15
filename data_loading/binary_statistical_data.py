import pandas as pd
import numpy as np

from typing import Dict

def binary_entropy(p: float) -> float:
    """
    Compute binary entropy (in bits) for probability p.

    :param p: Probability of 1 in binary signal
    :type p: float
    :return: Binary entropy in bits
    :rtype: float
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p*np.log2(p) + (1-p)*np.log2(1-p)))

def summarize_binary_column(values: pd.Series) -> Dict[str, float]:
    """
    Compute summary statistics for one binary-like column.

    Metrics:
    - active_samples: number of samples equal to 1
    - num_events: number of activation runs
    - activation_ratio: proportion of active samples
    - entropy: binary entropy
    - mean_run_length: mean duration of activation runs in samples
    - max_run_length: max duration of activation runs in samples
    - mean_gap_between_events: mean distance between consecutive activations in samples

    :param values: Binary signal column.
    :type values: pd.Series
    :return: Dictionary with summary metrics.
    :rtype: Dict[str, float]
    """
    values = values.dropna()

    if values.empty:
        return {
            "active_samples": 0,
            "num_events": 0,
            "activation_ratio": np.nan,
            "entropy": np.nan,
            "mean_run_length": np.nan,
            "max_run_length": np.nan,
            "mean_gap_between_events": np.nan
        }

    # Force true binary signal
    x = (values.fillna(0).to_numpy() > 0).astype(np.int8)

    n = len(x)
    active_samples = int(x.sum())
    activation_ratio = float(active_samples / n)
    entropy = binary_entropy(float(x.mean()))

    # Robust run detection
    padded = np.pad(x, (1, 1), constant_values=0)
    changes = np.diff(padded)

    run_starts = np.where(changes == 1)[0]
    run_ends = np.where(changes == -1)[0]
    run_lengths = run_ends - run_starts

    mean_run_length = float(run_lengths.mean()) if len(run_lengths) > 0 else 0.0
    max_run_length = float(run_lengths.max()) if len(run_lengths) > 0 else 0.0
    mean_gap_between_events = (
        float(np.diff(run_starts).mean()) if len(run_starts) >= 2 else np.nan
    )

    return {
        "active_samples": active_samples,
        "num_events": int(len(run_starts)),
        "activation_ratio": activation_ratio,
        "entropy": entropy,
        "mean_run_length": mean_run_length,
        "max_run_length": max_run_length,
        "mean_gap_between_events": mean_gap_between_events,
    }



def summarize_binary_dataframe(
    df: pd.DataFrame,
    partition_name: str = "data"
) -> pd.DataFrame:
    """
    Compute summary statistics for all binary columns in a dataframe.
    """
    cols = [
        "partition",
        "variable",
        "active_samples",
        "num_events",
        "activation_ratio",
        "entropy",
        "mean_run_length",
        "max_run_length",
        "mean_gap_between_events"
    ]

    if df.empty or len(df.columns) == 0:
        return pd.DataFrame(columns=cols)

    work_df = df.copy()
    work_df = work_df.rolling(window=10, min_periods=1).max().reset_index(drop=True)

    results = []
    for col in work_df.columns:
        stats = summarize_binary_column(work_df[col])
        stats["variable"] = col
        stats["partition"] = partition_name
        results.append(stats)

    return pd.DataFrame(results, columns=cols)