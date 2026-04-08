from typing import Tuple, List, Optional

import numpy as np
import pandas as pd

LoadedData = Tuple[
    Tuple[List, List, List],
    Tuple[List, List, List],
    Tuple[List, List, List]
]

def load_data(
    input: str,
    fail_to_pred: str,
    val_prop: float=0.3,
    test_prop: float = 0.15,
    horizon_h: int = 20,
    start_idx: int = 0
) -> LoadedData:
    """
    Load dataset and construct run-to-failure survival splits.

    The function:
      1. Reads the target failure column and time column.
      2. Detects failure onset events (0 → 1 transitions).
      3. Identifies run-to-failure cycles.
      4. Converts each cycle into Time-to-Event format.
      5. Splits the cycles temporally into train/validation/test sets.

    :param input: Path to the input CSV file.
    :type input: str
    :param fail_to_pred: Name of the failure column to predict.
    :type fail_to_pred: str
    :param val_prop: Proportion of cycles used for validation.
    :type val_prop: float
    :param test_prop: Proportion of cycles used for testing.
    :type test_prop: float
    :param horizon_h: Prediction horizon in hours.
    :type horizon_h: int
    :param start_idx: Initial index if the dataset.
    :type start_idx: int
    :return: Three tuples corresponding to train, validation and test splits.
             Each tuple contains:
                 - List of feature DataFrames
                 - List of TTE target Series
                 - List of censoring Series
    :rtype: LoadedData = Tuple[
        Tuple[List, List, List],
        Tuple[List, List, List],
        Tuple[List, List, List]
    ]
    """
    df = pd.read_csv(input, usecols=["time", fail_to_pred])
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.iloc[start_idx:].reset_index(drop=True)

    # Detect rising edge of the failure signal (0 -> 1 transition)
    # This defines the exact time instant when the failure begins
    fail_event = (df[fail_to_pred] == 1) & (df[fail_to_pred].shift(1, fill_value=0) == 0)

    # Identify "Run-to-Failure" cycles
    fallo_col = f"fallo_{fail_to_pred}"
    df[fallo_col] = np.nan
    df.loc[fail_event, fallo_col] = range(fail_event.sum())
    df[fallo_col] = df[fallo_col].bfill()

    y_data = []
    censorship_data = []
    run_to_failure_index = []
    
    # Define prediction horizon in seconds
    HORIZON_SECONDS = horizon_h * 3600 

    for u in df[fallo_col].dropna().unique():
        mask = df[fallo_col] == u 

        # Filter very short cycles
        if mask.sum() > 2*3600:
            df_cycle_index = df.loc[mask].index
            
            df_cycle = df.iloc[df_cycle_index]

            run_to_failure_index.append(df_cycle_index)

            t_obs, censorship = binary_to_tte_censored(
                df_cycle[fail_to_pred], 
                H=HORIZON_SECONDS
            )
            
            y_data.append(t_obs)
            censorship_data.append(censorship)

    # Add index offset
    run_to_failure_index_original = [
        batch + start_idx for batch in run_to_failure_index
    ]

    # Read feature matrix X
    x_data = read_csv_by_index_batches_sorted(
        input, 
        run_to_failure_index_original
    )

    total_cycles = len(x_data)

    n_val = 1 if int(total_cycles * val_prop) < 1 else int(total_cycles * val_prop)
    n_test = 1 if int(total_cycles * test_prop) < 1 else int(total_cycles * test_prop)
    n_train = total_cycles - n_val - n_test

    x_train, y_train, c_train = x_data[:n_train], y_data[:n_train], censorship_data[:n_train]
    x_val, y_val, c_val = x_data[n_train:n_train+n_val], y_data[n_train:n_train+n_val], censorship_data[n_train:n_train+n_val]
    x_test, y_test, c_test = x_data[n_train+n_val:], y_data[n_train+n_val:], censorship_data[n_train+n_val:]

    return (x_train, y_train, c_train), (x_val, y_val, c_val), (x_test, y_test, c_test)


def binary_to_tte_censored(binary_series: pd.Series, H: int) -> Tuple[pd.Series, pd.Series]:
    """
    Convert a binary failure series into Time-to-Event (TTE) format
    with right censoring at a fixed time horizon.

    The function computes the time remaining until the next failure
    event for each time step. If no future event exists or the time
    to event exceeds the horizon ``H``, the observation is right-censored.

    :param binary_series: Binary time series indicating failure state
                          (1 = failure event, 0 = no failure).
    :type binary_series: pandas.Series
    :param H: Right-censoring horizon (in the same time units as the series).
    :type H: int
    :return: A tuple containing:
             - Observed time-to-event values (capped at H).
             - Censoring indicator (1 = censored, 0 = event observed).
    :rtype: Tuple[pandas.Series, pandas.Series]
    """
    f = binary_series.values.astype(float)
    n = len(f)

    # Index where the event happends (value 1)
    idx = np.arange(n)
    event_idx = np.where(f == 1, idx, np.nan)

    # Backward fill to determine where the next event occurs
    next_event = pd.Series(event_idx).bfill().values.astype(float)
    
    # Time To Event (TTE)
    tte = next_event - idx

    # --- CENSORING LOGIC ---
    # 1. Initialize t_obs with the computed time-to-event
    # 2. If tte is NaN (no more failures in the dataset), assign H
    # 3. If tte > H (failure too far in the future), truncate to H
    
    # Handle NaNs (end of dataset without failure)
    tte_filled = np.nan_to_num(tte, nan=H + 1.0) # Temporarily assign H+1
    
    # Apply horizon truncation (Right Censoring at H)
    t_obs = np.minimum(tte_filled, H)
    
    censored = np.zeros(n, dtype=int)
    
    # Case A: The actual failure occurs beyond H -> Censored
    censored[tte_filled > H] = 1
    
    # Case B: No future failure existed (originally NaN) -> Censored
    censored[np.isnan(tte)] = 1

    return (
        pd.Series(t_obs, index=binary_series.index),
        pd.Series(censored, index=binary_series.index),
    )


def read_csv_by_index_batches_sorted(
    filepath: str,
    index_batches: List[List[int]],
    usecols: Optional[List] = None,
    chunksize: int = 100_000,
) -> List[pd.DataFrame]:
    """
    Efficiently read subsets of a CSV file based on sorted index batches.

    The function processes the CSV file in chunks and extracts
    only the rows corresponding to the provided index batches.
    This avoids loading the full dataset into memory.

    :param filepath: Path to the CSV file.
    :type filepath: str
    :param index_batches: List of index arrays defining each cycle.
    :type index_batches: List[List[int]]
    :param usecols: Columns to read from the CSV file.
    :type usecols: list or None
    :param chunksize: Number of rows per chunk when streaming the CSV.
    :type chunksize: int
    :param y_col: Name of the target column (optional).
    :type y_col: str
    :return: List of DataFrames containing feature data per cycle.
    :rtype: List[pandas.DataFrame]
    """
    index_batches = [sorted(batch) for batch in index_batches]
    pointers = [0] * len(index_batches)
    results = [[] for _ in index_batches]
    
    reader = pd.read_csv(filepath, usecols=usecols, chunksize=chunksize)
    current_idx = 0

    for chunk in reader:
        for i, batch in enumerate(index_batches):
            ptr = pointers[i]
            selected = []
            while ptr < len(batch):
                idx = batch[ptr]
                if idx < current_idx:
                    ptr += 1
                elif idx >= current_idx + len(chunk):
                    break
                else:
                    selected.append(idx - current_idx)
                    ptr += 1
            pointers[i] = ptr
            if selected:
                results[i].append(chunk.iloc[selected])
        current_idx += len(chunk)

    dfs = [pd.concat(dfs, ignore_index=True) for dfs in results]
    # Select all features, including fail_to_pred
    x_cols = [c for c in dfs[0].columns if c not in ["time", "id"]]
    x_data = [df[x_cols] for df in dfs]
    return x_data
    