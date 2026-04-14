import os
import json
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional

from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from gensim.models import Word2Vec

from .event2vec import tokens_list_to_embeddings
from .compute_dataloaders import compute_DataLoaders
from .bundle import (
    load_event2vec_bundle_splits,
    save_event2vec_bundle_splits,
    load_word2vec_model,
    save_word2vec_model
)


def drop_low_variance_features(df: pd.DataFrame, var_threshold: float = 0.0) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop numeric features whose variance is less than or equal to a threshold.

    Only columns with numeric dtype are considered when computing variance.

    :param df: Input dataframe.
    :type df: pandas.DataFrame
    :param var_threshold: Variance threshold. Columns with variance <= this value are dropped.
    :type var_threshold: float
    :return: A tuple ``(df_clean, dropped_columns)``.
    :rtype: Tuple[pandas.DataFrame, List[str]]
    """
    numeric_df = df.select_dtypes(include=[np.number])
    variances = numeric_df.var()
    low_var_cols = variances[variances <= var_threshold].index.tolist()
    df_clean = df.drop(columns=low_var_cols)
    return df_clean, low_var_cols


def drop_highly_correlated_features(df: pd.DataFrame, input_corr_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop features listed in a correlation-selection JSON file.

    The JSON file is expected to contain a top-level key ``"variables"`` with a list
    of column names to remove.

    :param df: Input dataframe.
    :type df: pandas.DataFrame
    :param input_corr_path: Path to a JSON file produced by the correlation script.
    :type input_corr_path: str
    :return: A tuple ``(df_clean, dropped_columns)``.
    :rtype: Tuple[pandas.DataFrame, List[str]]
    :raises KeyError: If the JSON does not contain the key ``"variables"``.
    :raises FileNotFoundError: If ``input_corr_path`` does not exist.
    :raises json.JSONDecodeError: If the file is not valid JSON.
    """
    with open(input_corr_path, "r") as f:
        to_drop = json.load(f)["variables"]

    to_drop = [c for c in to_drop if c in df.columns]

    df_clean = df.drop(columns=to_drop)

    return df_clean, to_drop


def split_binary_numeric_features(df: pd.DataFrame, numeric_cols: List[str] = []) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Split a dataframe into binary-like and numeric columns.

    A column is considered "binary-like" if the number of unique non-null values is <= 2.

    :param df: Input dataframe.
    :type df: pandas.DataFrame
    :return: A tuple ``(binary_df, numeric_df)``.
    :rtype: Tuple[pandas.DataFrame, pandas.DataFrame]
    """
    if not numeric_cols:
        binary_cols = [
            c for c in df.columns
            if df[c].dropna().nunique() <= 2
        ]

        numeric_cols = [c for c in df.columns if c not in binary_cols]
    else:
        binary_cols = [c for c in df.columns if c not in numeric_cols]

    binary_df = pd.DataFrame(df[binary_cols])
    numeric_df = pd.DataFrame(df[numeric_cols])

    return binary_df, numeric_df, numeric_cols


def binary_row_to_token(row):
    return "".join(row.astype(int).astype(str))


def binary_df_to_token_sequence(df_bin: pd.DataFrame, n_jobs: int = -1) -> List:
    """
    Convert a binary dataframe into a token sequence (one token per row).

    Each row is converted to a string by concatenating the integer-casted values.

    :param df_bin: Binary dataframe.
    :type df_bin: pandas.DataFrame
    :param n_jobs: Number of jobs for parallel execution. Use -1 to use all cores.
    :type n_jobs: int
    :return: Token list with one string token per row.
    :rtype: List[str]
    """
    tokens = Parallel(n_jobs=n_jobs)(
        delayed(binary_row_to_token)(row)
        for _, row in df_bin.iterrows()
    )

    return list(tokens)


def preprocess_df(
    df: pd.DataFrame,
    is_train: bool = False,
    dropped_cols: Optional[List[str]] = None,
    binary_model: Optional[Word2Vec] = None,
    scaler: Optional[StandardScaler] = None,
    numeric_cols: List[str] = [],
    var_threshold: float = 0.0,
    corr_input: str = "variables_a_eliminar.json",
    window_size: int = 900,
    stride: int = 900,
    use_word2vec: bool = True
) -> Tuple[List[np.ndarray], List[np.ndarray], Optional[Word2Vec], List[str], StandardScaler, Optional[List[List[str]]], List[str]]:
    """
    Preprocess a dataframe split for model input generation.

    The preprocessing pipeline performs the following steps:
      1) Drop features:
         - Training: compute low-variance and high-correlation drops.
         - Validation/Test: reuse the list of dropped columns computed on training.
      2) Split features into binary and numeric subsets.
      3) Scale numeric features:
         - Training: fit a new scaler.
         - Validation/Test: reuse the fitted scaler from training.
      4) Convert binary rows into token sequences.
      5) Apply sliding windows to both token and numeric sequences.
      6) Transform token windows into embeddings using a Word2Vec model.

    If ``use_word2vec=True``, binary rows are tokenized and embedded with Word2Vec.
    If ``use_word2vec=False``, raw binary windows are returned directly without
    tokenization or embedding.

    :param df: Input feature dataframe.
    :type df: pandas.DataFrame
    :param is_train: Whether the dataframe belongs to the training split.
        If ``True``, dropped columns and the scaler are computed/fitted.
    :type is_train: bool
    :param dropped_cols: Columns to drop in validation/test splits.
        These should be the columns identified during training.
    :type dropped_cols: Optional[List[str]]
    :param binary_model: Word2Vec model used to transform token windows into
        embeddings. If ``is_train=True`` and this parameter is ``None``, a new
        model is trained.
    :type binary_model: Optional[gensim.models.Word2Vec]
    :param scaler: Fitted scaler used to normalize numeric features.
        If ``is_train=True`` and this parameter is ``None``, a new
        :class:`sklearn.preprocessing.StandardScaler` is created and fitted.
    :type scaler: Optional[sklearn.preprocessing.StandardScaler]
    :param numeric_cols: List of numeric column names to reuse in validation/test.
        When training, numeric columns are inferred from the dataframe.
    :type numeric_cols: List[str]
    :param var_threshold: Variance threshold used to remove low-variance features
        during training.
    :type var_threshold: float
    :param corr_input: Path to the JSON file containing correlated variables to
        remove during training.
    :type corr_input: str
    :param window_size: Number of time steps per generated window.
    :type window_size: int
    :param stride: Step size between consecutive windows.
    :type stride: int
    :param use_word2vec: Whether to encode binary features with Word2Vec.
    :type use_word2vec: bool
    :return: A tuple containing:

        - ``binary_embeddings``: List of binary embedding windows.
        - ``numeric_data``: List of numeric feature windows.
        - ``binary_model``: Trained or reused Word2Vec model.
        - ``dropped_cols``: List of dropped columns.
        - ``scaler``: Fitted scaler for numeric features.
        - ``binary_tokens``: List of token windows.
        - ``numeric_cols``: List of numeric column names.
    :rtype: Tuple[List[numpy.ndarray], List[numpy.ndarray], gensim.models.Word2Vec,
        List[str], sklearn.preprocessing.StandardScaler, List[List[str]], List[str]]
    :raises ValueError: If ``is_train=False`` and ``scaler`` is ``None``.
    """
    
    # 1. Drop columns
    if is_train:
        df, dropped_cols_low_variance = drop_low_variance_features(df, var_threshold)
        df, dropped_high_multicollinearity = drop_highly_correlated_features(df, corr_input)
        dropped_cols = dropped_cols_low_variance + dropped_high_multicollinearity
    else:
        if dropped_cols is not None:
            valid_cols = [c for c in dropped_cols if c in df.columns]
            df.drop(columns=valid_cols, inplace=True)

    assert dropped_cols is not None

    # 2. Split binary / numeric
    print('Separación Binario / Numérico')
    if is_train:
        binary_df, numeric_df, numeric_cols = split_binary_numeric_features(df)
    else:
        binary_df, numeric_df, _ = split_binary_numeric_features(df, numeric_cols)

    # 3. Scale numeric
    print("Inicio escalado")
    if not numeric_df.empty:
        if is_train:
            scaler = StandardScaler()
            numeric_values = scaler.fit_transform(numeric_df)
        else:
            if scaler is None:
                raise ValueError("scaler is None in validation/test. Pass the fitted scaler from train.")
            numeric_values = scaler.transform(numeric_df)
        
        numeric_df = pd.DataFrame(numeric_values, index=numeric_df.index, columns=numeric_df.columns)

    assert scaler is not None

    # 4. Windowing
    binary_data = []
    numeric_data = []
    binary_tokens = [] if use_word2vec else None

    binary_values = binary_df.to_numpy(dtype=np.float32) if not binary_df.empty else np.empty((len(df), 0), dtype=np.float32)
    numeric_values = numeric_df.to_numpy(dtype=np.float32) if not numeric_df.empty else np.empty((len(df), 0), dtype=np.float32)

    tokens = None
    if use_word2vec:
        tokens = binary_df_to_token_sequence(binary_df)

    for i in range(0, len(df) - window_size + 1, stride):
        binary_window = binary_values[i:i + window_size]
        numeric_window = numeric_values[i:i + window_size]

        binary_data.append(binary_window)
        numeric_data.append(numeric_window)

        if use_word2vec and binary_tokens is not None and tokens is not None:
            binary_tokens.append(tokens[i:i + window_size])

    # 5. Binary representation
    if use_word2vec:
        if binary_tokens is None:
            raise ValueError("binary_tokens is None although use_word2vec=True.")

        binary_embeddings, binary_model = tokens_list_to_embeddings(
            binary_tokens,
            window_size,
            binary_model
        )
        return binary_embeddings, numeric_data, binary_model, dropped_cols, scaler, binary_tokens, numeric_cols

    # No Word2Vec
    return binary_data, numeric_data, None, dropped_cols, scaler, None, numeric_cols


def format_and_concat(
    data: Tuple[List, List, List]
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Concatenate a split composed of multiple chunks.

    The input tuple contains:
      - a list of feature dataframes,
      - a list of y arrays,
      - a list of censor/event-indicator arrays.

    :param data: Tuple ``(list_of_feature_dfs, list_of_y_arrays, list_of_censor_arrays)``.
    :type data: Tuple[List, List, List]
    :return: Tuple ``(x, y, c)`` where:
        - ``x`` is the concatenated feature dataframe,
        - ``y`` is a dataframe containing the concatenated target values,
        - ``c`` is a dataframe containing the concatenated censor indicators.
    :rtype: Tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
    """
    x = pd.DataFrame(pd.concat(data[0], axis=0, ignore_index=True))
    y = np.concatenate(data[1], axis=0).astype(np.float32)
    e = np.concatenate(data[2], axis=0).astype(np.float32)
    return x, y, e


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

def preprocess_data(
    train: Tuple[List, List, List],
    val: Tuple[List, List, List],
    test: Tuple[List, List, List],
    fail_to_pred: str,
    config: dict,
    use_word2vec: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prepare train, validation, and test :class:`torch.utils.data.DataLoader` objects
    for the survival LSTM pipeline.

    The function performs the following steps:

    1. Concatenates the input split data into unified feature and target arrays.
    2. Preprocesses the feature matrices by:
    - removing correlated columns,
    - scaling numerical variables,
    - generating token sequences,
    - computing Event2Vec embeddings using a Word2Vec model.
    3. Uses a disk cache (Event2Vec bundle) to avoid recomputing expensive
    preprocessing steps when available.
    4. Discretizes the target variable using the bin edges defined in the
    configuration.
    5. Builds PyTorch :class:`torch.utils.data.DataLoader` objects for training,
    validation, and testing.

    If a cached Event2Vec bundle and Word2Vec model exist, they are loaded from disk.
    Otherwise, the preprocessing pipeline is executed and the resulting artifacts
    are saved for future reuse.

    The configuration dictionary must include the following keys:

    - ``config["paths"]["input_correladas"]``: Path to the correlated feature file.
    - ``config["paths"]["event2vec_bundle_dir"]``: Directory where the preprocessing
    cache (Event2Vec bundle) is stored.
    - ``config["paths"]["event2vec_model_path"]``: Path to the Word2Vec model file.
    - ``config["parameters"]["window_size"]``: Sliding window size used for
    sequence generation.
    - ``config["parameters"]["prop_stride"]``: Stride proportion used to derive
    the window step.
    - ``config["bin_edges"]``: Time bin boundaries (in hours) used to discretize
    survival times.

    :param train: Training split containing feature DataFrames, event times,
        and censor indicators.
    :type train: Tuple[List[pandas.DataFrame], List, List]
    :param val: Validation split containing feature DataFrames, event times,
        and censor indicators.
    :type val: Tuple[List[pandas.DataFrame], List, List]
    :param test: Test split containing feature DataFrames, event times,
        and censor indicators.
    :type test: Tuple[List[pandas.DataFrame], List, List]
    :param config: Configuration dictionary controlling preprocessing,
        caching, and discretization parameters.
    :type config: dict
    :return: DataLoaders for training, validation, and testing.
    :rtype: Tuple[
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader
    ]
    :raises ValueError: If cached artifacts are missing, inconsistent,
        or incompatible with the expected preprocessing configuration.
    """
    print("Concatenating split data...")
    (x_train, y_train, c_train) = format_and_concat(train)
    (x_val, y_val, c_val) = format_and_concat(val)
    (x_test, y_test, c_test) = format_and_concat(test)

    print("Processing features and scaling...")
    
    corr_input_path = config['paths']['input_correladas']
    window_size = config['parameters']['window_size']
    stride = window_size // config['parameters']['prop_stride']

    event2vec_dir = config['paths']['event2vec_dir']

    bundle_path = os.path.join(event2vec_dir, fail_to_pred, "event2vec_bundle")
    print(bundle_path)
    event2vec_model_path = os.path.join(event2vec_dir, fail_to_pred, "binary_event2vec.model")

    binary_model = None
    if use_word2vec:
        binary_model = load_word2vec_model(event2vec_model_path)
        if binary_model is not None:
            print(f"[CACHE] Loading Word2Vec model from: {event2vec_model_path}")
        else:
            print(f"[CACHE] No Word2Vec model found at: {event2vec_model_path}. A new model will be trained.")

    if os.path.exists(bundle_path) and (binary_model is not None or not use_word2vec):
        print(f"[CACHE] Loading bundle from: {bundle_path}")
        bundle = load_event2vec_bundle_splits(bundle_path, event2vec_model_path)

        dropped_cols = bundle["dropped_cols"]
        scaler = bundle["scaler"]

        # embeddings (if exist)
        train_tokens = bundle["train_tokens"]
        val_tokens   = bundle["val_tokens"]
        test_tokens  = bundle["test_tokens"]

        x_train_num = bundle["train_num"]
        x_val_num   = bundle["val_num"]
        x_test_num  = bundle["test_num"]

        x_train_bin = bundle["train_bin"]
        x_val_bin   = bundle["val_bin"]
        x_test_bin  = bundle["test_bin"]

        # Recalculate embeddings if not exist
        if use_word2vec:
            if x_train_bin is None:
                x_train_bin, _ = tokens_list_to_embeddings(train_tokens, window_size, binary_model)
            if x_val_bin is None:
                x_val_bin, _ = tokens_list_to_embeddings(val_tokens, window_size, binary_model)
            if x_test_bin is None:
                x_test_bin, _ = tokens_list_to_embeddings(test_tokens, window_size, binary_model)

    else:
        print(f"[CACHE] Bundle not found. Building and saving to: {bundle_path}")
        x_train_bin, x_train_num, binary_model, dropped_cols, scaler, train_tokens, numeric_cols = preprocess_df(
            x_train,
            is_train=True,
            dropped_cols=[],
            binary_model=binary_model,
            scaler=None,
            corr_input=corr_input_path,
            window_size=window_size,
            stride=stride,
            use_word2vec=use_word2vec
        )

        x_val_bin, x_val_num, _, _, _, val_tokens, _ = preprocess_df(
            x_val, is_train=False,
            dropped_cols=dropped_cols,
            binary_model=binary_model,
            scaler=scaler,
            numeric_cols=numeric_cols,
            corr_input=corr_input_path,
            window_size=window_size,
            stride=stride,
            use_word2vec=use_word2vec
        )

        x_test_bin, x_test_num, _, _, _, test_tokens, _ = preprocess_df(
            x_test, is_train=False,
            dropped_cols=dropped_cols,
            binary_model=binary_model,
            scaler=scaler,
            numeric_cols=numeric_cols,
            corr_input=corr_input_path,
            window_size=window_size,
            stride=stride,
            use_word2vec=use_word2vec
        )

        if use_word2vec and binary_model is not None:
            save_word2vec_model(binary_model, event2vec_model_path)

        save_event2vec_bundle_splits(
            bundle_path,
            dropped_cols=dropped_cols,
            scaler=scaler,
            train_tokens=train_tokens if use_word2vec else None,
            val_tokens=val_tokens if use_word2vec else None,
            test_tokens=test_tokens if use_word2vec else None,
            train_num=x_train_num,
            val_num=x_val_num,
            test_num=x_test_num,
            train_bin=x_train_bin,
            val_bin=x_val_bin,
            test_bin=x_test_bin,
            metadata={
                "window_size": window_size,
                "stride": stride,
                "use_word2vec": use_word2vec
            }
        )

    h = 3600
    custom_bin_edges = [edge * h for edge in config['bin_edges']]
    print(f"Using bin edges (seconds): {custom_bin_edges}")

    train_loader, valid_loader, test_loader = compute_DataLoaders(
        x_train_bin, x_train_num, y_train, c_train,
        x_val_bin,   x_val_num,   y_val,   c_val,
        x_test_bin,  x_test_num,  y_test,  c_test,
        batch_size=64,  
        stride=stride,       
        bin_edges=custom_bin_edges
    )

    return train_loader, valid_loader, test_loader