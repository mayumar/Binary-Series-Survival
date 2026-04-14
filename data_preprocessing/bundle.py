import os
import joblib
import json
from typing import Any, Optional, Dict, Sequence

import numpy as np
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler

def load_word2vec_model(model_path: str) -> Optional[Word2Vec]:
    """
    Load a Word2Vec model from disk.

    :param model_path: Path to the saved Word2Vec model file.
    :type model_path: str
    :return: Loaded Word2Vec model, or None if the file does not exist.
    :rtype: Optional[Word2Vec]
    """
    if not os.path.exists(model_path):
        return None
    
    return Word2Vec.load(model_path)

def save_word2vec_model(model: Word2Vec, model_path: str):
    """
    Save a Word2Vec model to disk if it doesn't already exist.

    :param model: Trained Word2Vec model to save.
    :type model: Word2Vec
    :param model_path: Path where the model should be saved.
    :type model_path: str
    :return: None
    :rtype: None
    """
    if os.path.exists(model_path):
        return
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)

def save_event2vec_bundle_splits(
    out_dir: str,
    *,
    dropped_cols: Sequence[str],
    scaler: StandardScaler,
    train_tokens: Optional[Sequence[Sequence[str]]] = None,
    val_tokens: Optional[Sequence[Sequence[str]]] = None,
    test_tokens: Optional[Sequence[Sequence[str]]] = None,
    train_num: Optional[Sequence[np.ndarray]] = None,
    val_num: Optional[Sequence[np.ndarray]] = None,
    test_num: Optional[Sequence[np.ndarray]] = None,
    train_bin: Optional[Sequence[np.ndarray]] = None,
    val_bin: Optional[Sequence[np.ndarray]] = None,
    test_bin: Optional[Sequence[np.ndarray]] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Save the preprocessing bundle (train/val/test splits) to disk.

    This bundle stores everything needed to reproduce preprocessing without recomputing:
    - Word2Vec model (Event2Vec)
    - dropped columns
    - fitted StandardScaler
    - token windows for train/val/test
    - optionally numeric windows and binary embeddings (recommended)

    :param out_dir: Output directory where artifacts will be stored.
    :type out_dir: str
    :param w2v_model: Trained Word2Vec model for binary tokens.
    :type w2v_model: Word2Vec
    :param dropped_cols: List of feature names dropped during training.
    :type dropped_cols: Sequence[str]
    :param scaler: StandardScaler fitted on training numeric features.
    :type scaler: StandardScaler
    :param train_tokens: Windowed token sequences for training.
    :type train_tokens: Sequence[Sequence[str]]
    :param val_tokens: Windowed token sequences for validation.
    :type val_tokens: Sequence[Sequence[str]]
    :param test_tokens: Windowed token sequences for test.
    :type test_tokens: Sequence[Sequence[str]]
    :param train_num: Optional windowed numeric sequences for training (already scaled).
    :type train_num: Optional[Sequence[np.ndarray]]
    :param val_num: Optional windowed numeric sequences for validation (already scaled).
    :type val_num: Optional[Sequence[np.ndarray]]
    :param test_num: Optional windowed numeric sequences for test (already scaled).
    :type test_num: Optional[Sequence[np.ndarray]]
    :param train_bin: Optional windowed binary sequences for training (already scaled).
    :type train_bin: Optional[Sequence[np.ndarray]]
    :param val_bin: Optional windowed binary sequences for validation (already scaled).
    :type val_bin: Optional[Sequence[np.ndarray]]
    :param test_bin: Optional windowed binary sequences for test (already scaled).
    :type test_bin: Optional[Sequence[np.ndarray]]
    :param metadata: Optional dictionary with extra info (e.g., window_size, stride).
    :type metadata: Optional[Dict[str, Any]]
    :return: None
    :rtype: None
    """
    os.makedirs(out_dir, exist_ok=True)

    # w2v_model.save(os.path.join(out_dir, "binary_event2vec.model"))
    joblib.dump(list(dropped_cols), os.path.join(out_dir, "dropped_cols.joblib"), compress=3)
    joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"), compress=3)

    if train_tokens is not None:
        joblib.dump(train_tokens, os.path.join(out_dir, "train_tokens.joblib"), compress=3)
    if val_tokens is not None:
        joblib.dump(val_tokens,   os.path.join(out_dir, "val_tokens.joblib"), compress=3)
    if test_tokens is not None:
        joblib.dump(test_tokens,  os.path.join(out_dir, "test_tokens.joblib"), compress=3)

    if train_num is not None:
        joblib.dump(train_num, os.path.join(out_dir, "train_num.joblib"), compress=3)
    if val_num is not None:
        joblib.dump(val_num, os.path.join(out_dir, "val_num.joblib"), compress=3)
    if test_num is not None:
        joblib.dump(test_num, os.path.join(out_dir, "test_num.joblib"), compress=3)

    if train_bin is not None:
        joblib.dump(train_bin, os.path.join(out_dir, "train_bin.joblib"), compress=3)
    if val_bin is not None:
        joblib.dump(val_bin, os.path.join(out_dir, "val_bin.joblib"), compress=3)
    if test_bin is not None:
        joblib.dump(test_bin, os.path.join(out_dir, "test_bin.joblib"), compress=3)

    if metadata is None:
        metadata = {}

    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def load_event2vec_bundle_splits(bundle_dir: str, w2v_path: str) -> Dict[str, Any]:
    """
    Load the preprocessing bundle (train/val/test splits) from disk.

    :param bundle_dir: Bundle directory where artifacts were stored.
    :param w2v_path: Directory where the Word2Vec model is stored.
    :return: Dictionary with keys:
        - ``w2v_model`` (Word2Vec)
        - ``dropped_cols`` (List[str])
        - ``scaler`` (StandardScaler)
        - ``train_tokens``, ``val_tokens``, ``test_tokens``
        - ``train_num``, ``val_num``, ``test_num`` (optional, may be None)
        - ``train_bin``, ``val_bin``, ``test_bin`` (optional, may be None)
        - ``metadata`` (dict)
    """
    out: Dict[str, Any] = {}

    out["dropped_cols"] = joblib.load(os.path.join(bundle_dir, "dropped_cols.joblib"))
    out["scaler"] = joblib.load(os.path.join(bundle_dir, "scaler.joblib"))

    # optional keys
    def maybe(path):
        return joblib.load(path) if os.path.exists(path) else None

    out["train_tokens"] = maybe(os.path.join(bundle_dir, "train_tokens.joblib"))
    out["val_tokens"]   = maybe(os.path.join(bundle_dir, "val_tokens.joblib"))
    out["test_tokens"]  = maybe(os.path.join(bundle_dir, "test_tokens.joblib"))

    out["train_num"] = maybe(os.path.join(bundle_dir, "train_num.joblib"))
    out["val_num"]   = maybe(os.path.join(bundle_dir, "val_num.joblib"))
    out["test_num"]  = maybe(os.path.join(bundle_dir, "test_num.joblib"))

    out["train_bin"] = maybe(os.path.join(bundle_dir, "train_bin.joblib"))
    out["val_bin"]   = maybe(os.path.join(bundle_dir, "val_bin.joblib"))
    out["test_bin"]  = maybe(os.path.join(bundle_dir, "test_bin.joblib"))

    meta_path = os.path.join(bundle_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            out["metadata"] = json.load(f)
    else:
        out["metadata"] = {}

    return out