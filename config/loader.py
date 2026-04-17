import json
from typing import Dict, Any, Optional
from .schema import TrainingConfig, EvalConfig, RebalanceConfig, LSTMModelConfig

def load_json_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return normalize_config(raw)

def normalize_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    preprocess = raw.get("preprocess", {})
    training = raw.get("training", {})
    evaluation = raw.get("evaluation", {})
    legacy = raw.get("parameters", {})

    raw["parameters"] = {
        "start_idx": preprocess.get("start_idx", legacy.get("start_idx")),
        "max_horizon": preprocess.get("max_horizon", legacy.get("max_horizon")),
        "window_size": preprocess.get("window_size", legacy.get("window_size")),
        "prop_stride": preprocess.get("prop_stride", legacy.get("prop_stride")),
        "epochs": training.get("epochs", legacy.get("epochs", 100)),
        "learning_rate": training.get("learning_rate", legacy.get("learning_rate", 1e-3)),
        "weight_decay": training.get("weight_decay", 1e-4),
        "patience": training.get("patience", legacy.get("patience", 3)),
        "early_stopping_patience": training.get(
            "early_stopping_patience",
            legacy.get("early_stopping_patience", 20),
        ),
        "tau_severity": evaluation.get("tau_severity", legacy.get("tau_severity", 0.6)),
        "tau_alarm": evaluation.get("tau_alarm", legacy.get("tau_alarm", 0.6)),
    }

    raw.setdefault("preprocess", preprocess)
    raw.setdefault("training", training)
    raw.setdefault("evaluation", evaluation)
    raw.setdefault("model", {})
    raw.setdefault("rebalance", {})
    raw.setdefault("bin_edges", [0, 1, 4, 8])
    raw.setdefault("numeric_cols", [])

    return raw

def parse_training_config(raw: Dict[str, Any], device_override: Optional[str] = None) -> TrainingConfig:
    training = raw.get("training", {})
    params = raw["parameters"]
    return TrainingConfig(
        epochs=training.get("epochs", params["epochs"]),
        lr=training.get("learning_rate", params["learning_rate"]),
        weight_decay=training.get("weight_decay", params["weight_decay"]),
        early_stopping_patience=training.get(
            "early_stopping_patience",
            params["early_stopping_patience"]
        ),
        device=device_override or training.get("device", "cpu"),
    )

def parse_eval_config(raw: Dict[str, Any]) -> EvalConfig:
    evaluation = raw.get("evaluation", {})
    params = raw["parameters"]
    return EvalConfig(
        tau_severity=evaluation.get("tau_severity", params["tau_severity"]),
        tau_alarm=evaluation.get("tau_alarm", params["tau_alarm"]),
        bin_edges=raw.get("bin_edges", [0, 1, 4, 8])
    )

def parse_rebalance_config(raw: Dict[str, Any]) -> RebalanceConfig:
    rebalance = raw.get("rebalance", {})
    return RebalanceConfig(
        enabled=rebalance.get("enabled", True),
        max_censor_ratio=rebalance.get("max_censor_ratio", 0.5)
    )

def parse_lstm_model_config(raw: Dict[str, Any]) -> LSTMModelConfig:
    model = raw.get("model", {})
    return LSTMModelConfig(
        num_events=model.get("num_events", 1),
        hidden_size1=model.get("hidden_size1", 64),
        hidden_size2=model.get("hidden_size2", 128),
    )