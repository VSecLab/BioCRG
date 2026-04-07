# src/detect.py
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score


def window_errors(model, X, batch_size=128):
    """
    MSE per ogni finestra.
    X: (N, 50, 18) normalizzato
    Returns: (N,) scalare per finestra
    """
    X_hat  = model.predict(X, batch_size=batch_size, verbose=0)
    errors = np.mean((X - X_hat) ** 2, axis=(1, 2))
    return errors


def execution_score(win_errors, strategy="max"):
    """
    Aggrega gli errori delle finestre di un'esecuzione.
    strategy: "max" | "mean" | "p95"
    """
    if strategy == "max":  return float(win_errors.max())
    if strategy == "mean": return float(win_errors.mean())
    if strategy == "p95":  return float(np.percentile(win_errors, 95))
    raise ValueError(f"Strategia sconosciuta: {strategy}")


def compute_threshold(model, X_train_normal, percentile=95):
    """
    Soglia = percentile degli errori sui dati di training normali.
    """
    errors    = window_errors(model, X_train_normal)
    threshold = float(np.percentile(errors, percentile))
    print(f"Threshold ({percentile}p | {len(errors)} finestre): {threshold:.6f}")
    return threshold


def evaluate_executions(model, executions_test, scaler,
                        threshold, agg_strategy="max"):
    """
    Valuta a livello di esecuzione (non di finestra).
    Ogni oggetto in executions_test deve avere:
      .windows        -> np.ndarray (N_i, 50, 18) raw
      .label          -> int 0/1
      .execution_id   -> str
    """
    from preprocessing import apply_scaler

    y_true, y_pred, scores_all = [], [], []

    for exc in executions_test:
        X_s    = apply_scaler(exc.windows, scaler)
        errors = window_errors(model, X_s)
        score  = execution_score(errors, agg_strategy)
        pred   = int(score > threshold)

        y_true.append(exc.label)
        y_pred.append(pred)
        scores_all.append(score)

        tag = "ANOMALO" if pred else "normale"
        gt  = " <- GT anomalo" if exc.label else ""
        print(f"  {exc.execution_id:45s}  score={score:.5f}  {tag}{gt}")

    print()
    print(classification_report(
        y_true, y_pred,
        target_names=["normale", "anomalo"],
        zero_division=0,
    ))
    if len(set(y_true)) > 1:
        print(f"ROC-AUC: {roc_auc_score(y_true, scores_all):.3f}")