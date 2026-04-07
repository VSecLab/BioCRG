import os
import numpy as np

from preprocessing import (
	config,
	load_all_sessions,
	load_session_executions,
	fit_and_save_scaler,
	apply_scaler,
)
from dataclasses import dataclass
from train import train_autoencoder
from detect import compute_threshold, evaluate_executions


ACTIVITY = "sphereActivity"
VAL_RATIO = 0.2
SEED = 42
TEST_LABEL = 0


@dataclass
class Execution:
	windows: np.ndarray
	label: int
	execution_id: str


def split_train_val(X, val_ratio=0.2, seed=42):
	if len(X) < 2:
		raise ValueError("Servono almeno 2 finestre per fare train/val split")

	rng = np.random.default_rng(seed)
	idx = rng.permutation(len(X))
	n_val = max(1, int(len(X) * val_ratio))

	val_idx = idx[:n_val]
	train_idx = idx[n_val:]
	if len(train_idx) == 0:
		train_idx = val_idx[:1]
		val_idx = val_idx[1:]

	return X[train_idx], X[val_idx]


def build_test_executions(test_dir, activity, label=0):
	executions = []

	for subfolder in sorted(os.listdir(test_dir)):
		subfolder_path = os.path.join(test_dir, subfolder)
		if not os.path.isdir(subfolder_path):
			continue

		session_executions = load_session_executions(subfolder_path, activity=activity)
		for execution_id, wins in session_executions:
			executions.append(
				Execution(
					windows=wins,
					label=label,
					execution_id=execution_id,
				)
			)

	if not executions:
		raise ValueError("Nessuna esecuzione test trovata")

	return executions


def main():
	print(f"Attivita': {ACTIVITY}")

	# 1) Preprocess: CSV -> finestre train (raw_data)
	X_all = load_all_sessions(config["raw_data"], activity=ACTIVITY)
	X_train_raw, X_val_raw = split_train_val(X_all, val_ratio=VAL_RATIO, seed=SEED)

	# 2) Preprocess: esecuzioni test (una esecuzione per sottocartella)
	test_executions = build_test_executions(
		test_dir=config["test_data"],
		activity=ACTIVITY,
		label=TEST_LABEL,
	)
	X_test_raw = np.concatenate([exc.windows for exc in test_executions], axis=0)

	print(
		f"Split -> train={len(X_train_raw)} | val={len(X_val_raw)} | test_windows={len(X_test_raw)} | test_executions={len(test_executions)}"
	)

	# 3) Scaling (fit solo su train)
	scaler = fit_and_save_scaler(X_train_raw, f"outputs/scalers/{ACTIVITY}.pkl")
	X_train = apply_scaler(X_train_raw, scaler)
	X_val = apply_scaler(X_val_raw, scaler)

	# 4) Train modello
	model = train_autoencoder(
		X_train,
		X_val,
		activity=ACTIVITY,
		timesteps=X_train.shape[1],
		n_features=X_train.shape[2],
	)

	# 5) Threshold sui normali di train
	threshold = compute_threshold(model, X_train, percentile=95)

	# 6) Detection a livello di esecuzione
	print("\n=== RISULTATI DETECTION (execution-level) ===")
	evaluate_executions(
		model,
		test_executions,
		scaler,
		threshold,
		agg_strategy="max",
	)


if __name__ == "__main__":
	main()