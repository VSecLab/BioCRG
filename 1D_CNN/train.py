import os
import numpy as np
import tensorflow as tf
from tensorflow import keras # pyright: ignore[reportMissingModuleSource]
from model import build_autoencoder, compile_model


def train_autoencoder(
    X_train,          # (N_train, 50, 18) — utenti train, solo normali
    X_val,            # (N_val,   50, 18) — utenti train, solo normali
    activity,
    epochs=60,
    batch_size=64,
    lr=1e-3,
    patience=10,
    save_dir="outputs/models",
    timesteps=None,
    n_features=None,
):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{activity}_autoencoder.keras")

    print(f"[{activity}] train={len(X_train)} | val={len(X_val)} finestre")
    print(f"GPU: {tf.config.list_physical_devices('GPU')}")

    if timesteps is None or n_features is None:
        _, inferred_timesteps, inferred_features = X_train.shape
        timesteps = timesteps or inferred_timesteps
        n_features = n_features or inferred_features

    model = build_autoencoder(timesteps=timesteps, n_features=n_features)
    compile_model(model, lr=lr)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=save_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            verbose=1,
        ),
    ]

    # input = target: autoencoder
    history = model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )

    best = min(history.history["val_loss"])
    print(f"Best val_loss: {best:.5f} -> {save_path}")
    return model


def load_model(activity, save_dir="outputs/models"):
    path = os.path.join(save_dir, f"{activity}_autoencoder.keras")
    return keras.models.load_model(path)