import numpy as np
from tensorflow import keras # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras import layers # pyright: ignore[reportMissingModuleSource]


def build_autoencoder(timesteps=50, n_features=18):
    """
    CNN-1D Autoencoder per anomaly detection per-attivita'.
    Input/Output: (batch, timesteps, n_features) = (B, 50, 18)
    """
    inputs = keras.Input(shape=(timesteps, n_features))

    # ── ENCODER ──────────────────────────────────────────────
    x = layers.Conv1D(32, kernel_size=3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)       # 50 -> 25

    x = layers.Conv1D(64, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)       # 25 -> 12

    # bottleneck
    x = layers.Conv1D(128, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # shape: (B, 12, 128)

    # ── DECODER ──────────────────────────────────────────────
    x = layers.UpSampling1D(size=2)(x)            # 12 -> 24
    x = layers.Conv1D(64, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.UpSampling1D(size=2)(x)            # 24 -> 48
    x = layers.Conv1D(32, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Ripristina esattamente la lunghezza temporale di input.
    # Dopo 2xMaxPool + 2xUpSampling la lunghezza diventa floor(timesteps/4)*4.
    # Esempio: 50 -> 48 (serve +2), 100 -> 100 (serve +0).
    decoded_timesteps = (timesteps // 4) * 4
    pad_needed = timesteps - decoded_timesteps
    if pad_needed > 0:
        left_pad = pad_needed // 2
        right_pad = pad_needed - left_pad
        x = layers.ZeroPadding1D(padding=(left_pad, right_pad))(x)

    # Nessuna attivazione finale: regressione su valori normalizzati
    outputs = layers.Conv1D(n_features, kernel_size=3, padding="same")(x)
    # shape: (B, 50, 18)

    return keras.Model(inputs=inputs, outputs=outputs,
                       name="cnn1d_autoencoder")


def compile_model(model, lr=1e-3):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"],
    )
    return model


# ── Sanity check ──────────────────────────────────────────────
if __name__ == "__main__":
    model = build_autoencoder(timesteps=50, n_features=18)
    compile_model(model)
    model.summary()

    x   = np.random.randn(32, 50, 18).astype("float32")
    out = model.predict(x, verbose=0)
    print("Input :", x.shape)    # (32, 50, 18)
    print("Output:", out.shape)  # (32, 50, 18) OK