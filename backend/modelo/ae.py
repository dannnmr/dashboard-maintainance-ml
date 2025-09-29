import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

def build_lstm_ae(steps: int, feats: int) -> tf.keras.Model:
    inp = layers.Input(shape=(steps, feats))
    x = layers.Masking()(inp)
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(64)(x)
    z = layers.Dense(32, activation="relu")(x)
    x = layers.RepeatVector(steps)(z)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.LSTM(128, return_sequences=True)(x)
    out = layers.TimeDistributed(layers.Dense(feats))(x)
    m = models.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return m

def train_ae(Xtr_seq, Xva_seq, patience=6, verbose=1):
    m = build_lstm_ae(Xtr_seq.shape[1], Xtr_seq.shape[2])
    es  = callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    rlr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)
    hist = m.fit(Xtr_seq, Xtr_seq, validation_data=(Xva_seq, Xva_seq),
                 epochs=60, batch_size=128, callbacks=[es, rlr], verbose=verbose)
    return m, hist.history

def recon_error(model, Xseq):
    rec = model.predict(Xseq, verbose=0)
    return np.mean((Xseq - rec)**2, axis=(1,2))
