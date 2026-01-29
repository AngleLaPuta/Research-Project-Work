import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, Activation,
                                     GlobalAveragePooling2D, Dense, Dropout,
                                     Concatenate, RandomFlip, RandomRotation,
                                     MultiHeadAttention, LayerNormalization, Reshape, UpSampling2D, MaxPooling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
IMAGE_SIZE = 224
SAR_DEPTH = 10
BATCH_SIZE = 4  # Reduced to 4 to prevent OOM errors overnight
EPOCHS_INITIAL = 1000
EPOCHS_FINE = 1000
LEARNING_RATE_INITIAL = 1e-4
LEARNING_RATE_FINE = 5e-7


def build_hydra_spatial_attention():
    sar_input = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, SAR_DEPTH), name='sar_input')
    rgb_input = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name='rgb_input')

    seed = 42

    def augment(x):
        x = RandomFlip("horizontal_and_vertical", seed=seed)(x)
        x = RandomRotation(0.2, seed=seed)(x)
        return x

    aug_sar = augment(sar_input)
    aug_rgb = augment(rgb_input)

    # --- SAR Branch (Downsampled to 28x28 for Memory) ---
    s = Conv2D(64, (3, 3), strides=2, padding='same', activation='relu')(aug_sar)
    s = BatchNormalization()(s)
    s = Conv2D(128, (3, 3), strides=2, padding='same', activation='relu')(s)
    s = MaxPooling2D(pool_size=(2, 2))(s)  # Results in 28x28
    s_spatial = s
    s_seq = Reshape((28 * 28, 128))(s_spatial)

    # --- RGB Branch (Matched to 28x28) ---
    base_rgb = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_rgb.trainable = False

    r_spatial = base_rgb(aug_rgb)  # Output is 7x7x1280
    r_resized = Conv2D(128, (1, 1), padding='same')(r_spatial)
    r_resized = UpSampling2D(size=(4, 4))(r_resized)  # Results in 28x28
    r_seq = Reshape((28 * 28, 128))(r_resized)

    # --- Spatial Cross-Attention ---
    attn_layer = MultiHeadAttention(num_heads=4, key_dim=128, name='spatial_attn')
    query_out, attn_scores = attn_layer(
        query=r_seq,
        value=s_seq,
        key=s_seq,
        return_attention_scores=True
    )

    attended_features = LayerNormalization()(query_out + r_seq)

    # Global Pooling for Classification
    fused_spatial = GlobalAveragePooling2D()(Reshape((28, 28, 128))(attended_features))
    sar_global = GlobalAveragePooling2D()(s_spatial)

    final_features = Concatenate()([fused_spatial, sar_global])

    x = Dense(256, activation='relu')(final_features)
    x = Dropout(0.4)(x)
    output = Dense(1, activation='sigmoid', name='classification_output')(x)

    # Define training and inference models
    train_model = Model(inputs=[sar_input, rgb_input], outputs=output)
    viz_model = Model(inputs=[sar_input, rgb_input], outputs=[output, attn_scores])

    train_model.compile(
        optimizer=Adam(LEARNING_RATE_INITIAL),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return train_model, viz_model, base_rgb


def load_data_by_folder(sar_root, rgb_root):
    X_sar, X_rgb, Y = [], [], []
    folders = sorted([f for f in os.listdir(rgb_root) if os.path.isdir(os.path.join(rgb_root, f))])
    print(f"Loading data from {len(folders)} folders...")

    for folder in folders:
        try:
            rgb_path = os.path.join(rgb_root, folder, f"{folder}.jpg")
            sar_path = os.path.join(sar_root, folder)
            if not os.path.exists(rgb_path) or not os.path.exists(sar_path): continue

            img_rgb = cv2.imread(rgb_path)
            if img_rgb is None: continue
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE)) / 255.0

            sar_files = sorted([f for f in os.listdir(sar_path) if f.endswith('.png')])[:SAR_DEPTH]
            if len(sar_files) < SAR_DEPTH: continue

            sar_vol = []
            for s_f in sar_files:
                s_img = cv2.imread(os.path.join(sar_path, s_f), cv2.IMREAD_GRAYSCALE)
                s_img = cv2.resize(s_img, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
                v_max = np.percentile(s_img, 98)
                sar_vol.append(np.clip(s_img, 0, v_max) / (v_max + 1e-6))

            X_sar.append(np.stack(sar_vol, axis=-1))
            X_rgb.append(img_rgb)
            Y.append(1 if folder.startswith('1_') else 0)
        except:
            continue
    return np.array(X_sar), np.array(X_rgb), np.array(Y)


def run():
    SAR_PATH = r"C:\Users\DooDooFartious\Research-Project-Work\dataset\SAR Image"
    RGB_PATH = r"C:\Users\DooDooFartious\Research-Project-Work\dataset\RGB Image"

    xs, xr, y = load_data_by_folder(SAR_PATH, RGB_PATH)
    if len(y) == 0:
        print("Error: No data loaded.")
        return

    xs_train, xs_val, xr_train, xr_val, y_train, y_val = train_test_split(
        xs, xr, y, test_size=0.2, stratify=y, random_state=42
    )

    model, viz_model, base_rgb = build_hydra_spatial_attention()

    print("\n>>> Phase 1: Training Classification Head...")
    history1 = model.fit(
        x=[xs_train, xr_train], y=y_train,
        validation_data=([xs_val, xr_val], y_val),
        epochs=EPOCHS_INITIAL, batch_size=BATCH_SIZE,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)]
    )

    print("\n>>> Phase 2: Fine-Tuning...")
    base_rgb.trainable = True
    model.compile(optimizer=Adam(LEARNING_RATE_FINE), loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(
        x=[xs_train, xr_train], y=y_train,
        validation_data=([xs_val, xr_val], y_val),
        epochs=EPOCHS_INITIAL + EPOCHS_FINE,
        batch_size=BATCH_SIZE,
        initial_epoch=len(history1.history['accuracy']),
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)]
    )

    # --- VIZ: Center-pixel attention map ---
    idx = 0
    pred, weights = viz_model.predict([xs_val[idx:idx + 1], xr_val[idx:idx + 1]])

    map_dim = 28
    center_idx = (map_dim // 2) * map_dim + (map_dim // 2)
    heatmap = weights[0, 0, center_idx, :].reshape(map_dim, map_dim)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1);
    plt.imshow(xr_val[idx]);
    plt.title("RGB")
    plt.subplot(1, 2, 2);
    plt.imshow(heatmap, cmap='jet');
    plt.title("Spatial Attention Map (28x28)")
    plt.savefig('attention_result.png')
    plt.show()


if __name__ == "__main__":
    run()