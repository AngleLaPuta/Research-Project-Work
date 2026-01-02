import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, Activation,
                                     GlobalAveragePooling2D, Dense, Dropout,
                                     Concatenate, RandomFlip, RandomRotation)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
IMAGE_SIZE = 224
SAR_DEPTH = 10
BATCH_SIZE = 32
EPOCHS_INITIAL = 1000
EPOCHS_FINE = 1000
LEARNING_RATE_INITIAL = 1e-4
# Phase 2 starts very low to handle the "system shock" of unfreezing
LEARNING_RATE_FINE = 5e-7


def build_hydra_final():
    sar_input = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, SAR_DEPTH), name='sar_input')
    rgb_input = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name='rgb_input')

    seed = 42

    def augment(x):
        x = RandomFlip("horizontal_and_vertical", seed=seed)(x)
        x = RandomRotation(0.2, seed=seed)(x)
        return x

    aug_sar = augment(sar_input)
    aug_rgb = augment(rgb_input)

    # SAR Branch: Specialized for 10-layer depth
    s = Conv2D(32, (3, 3), padding='same', activation='relu')(aug_sar)
    s = BatchNormalization()(s)
    s = Conv2D(64, (3, 3), strides=2, padding='same', activation='relu')(s)
    s = GlobalAveragePooling2D()(s)

    # RGB Branch: Multimodal transfer learning component
    base_rgb = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_rgb.trainable = False  # Start frozen for Phase 1

    r = base_rgb(aug_rgb)
    r = GlobalAveragePooling2D()(r)

    # Fusion of the two distinct modalities
    fused = Concatenate()([s, r])

    x = Dense(256, activation='relu')(fused)
    x = Dropout(0.4)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[sar_input, rgb_input], outputs=output)
    model.compile(optimizer=Adam(LEARNING_RATE_INITIAL), loss='binary_crossentropy', metrics=['accuracy'])
    return model, base_rgb


def load_data_by_folder(sar_root, rgb_root):
    X_sar, X_rgb, Y = [], [], []
    folders = sorted([f for f in os.listdir(rgb_root) if os.path.isdir(os.path.join(rgb_root, f))])

    print(f"Loading data from {len(folders)} folders...")

    for folder in folders:
        try:
            rgb_path = os.path.join(rgb_root, folder, f"{folder}.jpg")
            sar_path = os.path.join(sar_root, folder)

            if not os.path.exists(rgb_path) or not os.path.exists(sar_path):
                continue

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
                # Percentile clipping to handle SAR dynamic range/outliers
                v_max = np.percentile(s_img, 98)
                s_img = np.clip(s_img, 0, v_max) / (v_max + 1e-6)
                sar_vol.append(s_img)

            X_sar.append(np.stack(sar_vol, axis=-1))
            X_rgb.append(img_rgb)
            Y.append(1 if folder.startswith('1_') else 0)
        except Exception as e:
            print(f"Skipping {folder} due to error: {e}")
            continue

    return np.array(X_sar), np.array(X_rgb), np.array(Y)


def run():
    SAR_PATH = r"C:\Users\DooDooFartious\Research-Project-Work\dataset\SAR Image"
    RGB_PATH = r"C:\Users\DooDooFartious\Research-Project-Work\dataset\RGB Image"

    xs, xr, y = load_data_by_folder(SAR_PATH, RGB_PATH)

    # Stratified split ensures the folder-based classes are balanced across sets
    xs_train, xs_val, xr_train, xr_val, y_train, y_val = train_test_split(
        xs, xr, y, test_size=0.2, stratify=y, random_state=42
    )

    model, base_rgb = build_hydra_final()

    # --- PHASE 1: INITIAL TRAINING (FROZEN BACKBONE) ---
    print("\n>>> Phase 1: Training Classification Head...")
    history1 = model.fit(
        x={'sar_input': xs_train, 'rgb_input': xr_train}, y=y_train,
        validation_data=({'sar_input': xs_val, 'rgb_input': xr_val}, y_val),
        epochs=EPOCHS_INITIAL, batch_size=BATCH_SIZE,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
    )

    # --- PHASE 2: FINE-TUNING (UNFROZEN BACKBONE) ---
    print("\n>>> Phase 2: Unfreezing RGB Backbone for Final Polish...")
    base_rgb.trainable = True

    # Re-compile with the tiny learning rate for the recovery phase
    model.compile(
        optimizer=Adam(LEARNING_RATE_FINE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # LR Scheduler to break through plateaus
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=5, min_lr=1e-9, verbose=1
    )

    fine_tune_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=25,  # Extended patience to wait for the "V" recovery
            restore_best_weights=True,
            verbose=1
        ),
        lr_scheduler,
        tf.keras.callbacks.ModelCheckpoint('ultimate_multimodal_model.h5', save_best_only=True)
    ]

    history2 = model.fit(
        x={'sar_input': xs_train, 'rgb_input': xr_train}, y=y_train,
        validation_data=({'sar_input': xs_val, 'rgb_input': xr_val}, y_val),
        epochs=EPOCHS_FINE,
        batch_size=BATCH_SIZE,
        initial_epoch=len(history1.history['accuracy']),
        callbacks=fine_tune_callbacks
    )

    # --- VISUALIZATION ---
    full_acc = history1.history['accuracy'] + history2.history['accuracy']
    full_val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    full_loss = history1.history['loss'] + history2.history['loss']
    full_val_loss = history1.history['val_loss'] + history2.history['val_loss']

    plt.figure(figsize=(14, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(full_acc, label='Train Accuracy')
    plt.plot(full_val_acc, label='Val Accuracy')
    plt.axvline(x=len(history1.history['accuracy']), color='r', linestyle='--', label='Fine-tuning Starts')
    plt.title('Multimodal Accuracy')
    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(full_loss, label='Train Loss')
    plt.plot(full_val_loss, label='Val Loss')
    plt.axvline(x=len(history1.history['accuracy']), color='r', linestyle='--', label='Fine-tuning Starts')
    plt.title('Multimodal Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()