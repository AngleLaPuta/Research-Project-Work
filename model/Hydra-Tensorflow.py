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
EPOCHS = 1000
LEARNING_RATE = 0.0001


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

    s = Conv2D(32, (3, 3), padding='same', activation='relu')(aug_sar)
    s = BatchNormalization()(s)
    s = Conv2D(64, (3, 3), strides=2, padding='same', activation='relu')(s)
    s = GlobalAveragePooling2D()(s)

    base_rgb = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_rgb.trainable = False

    r = base_rgb(aug_rgb)
    r = GlobalAveragePooling2D()(r)

    fused = Concatenate()([s, r])

    x = Dense(256, activation='relu')(fused)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[sar_input, rgb_input], outputs=output)

    model.compile(
        optimizer=Adam(LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def load_data_by_folder(sar_root, rgb_root):
    X_sar, X_rgb, Y = [], [], []
    folders = sorted([f for f in os.listdir(rgb_root) if os.path.isdir(os.path.join(rgb_root, f))])

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

            sar_files = sorted([f for f in os.listdir(sar_path) if f.endswith('.png')])
            if len(sar_files) < SAR_DEPTH: continue

            indices = np.linspace(0, len(sar_files) - 1, SAR_DEPTH, dtype=int)
            sar_vol = []
            for i in indices:
                s_img = cv2.imread(os.path.join(sar_path, sar_files[i]), cv2.IMREAD_GRAYSCALE)
                s_img = cv2.resize(s_img, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
                v_max = np.percentile(s_img, 98)
                s_img = np.clip(s_img, 0, v_max) / (v_max + 1e-6)
                sar_vol.append(s_img)

            X_sar.append(np.stack(sar_vol, axis=-1))
            X_rgb.append(img_rgb)
            Y.append(1 if folder.startswith('1_') else 0)
        except Exception:
            continue

    return np.array(X_sar), np.array(X_rgb), np.array(Y)


def run():
    SAR_PATH = r"C:\Users\DooDooFartious\Research-Project-Work\dataset\SAR Image"
    RGB_PATH = r"C:\Users\DooDooFartious\Research-Project-Work\dataset\RGB Image"

    xs, xr, y = load_data_by_folder(SAR_PATH, RGB_PATH)

    xs_train, xs_val, xr_train, xr_val, y_train, y_val = train_test_split(
        xs, xr, y, test_size=0.2, stratify=y, random_state=42
    )

    model = build_hydra_final()

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint('best_fixed_model.h5', save_best_only=True)
    ]

    history = model.fit(
        x={'sar_input': xs_train, 'rgb_input': xr_train},
        y=y_train,
        validation_data=({'sar_input': xs_val, 'rgb_input': xr_val}, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )

    # Visualization
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1);
    plt.plot(history.history['accuracy'], label='Train');
    plt.plot(history.history['val_accuracy'], label='Val');
    plt.title('Accuracy');
    plt.legend()
    plt.subplot(1, 2, 2);
    plt.plot(history.history['loss'], label='Train');
    plt.plot(history.history['val_loss'], label='Val');
    plt.title('Loss');
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run()
