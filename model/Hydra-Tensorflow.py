import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, Activation,
                                     Add, GlobalAveragePooling2D, Dense, Dropout,
                                     Concatenate, RandomFlip, RandomRotation)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# --- CONFIGURATION ---
IMAGE_SIZE = 224
SAR_DEPTH = 10
BATCH_SIZE = 8
EPOCHS = 1000
LEARNING_RATE = 0.0001


def residual_block(x, filters, stride=1):
    input_channels = x.shape[-1]
    shortcut = x
    reg = l2(0.01)  # High regularization to fight overfitting

    if stride != 1 or input_channels != filters:
        shortcut = Conv2D(filters, (1, 1), strides=stride, kernel_regularizer=reg)(x)
        shortcut = BatchNormalization()(shortcut)

    x = Conv2D(filters, (3, 3), strides=stride, padding='same', kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (3, 3), strides=1, padding='same', kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)

    return Activation('relu')(Add()([x, shortcut]))


def build_resnet_backbone(input_tensor):
    """Simplified ResNet logic for small datasets."""
    x = Conv2D(64, (7, 7), strides=2, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 256, stride=2)

    x = GlobalAveragePooling2D()(x)
    return x


def build_hydra_v2():
    sar_input = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, SAR_DEPTH), name='sar_input')
    rgb_input = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name='rgb_input')

    # Combined Augmentation
    combined = Concatenate()([sar_input, rgb_input])
    aug = RandomFlip("horizontal_and_vertical")(combined)
    aug = RandomRotation(0.2)(aug)

    aug_sar = aug[:, :, :, :SAR_DEPTH]
    aug_rgb = aug[:, :, :, SAR_DEPTH:]

    # Fuse SAR into RGB-like channels
    sar_features = Conv2D(3, (3, 3), padding='same', activation='relu')(aug_sar)
    fused = Concatenate()([sar_features, aug_rgb])

    # ResNet Backbone
    features = build_resnet_backbone(fused)

    # Output Head
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(features)
    x = Dropout(0.6)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[sar_input, rgb_input], outputs=output)
    model.compile(optimizer=Adam(LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# --- DATA LOADER (NOW WITH ERROR HANDLING) ---
def load_and_pair_data(sar_root, rgb_root):
    X_sar, X_rgb, Y = [], [], []

    if not os.path.exists(rgb_root):
        print(f"Error: RGB Path not found: {rgb_root}")
        return np.array([]), np.array([]), np.array([])

    # We iterate through the RGB folders and look for matching SAR folders
    folders = sorted([f for f in os.listdir(rgb_root) if os.path.isdir(os.path.join(rgb_root, f))])

    for folder in folders:
        try:
            # 1. Path Setup
            rgb_path = os.path.join(rgb_root, folder, f"{folder}.jpg")
            sar_path = os.path.join(sar_root, folder)

            # 2. Check if BOTH paths exist
            if not os.path.exists(rgb_path) or not os.path.exists(sar_path):
                # Silently skip missing pairs
                continue

            # 3. Load RGB
            img_rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            if img_rgb is None: continue
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE)) / 255.0

            # 4. Load SAR Slices
            sar_files = sorted([f for f in os.listdir(sar_path) if f.endswith('.png')])[:SAR_DEPTH]
            if len(sar_files) < SAR_DEPTH:
                continue

            sar_vol = []
            for s in sar_files:
                s_img = cv2.imread(os.path.join(sar_path, s), cv2.IMREAD_GRAYSCALE)
                if s_img is None: raise FileNotFoundError
                sar_vol.append(cv2.resize(s_img, (IMAGE_SIZE, IMAGE_SIZE)) / 255.0)

            # 5. Success - Append Data
            X_sar.append(np.stack(sar_vol, axis=-1))
            X_rgb.append(img_rgb)
            Y.append(1 if folder.startswith('1_') else 0)

        except Exception:
            # Skip any folders that cause issues (corrupt files, permission errors, etc.)
            continue

    return np.array(X_sar), np.array(X_rgb), np.array(Y)


def run():
    print("Loading data and ignoring missing paths...")
    xs, xr, y = load_and_pair_data(r"C:\Users\DooDooFartious\Research-Project-Work\dataset\SAR Image",
                                   r"C:\Users\DooDooFartious\Research-Project-Work\dataset\RGB Image")

    if len(y) == 0:
        print("Error: No data loaded. Check your root path strings.")
        return

    print(f"Dataset ready: {len(y)} samples. Distribution: {np.bincount(y)}")

    indices = np.arange(len(y))
    np.random.shuffle(indices)
    xs, xr, y = xs[indices], xr[indices], y[indices]

    model = build_hydra_v2()

    # Increased patience to give the model room to learn
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )

    history = model.fit(
        [xs, xr], y,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop]
    )

    # Plot Accuracy
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('Final Hydra Model Accuracy')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run()