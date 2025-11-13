#!/usr/bin/python3
'''
Enhanced and Comparative Single-Layer Hydra Model Script (FIXED)

This script implements the original Hydra model architecture and creates 10 different
experimental variations, focusing on regularization, capacity, and augmentation.

FIX: The tf.data.Dataset pipeline is corrected to pass a dictionary of features
matching the model's named inputs ('sar_input', 'image_input'), resolving the
"expects 2 input(s), but it received 1" ValueError.
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import (
    Input, Layer, Flatten, Conv2D, MaxPooling2D,
    GlobalAveragePooling2D, Dense, Activation,
    BatchNormalization, Add, Dropout
)
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler

# --- Global Configurations ---
IMAGE_SIZE = 64  # Reduced for faster synthetic training
SAR_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 1)
IMAGE_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 1)
EPOCHS = 500
BATCH_SIZE = 32
BASE_LR = 0.0001


# --- 1. Custom Layers and Base Architecture ---

class MaskScalingLayer(Layer):
    """
    Custom layer for Multi-modal Fusion: SAR (mask) scales the Image feature.
    """

    def __init__(self, **kwargs):
        super(MaskScalingLayer, self).__init__(**kwargs)
        self.conv_scaling = Conv2D(1, (1, 1), activation='linear')

    def call(self, inputs, **kwargs):
        mask, image = inputs  # mask is SAR, image is optical

        # Simple conv to act as a 'Dense on a mask' layer (channel-wise scaling)
        scaling_map = self.conv_scaling(mask)

        # Element-wise scaling of the image by the scaling map
        return image * scaling_map

    def compute_output_shape(self, input_shape):
        return input_shape[1]


def residual_block(x, filters, blocks, stride=1, l2_reg=0.0):
    """
    Residual Block with optional L2 regularization.
    """
    shortcut = x
    for i in range(blocks):
        # 1x1 Conv shortcut only for the first block if stride > 1 or filter size changes
        if i == 0:
            if stride != 1 or tf.keras.backend.int_shape(shortcut)[-1] != filters:
                shortcut = Conv2D(filters, (1, 1), strides=(stride, stride), kernel_regularizer=l2(l2_reg))(shortcut)

        # First 3x3 Conv
        x = Conv2D(filters, (3, 3), strides=(stride, stride), padding='same', kernel_regularizer=l2(l2_reg))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # Second 3x3 Conv (stride is reset to 1 after the first block)
        x = Conv2D(filters, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(l2_reg))(x)
        x = BatchNormalization()(x)

        stride = 1  # Subsequent blocks in the group have stride 1

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def resnet18(input_tensor, l2_reg=0.0, capacity_factor=1.0):
    """
    ResNet18 Architecture with optional L2 regularization and capacity factor.
    """
    filters = int(64 * capacity_factor)

    # Initial conv layer
    x = Conv2D(filters, (7, 7), strides=(2, 2), padding='same', kernel_regularizer=l2(l2_reg))(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # ResNet Stages
    x = residual_block(x, filters, 2, l2_reg=l2_reg)
    x = residual_block(x, int(128 * capacity_factor), 2, stride=2, l2_reg=l2_reg)
    x = residual_block(x, int(256 * capacity_factor), 2, stride=2, l2_reg=l2_reg)
    x = residual_block(x, int(512 * capacity_factor), 2, stride=2, l2_reg=l2_reg)

    # Global average pooling
    x = GlobalAveragePooling2D()(x)

    return x


def build_hydra_model(name, sar_shape, image_shape, l2_reg=0.0, capacity_factor=1.0, dense_units=1, dropout_rate=0.0):
    """
    Factory function to build the Hydra model with experimental parameters.
    """
    sar_input = Input(shape=sar_shape, name='sar_input')
    image_input = Input(shape=image_shape, name='image_input')

    # Fusion
    mask_scaling_layer = MaskScalingLayer()
    fused_features = mask_scaling_layer([sar_input, image_input])

    # Feature Extraction (ResNet18)
    features = resnet18(fused_features, l2_reg=l2_reg, capacity_factor=capacity_factor)

    # Classification Head (with optional Dropout and reduced capacity)
    if dropout_rate > 0.0:
        features = Dropout(dropout_rate)(features)

    # Use a hidden dense layer if requested for capacity testing, otherwise go straight to output
    if dense_units > 1:
        features = Dense(dense_units, activation='relu', kernel_regularizer=l2(l2_reg))(features)

    # Final classification layer
    output = Dense(1, activation='sigmoid', name='wetness_prediction')(features)

    model = Model(inputs=[sar_input, image_input], outputs=output, name=name)

    return model


# --- 2. Data Augmentation ---

def augment_data_mild(sar, image):
    """Mild augmentation: only rotation and horizontal flip."""
    sar_aug = tf.image.rot90(sar, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    image_aug = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    if tf.random.uniform(()) > 0.5:
        sar_aug = tf.image.flip_left_right(sar_aug)
        image_aug = tf.image.flip_left_right(image_aug)
    return sar_aug, image_aug


def augment_data_heavy(sar, image):
    """Heavy augmentation: rotation, flip, translation, contrast, and noise."""
    sar_aug, image_aug = augment_data_mild(sar, image)

    # Random brightness and contrast
    image_aug = tf.image.random_brightness(image_aug, max_delta=0.1)
    image_aug = tf.image.random_contrast(image_aug, lower=0.8, upper=1.2)

    # Random crop (proxy for translation)
    paddings = 8
    sar_aug_padded = tf.image.resize_with_crop_or_pad(sar_aug, IMAGE_SIZE + paddings, IMAGE_SIZE + paddings)
    image_aug_padded = tf.image.resize_with_crop_or_pad(image_aug, IMAGE_SIZE + paddings, IMAGE_SIZE + paddings)

    sar_aug = tf.image.random_crop(sar_aug_padded, size=[IMAGE_SIZE, IMAGE_SIZE, 1])
    image_aug = tf.image.random_crop(image_aug_padded, size=[IMAGE_SIZE, IMAGE_SIZE, 1])

    # Add slight Gaussian noise
    noise_sar = tf.random.normal(shape=tf.shape(sar_aug), mean=0.0, stddev=0.01, dtype=tf.float32)
    noise_image = tf.random.normal(shape=tf.shape(image_aug), mean=0.0, stddev=0.01, dtype=tf.float32)
    sar_aug = sar_aug + noise_sar
    image_aug = image_aug + noise_image

    # Clip values to ensure they stay in a reasonable range (0 to 1)
    sar_aug = tf.clip_by_value(sar_aug, 0.0, 1.0)
    image_aug = tf.clip_by_value(image_aug, 0.0, 1.0)

    return sar_aug, image_aug


# FIX: Changed signature to accept features as a dictionary (as returned by tf.data.Dataset)
def process_data(features, label, augment_fn):
    """Wrapper function to apply augmentation to a dataset element."""
    sar = features['sar_input']
    image = features['image_input']

    sar_aug, image_aug = augment_fn(sar, image)

    # Return dictionary of augmented features and the label
    return {'sar_input': sar_aug, 'image_input': image_aug}, label


# --- 3. Learning Rate Schedulers ---

def cosine_decay(epoch, initial_lrate=BASE_LR, total_epochs=EPOCHS):
    """Cosine learning rate decay schedule."""
    if epoch < 10:  # Warmup phase
        return initial_lrate

    # Cosine decay formula
    cos_inner = np.pi * (epoch - 10) / (total_epochs - 10)
    lrate = initial_lrate * 0.5 * (1 + np.cos(cos_inner))
    return lrate


# --- 4. Synthetic Data Generation (for demonstration) ---

def generate_synthetic_data(num_samples):
    """
    Generates synthetic data and creates tf.data.Dataset objects
    with the required dictionary structure for multi-input Keras models.
    """
    X_sar = np.random.rand(num_samples, IMAGE_SIZE, IMAGE_SIZE, 1).astype(np.float32) * 0.5
    X_image = np.random.rand(num_samples, IMAGE_SIZE, IMAGE_SIZE, 1).astype(np.float32) * 0.5

    # Simulate a "wet" patch (higher value)
    patch_size = IMAGE_SIZE // 4
    for i in range(num_samples):
        x_start, y_start = np.random.randint(0, IMAGE_SIZE - patch_size, 2)
        X_sar[i, x_start:x_start + patch_size, y_start:y_start + patch_size, 0] += 0.5
        X_image[i, x_start:x_start + patch_size, y_start:y_start + patch_size, 0] += 0.5

    X_sar = np.clip(X_sar, 0, 1)
    X_image = np.clip(X_image, 0, 1)

    center_region = X_sar[:, IMAGE_SIZE // 2 - patch_size // 2: IMAGE_SIZE // 2 + patch_size // 2,
                    IMAGE_SIZE // 2 - patch_size // 2: IMAGE_SIZE // 2 + patch_size // 2, 0]

    Y = (np.mean(center_region, axis=(1, 2)) > 0.5).astype(np.float32)

    # Split into train and validation
    split = int(0.8 * num_samples)
    X_sar_train, X_sar_val = X_sar[:split], X_sar[split:]
    X_image_train, X_image_val = X_image[:split], X_image[split:]
    Y_train, Y_val = Y[:split], Y[split:]

    # FIX: Create features as a dictionary for tf.data.Dataset
    X_train_dict = {'sar_input': X_sar_train, 'image_input': X_image_train}
    X_val_dict = {'sar_input': X_sar_val, 'image_input': X_image_val}

    # Creating the raw dataset where element is (features_dict, label)
    train_ds_raw = tf.data.Dataset.from_tensor_slices((X_train_dict, Y_train)).shuffle(len(Y_train))

    # The validation dataset must also be in the dictionary structure
    val_ds = tf.data.Dataset.from_tensor_slices((X_val_dict, Y_val)).batch(BATCH_SIZE).cache().prefetch(
        tf.data.AUTOTUNE)

    print(f"Synthetic Data Generated: Train Samples={len(Y_train)}, Validation Samples={len(Y_val)}")

    return train_ds_raw, val_ds


# --- 5. Experimental Variations Setup ---

# Define the 10 experimental variations
EXPERIMENTS = {
    # Baseline Model
    '01_Baseline': {
        'l2_reg': 0.0, 'capacity_factor': 1.0, 'augment_fn': augment_data_mild,
        'dense_units': 0, 'lr_scheduler': None, 'name': '01_Baseline_MildAug'
    },

    # Data Augmentation Tests
    '02_HeavyAug': {
        'l2_reg': 0.0, 'capacity_factor': 1.0, 'augment_fn': augment_data_heavy,
        'dense_units': 0, 'lr_scheduler': None, 'name': '02_HeavyAug'
    },

    # Model Capacity Reduction (Lower Variance)
    '03_LowCapacity_0.5x': {
        'l2_reg': 0.0, 'capacity_factor': 0.5, 'augment_fn': augment_data_mild,
        'dense_units': 0, 'lr_scheduler': None, 'name': '03_LowCap_0.5x'
    },
    '04_LowCapacity_Dropout': {
        'l2_reg': 0.0, 'capacity_factor': 1.0, 'augment_fn': augment_data_mild,
        'dense_units': 0, 'lr_scheduler': None, 'dropout_rate': 0.5, 'name': '04_Dropout_0.5'
    },

    # L2 Regularization (Weight Decay)
    '05_L2_Small_0.0001': {
        'l2_reg': 0.0001, 'capacity_factor': 1.0, 'augment_fn': augment_data_mild,
        'dense_units': 0, 'lr_scheduler': None, 'name': '05_L2_1e-4'
    },
    '06_L2_Medium_0.001': {
        'l2_reg': 0.001, 'capacity_factor': 1.0, 'augment_fn': augment_data_mild,
        'dense_units': 0, 'lr_scheduler': None, 'name': '06_L2_1e-3'
    },

    # Learning Rate Scheduling
    '07_LRSched_ReducePlateau': {
        'l2_reg': 0.0, 'capacity_factor': 1.0, 'augment_fn': augment_data_mild,
        'dense_units': 0, 'lr_scheduler': ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6),
        'name': '07_ReduceLR_Plateau'
    },
    '08_LRSched_Cosine': {
        'l2_reg': 0.0, 'capacity_factor': 1.0, 'augment_fn': augment_data_mild,
        'dense_units': 0, 'lr_scheduler': LearningRateScheduler(lambda epoch: cosine_decay(epoch, total_epochs=EPOCHS)),
        'name': '08_CosineLR_Schedule'
    },

    # Combined Experiments
    '09_HeavyAug_L2_Small': {
        'l2_reg': 0.0001, 'capacity_factor': 1.0, 'augment_fn': augment_data_heavy,
        'dense_units': 0, 'lr_scheduler': None, 'name': '09_HeavyAug_L2'
    },
    '10_LowCap_CosineLR': {
        'l2_reg': 0.0, 'capacity_factor': 0.5, 'augment_fn': augment_data_mild,
        'dense_units': 0, 'lr_scheduler': LearningRateScheduler(lambda epoch: cosine_decay(epoch, total_epochs=EPOCHS)),
        'name': '10_LowCap_Cosine'
    }
}


# --- 6. Training and Plotting ---

def train_and_evaluate_all():
    """Main function to run all experiments, train, and plot results."""
    tf.keras.backend.clear_session()

    # 1. Prepare Data
    train_ds_raw, val_ds = generate_synthetic_data(num_samples=1000)

    histories = {}

    print("\n--- Starting Model Training ---")

    for exp_id, params in EXPERIMENTS.items():
        print(f"\n[Running Experiment {exp_id}] - {params['name']}")

        # --- Build Model ---
        model = build_hydra_model(
            name=params['name'],
            sar_shape=SAR_SHAPE,
            image_shape=IMAGE_SHAPE,
            l2_reg=params.get('l2_reg', 0.0),
            capacity_factor=params.get('capacity_factor', 1.0),
            dense_units=params.get('dense_units', 0),
            dropout_rate=params.get('dropout_rate', 0.0)
        )

        # --- Compile Model ---
        optimizer = Adam(learning_rate=BASE_LR)
        model.compile(
            loss=BinaryCrossentropy(),
            optimizer=optimizer,
            metrics=["accuracy"]
        )

        # --- Prepare Data Pipelines ---
        # Apply augmentation function specific to the experiment
        # FIX: The lambda now passes the feature dictionary and label correctly
        train_ds_aug = train_ds_raw.map(
            lambda features, label: process_data(features, label, params['augment_fn']),
            num_parallel_calls=tf.data.AUTOTUNE
        ).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)

        # --- Callbacks ---
        # We cap the patience to 500 epochs, but set early stopping to monitor
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)]
        if params['lr_scheduler']:
            callbacks.append(params['lr_scheduler'])

        # --- Train ---
        history = model.fit(
            train_ds_aug,
            epochs=EPOCHS,
            validation_data=val_ds,
            callbacks=callbacks,
            verbose=0  # Set to 1 for live output
        )
        histories[params['name']] = history.history

        print(f"  Training finished. Best Val Loss: {min(history.history['val_loss']):.4f}")

    print("\n--- Training Complete. Generating Plots ---")
    plot_results(histories)
    print("Plots saved successfully to the current directory.")


def plot_results(histories):
    """
    Generates and saves two types of plots:
    1. Separate plots for each model's Loss and Accuracy.
    2. Combined plots for comparison.
    """
    num_models = len(histories)
    model_names = list(histories.keys())

    # --- 1. Separate Plots (Subplots) ---
    fig_sep, axes = plt.subplots(num_models, 2, figsize=(14, 4 * num_models))
    fig_sep.suptitle('Individual Model Performance (Loss & Accuracy)', fontsize=16)

    for i, name in enumerate(model_names):
        hist = histories[name]

        # Determine the correct axis index (handle case where num_models is 1, axes is 1D)
        if num_models == 1:
            ax_loss = axes[0]
            ax_acc = axes[1]
        else:
            ax_loss = axes[i, 0]
            ax_acc = axes[i, 1]

        # Loss subplot
        ax_loss.plot(hist['loss'], label='Train Loss')
        ax_loss.plot(hist['val_loss'], label='Validation Loss')
        ax_loss.set_title(f'[{name}] Loss')
        ax_loss.set_ylabel('Binary Crossentropy')
        ax_loss.legend()

        # Accuracy subplot
        ax_acc.plot(hist['accuracy'], label='Train Accuracy')
        ax_acc.plot(hist['val_accuracy'], label='Validation Accuracy')
        ax_acc.set_title(f'[{name}] Accuracy')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.legend()

    # Set common X label for the last row
    if num_models == 1:
        axes[0].set_xlabel('Epoch')
        axes[1].set_xlabel('Epoch')
    else:
        axes[-1, 0].set_xlabel('Epoch')
        axes[-1, 1].set_xlabel('Epoch')

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig_sep.savefig('training_comparison_separate_plots.png')
    plt.close(fig_sep)

    # --- 2. Combined Plots ---

    # Loss Comparison
    fig_loss, ax_loss = plt.subplots(1, 1, figsize=(12, 8))
    for name in model_names:
        ax_loss.plot(histories[name]['val_loss'], label=name)

    ax_loss.set_title('Validation Loss Comparison Across All Models')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Validation Loss')
    ax_loss.legend(loc='upper right', frameon=True)
    ax_loss.grid(True)
    fig_loss.savefig('training_comparison_loss.png')
    plt.close(fig_loss)

    # Accuracy Comparison
    fig_acc, ax_acc = plt.subplots(1, 1, figsize=(12, 8))
    for name in model_names:
        ax_acc.plot(histories[name]['val_accuracy'], label=name)

    ax_acc.set_title('Validation Accuracy Comparison Across All Models')
    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Validation Accuracy')
    ax_acc.legend(loc='lower right', frameon=True)
    ax_acc.grid(True)
    fig_acc.savefig('training_comparison_acc.png')
    plt.close(fig_acc)


if __name__ == "__main__":
    train_and_evaluate_all()
    print("Script execution complete.")