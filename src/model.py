"""TensorFlow / Keras CNN Model Architectures (MobileNetV2 Transfer Learning and Custom CNN)."""

import tensorflow as tf
from tensorflow.keras import layers, models
from src.config import IMG_HEIGHT, IMG_WIDTH, CHANNELS, CLASS_NAMES, LEARNING_RATE


def build_transfer_model(
    num_classes: int = len(CLASS_NAMES),
    learning_rate: float = LEARNING_RATE,
    fine_tune_at: int = 100,
) -> tf.keras.Model:
    """
    Builds a Transfer Learning model using MobileNetV2 pretrained on ImageNet.
    
    Includes an integrated MobileNetV2 preprocessing layer so raw RGB [0, 255]
    arrays can be passed directly during inference.
    """
    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS), name="input_image")

    # Built-in normalization: scales [0, 255] to [-1, 1] as required by MobileNetV2
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)(inputs)

    # Base MobileNetV2 feature extractor
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS),
        include_top=False,
        weights="imagenet",
    )

    # Freeze base model weights up to `fine_tune_at` for stable transfer learning
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation="relu", name="dense_features")(x)
    x = layers.Dropout(0.3, name="dropout_regularization")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classification_head")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Potato_MobileNetV2")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def build_custom_cnn(
    num_classes: int = len(CLASS_NAMES),
    learning_rate: float = LEARNING_RATE,
) -> tf.keras.Model:
    """
    Alternative lightweight 4-block Custom CNN architecture for comparison.
    """
    model = models.Sequential([
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)),
        layers.Rescaling(1.0 / 255.0),

        # Block 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 3
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 4
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Classifier head
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ], name="Potato_Custom_CNN")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model

