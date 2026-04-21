import numpy as np
import tensorflow as tf
from pathlib import Path

PREFIX = "TinyML_MNIST"
EPOCHS = 12
BATCH_SIZE = 64
NUM_CALIBRATION_SAMPLES = 100
IMAGE_SIZE = (28, 28)
DATASET_DIR = Path("mnist_samples")


def parse_label_from_path(path):
    stem_parts = Path(path).stem.split("_")
    return int(stem_parts[-1])


def load_split_from_directory(split_name):
    split_dir = DATASET_DIR / split_name
    image_paths = sorted(split_dir.glob("*.png"))

    if not image_paths:
        raise FileNotFoundError(
            f"No PNG files found in {split_dir.resolve()}. "
            "Run export_mnist_samples.py first."
        )

    images = []
    labels = []

    for image_path in image_paths:
        image_bytes = tf.io.read_file(str(image_path))
        image = tf.io.decode_png(image_bytes, channels=1)
        image = tf.image.resize(image, IMAGE_SIZE, method="nearest")
        images.append(image.numpy())
        labels.append(parse_label_from_path(image_path))

    images = np.stack(images).astype("float32") / 255.0
    labels = np.array(labels, dtype="uint8")

    return images, labels


def load_mnist_from_images():
    x_train, y_train = load_split_from_directory("train")
    x_test, y_test = load_split_from_directory("test")

    # Loaded PNGs already include the channel dimension.
    if x_train.shape[1:] != (28, 28, 1) or x_test.shape[1:] != (28, 28, 1):
        raise ValueError("Expected images with shape (28, 28, 1)")

    return (x_train, y_train), (x_test, y_test)


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.RandomRotation(0.03),
        tf.keras.layers.RandomTranslation(0.05, 0.05),
        tf.keras.layers.Conv2D(16, kernel_size=3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Conv2D(48, kernel_size=3, activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def representative_dataset(x_train):
    samples = x_train[:NUM_CALIBRATION_SAMPLES]

    def generator():
        for sample in samples:
            yield [np.expand_dims(sample, axis=0)]

    return generator


def convert_to_tflite(model, x_train):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset(x_train)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    return converter.convert()


def write_c_header(tflite_path, output_header_path):
    with open(tflite_path, "rb") as tflite_file:
        tflite_content = tflite_file.read()

    hex_lines = [
        ", ".join(f"0x{byte:02x}" for byte in tflite_content[i:i + 12])
        for i in range(0, len(tflite_content), 12)
    ]
    hex_array = ",\n  ".join(hex_lines)

    with open(output_header_path, "w") as header_file:
        header_file.write("const unsigned char model[] = {\n  ")
        header_file.write(f"{hex_array}\n")
        header_file.write("};\n\n")
        header_file.write(f"const unsigned int model_len = {len(tflite_content)};\n")


def main():
    (x_train, y_train), (x_test, y_test) = load_mnist_from_images()
    model = build_model()

    model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
    )

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")

    h5_path = PREFIX + ".h5"
    tflite_path = PREFIX + ".tflite"
    header_path = PREFIX + ".h"

    model.save(h5_path)
    tflite_model = convert_to_tflite(model, x_train)

    with open(tflite_path, "wb") as tflite_file:
        tflite_file.write(tflite_model)

    write_c_header(tflite_path, header_path)

    print(f"Saved Keras model to {h5_path}")
    print(f"Saved TFLite model to {tflite_path}")
    print(f"Saved C header to {header_path}")


if __name__ == "__main__":
    main()
