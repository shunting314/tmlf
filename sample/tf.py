# reference https://www.tensorflow.org/tensorboard/graphs
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import (
    Dense,
    Flatten,
    Dropout,
)
import tensorflow as tf
from tensorflow import keras

def main():
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(8, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation="softmax"))
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    (train_images, train_labels), _ = keras.datasets.fashion_mnist.load_data()
    train_images = train_images / 255.0
    logdir = "/tmp/tb.log"
    tb_cb = keras.callbacks.TensorBoard(log_dir=logdir)
    model.fit(
        train_images,
        train_labels,
        batch_size=64,
        epochs=5,
        callbacks=[tb_cb])
    print("Fall into pdb")
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()
