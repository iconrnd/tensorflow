import os
from pathlib import  Path
import tempfile
import tensorflow as tf

strategy = tf.distribute.MultiWorkerMirroredStrategy()
resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()

print(f"Starting task {resolver.task_type} #{resolver.task_id}")

if resolver.task_type == 'chief':
    model_dir = os.getenv("AIP_MODEL_DIR") # Provided by Vertex AI
    tensorboard_log_dir = os.getenv("AIP_TENSORBOARD_LOG_DIR")
    checkpoint_dir = os.getenv("AIP_CHECKPOINT_DIR")
else:
    tmp_dir = Path(tempfile.mkdtemp()) # tmpdirs for non-chief workers
    model_dir = tmp_dir / "model"
    tensorboard_log_dir = tmp_dir / "logs"
    checkpoint_dir = tmp_dir / "ckpt"

callbacks = [tf.keras.callbacks.TensorBoard(tensorboard_log_dir),
             tf.keras.callbacks.ModelCheckpoint(checkpoint_dir)]

with strategy.scope():

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=[28, 28]))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(300, activation='relu'))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
             optimizer=tf.keras.optimizers.SGD(),
             metrics=[tf.keras.metrics.sparse_categorical_accuracy])

fashion_mnist=tf.keras.datasets.fashion_mnist.load_data()

(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist

X_train, y_train = X_train_full[:-5000], y_train_full[:-5000]
X_valid, y_valid = X_train_full[-5000:], y_train_full[-5000:]

# Data normalization
X_train, X_valid, X_test = X_train/255., X_valid/255., X_test/255. 

model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=10,
          callbacks=callbacks)

model.save(model_dir, save_format="tf")
