import os
from pathlib import  Path
import tempfile
import argparse
import hypertune

parser = argparse.ArgumentParser()
parser.add_argument("--n_hidden", type=int, default=2)
parser.add_argument("--n_neurons", type=int, default=256)
parser.add_argument("--learning_rate", type=float, default=1e-2)
parser.add_argument("--optimizer", default="adam")

args = parser.parse_args()

import tensorflow as tf

def build_model(args):

    with tf.distribute.MirroredStrategy().scope():
    
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=[28, 28], dtype=tf.uint8))
        for _ in range(args.n_hidden):
            model.add(tf.keras.layers.Dense(args.n_neurons, activation='relu'))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))

        opt = tf.keras.optimizers.get(args.optimizer)
        opt.learning_rate = args.learning_rate
    
        model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
             optimizer=opt,
             #metrics=[tf.keras.metrics.sparse_categorical_accuracy])
             metrics=["accuracy"])

        return model
    
fashion_mnist=tf.keras.datasets.fashion_mnist.load_data()

(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist

X_train, y_train = X_train_full[:-5000], y_train_full[:-5000]
X_valid, y_valid = X_train_full[-5000:], y_train_full[-5000:]

# Data normalization
X_train, X_valid, X_test = X_train/255., X_valid/255., X_test/255. 

model = build_model(args)

history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=10)

model_dir = os.getenv("AIP_MODEL_DIR")

model.save(model_dir, save_format="tf")

hypertune = hypertune.HyperTune()
hypertune.report_hyperparameter_tuning_metric(
    hyperparameter_metric_tag="accuracy",
    metric_value=max(history.history["val_accuracy"]),
    global_step=model.optimizer.iterations.numpy(),
)