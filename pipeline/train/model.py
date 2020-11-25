import tensorflow as tf
import numpy as np
import logging
from datetime import datetime
import os
import argparse
import json


logger = tf.get_logger()
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    level=logging.INFO
)

print(f"Tensorflow Version - {tf.__version__}")


def build_model(lr=0.001):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(64, activation='relu'))

    model.compile(
        loss=tf.keras.losses.rmse,
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        metrics=['accuracy']
    )

    return model

def get_callbacks():
    checkpointdir = 'tmp/model-ckpt'

    class CustomLog(tf.keras.layers.callbacks.Callback):
        def on_epoch_end(self, epoch, logs):
            logging.info(f"epoch: {epoch+1}")
            logging.info(f"loss: {logs['loss']}")
            logging.info(f"accuracy: {logs['accuracy']}")
            logging.info(f"val_accuracy: {logs['val_accuracy']}")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpointdir),
        CustomLog()
    ]

    return callbacks


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf-mode',
                        type=str,
                        default='local',
                        help="Use either local or cloud storage"
    )

    parser.add_argument('--tf-export-dir',
                        type=str,
                        default='/tmp/export',
                        help="Local directory or GCS path to export to"
    )

    parser.add_argument('--tf-train-steps',
                        type=int,
                        default=3,
                        help="The number of trainin steps to perform"
    )

    parser.add_argument('--tf-learning-rate',
                        type=float,
                        default=0.001,
                        help="Use either local or cloud storage"
    )

    parser.add_argument('--tf-trainset',
                        type=str,
                        default='local',
                        help="Learning rate for training"
    )

    args, _ = parser.parse_known_args()

    return args

def main():
    args = parse_arguments()

    tf_config = os.environ.get('TF_CONFIG', '{}')
    logging.info(f"TF_INFO {tf_config}")
    tf_config_json = json.loads(tf_config)

    cluster = tf_config_json.get('cluster')
    job_name = tf_config_json.get('task', '{}').get('type')
    task_index = tf_config_json.get('task', '{}').get('index')
    logging.info(f"cluster={cluster} job_name={job_name} task_index={task_index}")

    is_chief = False
    if not job_name or job_name.lower() in ["chief", "master"]:
        is_chief = True
        logging.info("Will export model")
    else:
        logging.info("Will not export model")

    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    logging.info(f"Number of devices: {strategy.num_replicas_in_sync}")

    with strategy.scope():

        BUFFER_SIZE = 10000
        BATCH_SIZE = 64
        BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync

        learning_rate = float(args.learning_rate)
        logging.info(f"learning rate {learning_rate}")

        model = build_model(lr=learning_rate)

        logging.info("Training started...")

        TF_STEPS_PER_EPOCH = 5

        model.fit(
            trainset,
            epochs=int(args.tf_train_steps),
            steps_per_epoch=TF_STEPS_PER_EPOCH,
            validation_data=valset,
            validation_steps=1,
            callbacks=get_callbacks()
        )

        logging.info("Training Complete")

        if is_chief:
            model.save("model.h5")
            logging.info("Model saved...")

    model_loaded = tf.keras.models.load_model('model.h5')
    