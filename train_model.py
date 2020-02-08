from __future__ import absolute_import, division, print_function, unicode_literals
import functools

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from kerastuner.tuners import RandomSearch
from tensorflow.keras.layers import Dense, Dropout, Input, Activation, Bidirectional, LSTM, Reshape
from tensorflow.keras.models import model_from_json
import datetime

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)

np.set_printoptions(precision=3, suppress=True)

TRAIN_FILE_PATH = './data/train.txt'
TEST_FILE_PATH = './data/test.txt'
PREDICT_FILE_PATH = './data/predict.txt'
TEMP_DIR_PATH = './temp'
MODEL_H5_FILE_PATH = './core/usecase/tensorflow_model/model.h5'
MODEL_JSON_FILE_PATH = './core/usecase/tensorflow_model/model.json'

# Ref: https://www.tensorflow.org/tutorials/keras/text_classification_with_hub
EMBEDDING_URL = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
EPOCHS = 10


def load_datasets(batch_size):
    def parse(filename):
        values = tf.strings.split([filename], sep=',', maxsplit=1).values
        return values[1], tf.strings.to_number(values[0])

    train_dataset = tf.data.TextLineDataset(TRAIN_FILE_PATH).map(parse).batch(batch_size)
    train_dataset.shuffle(500)
    test_dataset = tf.data.TextLineDataset(TEST_FILE_PATH).map(parse).batch(batch_size)
    predict_dataset = tf.data.TextLineDataset(PREDICT_FILE_PATH).map(parse).batch(batch_size)

    return train_dataset, test_dataset, predict_dataset


def build_model(hp):
    embed = hub.KerasLayer(EMBEDDING_URL, input_shape=[],
                           dtype=tf.string, trainable=True)

    # Step 1: Define the hyper-parameters
    LR = hp.Choice('learning_rate', [0.001, 0.0005, 0.0001])
    DROPOUT_RATE = hp.Float('dropout_rate', 0.0, 0.5, 5)
    NUM_DIMS = hp.Int('num_dims', 8, 32, 8)
    NUM_LAYERS = hp.Int('num_layers', 1, 3)
    # Step 2: Replace static values with hyper-parameters
    model = tf.keras.models.Sequential()
    model.add(Input(shape=(), name="input", dtype=tf.string))
    model.add(embed)
    model.add(Reshape((1, 20)))
    model.add(Bidirectional(LSTM(64)))
    for _ in range(NUM_LAYERS):
        model.add(Dense(NUM_DIMS))
        model.add(Dropout(DROPOUT_RATE))
        model.add(Activation('relu'))
    model.add(Dense(1, activation='sigmoid', name="output"))
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.001), metrics=['accuracy'])
    return model


def search_and_pick_one_model(train_dataset, build_model, tensorboard_cb):
    tuner = RandomSearch(
        build_model,
        objective='accuracy',
        max_trials=5,
        executions_per_trial=3,
        directory=TEMP_DIR_PATH,
    )

    tuner.search_space_summary()

    tuner.search(train_dataset, epochs=EPOCHS, callbacks=[tensorboard_cb])

    models = tuner.get_best_models(num_models=1)

    tuner.results_summary()

    return models[0]


def save_and_verify_model_data(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open(MODEL_JSON_FILE_PATH, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(MODEL_H5_FILE_PATH)
    print("Saved model to disk")
    # load model
    json_file = open(MODEL_JSON_FILE_PATH, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json, custom_objects={'KerasLayer': hub.KerasLayer})
    # load weights into new model
    loaded_model.load_weights(MODEL_H5_FILE_PATH)
    # verify model data
    print(loaded_model.predict(np.array(["test"])))


log_dir = "logs/search/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

train_dataset, test_dataset, predict_dataset = load_datasets(64)

model = search_and_pick_one_model(train_dataset, build_model, tensorboard_callback)

model.summary()

print("\nevaluate model performance")
test_loss, test_accuracy = model.evaluate(test_dataset)
print('\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))

predict = predict_dataset.take(1)
for batch, label in predict:
    print("{}: {}\n".format(batch, label))

predictions = model.predict(predict)

for prediction, v in zip(predictions[:20], list(predict)[0][1][:20]):
    print("Predicted: {:.2%}".format(prediction[0]),
          " | Actual outcome: ",
          ("valid" if bool(v) else "invalid"))

save_and_verify_model_data(model)
