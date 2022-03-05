import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import MeanSquaredError as MSE
from tensorflow.keras.losses import MeanAbsoluteError as MAE
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import tensorflow.lite as tflite
import tensorflow_model_optimization as tfmot
from tensorflow.keras.losses import MeanSquaredError as MSE
import tempfile
import argparse
import zlib


class WindowGenerator:
    def __init__(self,batch_size, input_width,output_width, label_options, mean, std):
        self.batch_size = batch_size
        self.output_width = output_width
        self.input_width = input_width
        self.label_options = label_options
        self.mean = tf.reshape(tf.convert_to_tensor(mean), [1, 1, 2])
        self.std = tf.reshape(tf.convert_to_tensor(std), [1, 1, 2])

    def split_window(self, features):
        inputs = features[:, :self.input_width, :]
        labels = features[:, -self.output_width:, :]
        num_labels = 2

        inputs.set_shape([None, self.input_width, 2])
        labels.set_shape([None,  self.output_width, num_labels])

        return inputs, labels

    def normalize(self, features):
        features = (features - self.mean) / (self.std + 1.e-6)

        return features

    def preprocess(self, features):
        inputs, labels = self.split_window(features)
        inputs = self.normalize(inputs)

        return inputs, labels

    def make_dataset(self, data, train):
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=data,
                targets=None,
                sequence_length=self.input_width+self.output_width,
                sequence_stride=1,
                batch_size=self.batch_size)
        ds = ds.map(self.preprocess)
        ds = ds.cache()
        if train is True:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)

        return ds


def gen_dataset(train_data, val_data, test_data,  input_width, output_width, batch_size, label_options):
    mean_train = train_data.mean(axis=0)
    std_train = train_data.std(axis=0)

    generator = WindowGenerator(batch_size, input_width, output_width, label_options, mean_train, std_train)
    train_ds = generator.make_dataset(train_data, True)
    val_ds = generator.make_dataset(val_data, False)
    test_ds = generator.make_dataset(test_data, False)

    return train_ds, val_ds, test_ds


class MultiOutputMAE(tf.keras.metrics.Metric):
    def __init__(self, name='mean_absolute_error', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight('total', initializer='zeros', shape=(2,))
        self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        error = tf.abs(y_pred - y_true)

        error = tf.reduce_mean(error, axis=[0, 1])
        self.total.assign_add(error)
        self.count.assign_add(1.)

    def reset_states(self):
        self.count.assign(tf.zeros_like(self.count))
        self.total.assign(tf.zeros_like(self.total))
        return

    def result(self):
        result = tf.math.divide_no_nan(self.total, self.count)

        return result

def save_zip_model(tf_lite_model, version):
    tf_lite_model_dir = f"Group18_th_{version}.tflite.zlib "
    with open (tf_lite_model_dir, "wb") as f:
        tf_lite_compressed = zlib.compress(tf_lite_model)
        f.write(tf_lite_compressed)
    return os.path.getsize(tf_lite_model_dir)

def gen_model(input_shape, output_shape, scaling_factor, choose="MLP"):
    inputs = keras.Input(shape=input_shape)
    model = None

    if choose == "MLP":
        x = keras.layers.Flatten(input_shape=input_shape)(inputs)
        x = keras.layers.Dense(units=int(128 * scaling_factor), activation="relu")(x)
        # x = keras.layers.Dense(units = 128*scaling_factor, activation = "relu") (x)
        x = keras.layers.Dense(units=2 * output_shape[0])(x)
        outputs_mlp = keras.layers.Reshape(output_shape)(x)

        model = keras.Model(inputs=inputs, outputs=outputs_mlp, name="MLP")

    if choose == "MLP_2":
        x = keras.layers.Flatten(input_shape=input_shape)(inputs)
        x = keras.layers.Dense(units=int(128 * scaling_factor), activation="relu")(x)
        x = keras.layers.Dense(units=128 * scaling_factor, activation="relu")(x)
        x = keras.layers.Dense(units=2 * output_shape[0])(x)
        outputs_mlp = keras.layers.Reshape(output_shape)(x)

        model = keras.Model(inputs=inputs, outputs=outputs_mlp, name="MLP_2")

    if choose == "CNN_separable":
        x = tf.keras.layers.SeparableConv1D(filters=int(64 * scaling_factor), kernel_size=3, activation="relu")(inputs)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(units=int(64 * scaling_factor), activation="relu")(x)
        x = keras.layers.Dense(units=2 * output_shape[0])(x)
        outputs_cnn = keras.layers.Reshape(output_shape)(x)

        model = keras.Model(inputs=inputs, outputs=outputs_cnn, name="CNN_separable")

    if choose == "CNN_separable_3":
        x = tf.keras.layers.SeparableConv1D(filters=int(64 * scaling_factor), kernel_size=3, activation="relu")(inputs)
        x = keras.layers.Flatten()(x)
        # x = keras.layers.Dense(units = int(64 * scaling_factor), activation = "relu") (x)
        x = keras.layers.Dense(units=2 * output_shape[0])(x)
        outputs_cnn = keras.layers.Reshape(output_shape)(x)

        model = keras.Model(inputs=inputs, outputs=outputs_cnn, name="CNN_separable_3")

    if choose == "CNN":
        x = keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu")(inputs)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(units=32, activation="relu")(x)
        x = keras.layers.Dense(units=2 * output_shape[0])(x)
        outputs_cnn = keras.layers.Reshape(output_shape)(x)

        model = keras.Model(inputs=inputs, outputs=outputs_cnn, name="CNN")

    if choose == "LSTM":
        x = keras.layers.LSTM(units=64)(inputs)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(units=2 * output_shape[0])(x)
        outputs_lstm = keras.layers.Reshape(output_shape)(x)

        model = keras.Model(inputs=inputs, outputs=outputs_lstm, name="LSTM")

    return model


def compile_and_train(model, train_ds, learning_rate, n_epochs=20):
    # then we can compile and print the summary of the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=MSE(), metrics=[MultiOutputMAE()])
    summary = model.summary()
    print(f"Start training {model.name} model")
    history = model.fit(train_ds, epochs=n_epochs)

    return model, summary, history


def saving_model(model):
    saving_path = os.path.join('.', 'models', model.name)
    model.save(saving_path)

    return saving_path


def representative_data_gen(train_ds):
    for input_value in train_ds.take(100):
        yield [input_value]


def pruning(saving_path, train_ds, epochs=100, final_sparsity=0.9):
    model_new = tf.keras.models.load_model(filepath=saving_path, custom_objects={'MultiOutputMAE': MultiOutputMAE})
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    model_for_pruning = prune_low_magnitude(model_new)

    pruning_params = {
        "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.3, final_sparsity=final_sparsity, begin_step=len(train_ds) * 5,
            end_step=len(train_ds) * 15)
    }

    model = prune_low_magnitude(model_new, **pruning_params)

    batch_size = 32
    input_width = 6
    input_shape = [batch_size, input_width, 2]
    model.build(input_shape)
    model_for_pruning.compile(
        optimizer="adam",
        loss=MSE(),
        metrics=[MultiOutputMAE()])

    print(model_for_pruning.summary())
    logdir = tempfile.mkdtemp()

    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
    ]

    model_for_pruning.fit(train_ds,
                          batch_size=batch_size, epochs=epochs, callbacks=callbacks)

    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    _, pruned_keras_file = tempfile.mkstemp('.h5')
    tf.keras.models.save_model(model_for_export, pruned_keras_file, include_optimizer=False)
    print('Saved pruned Keras model to:', pruned_keras_file)
    return model_for_export


def evaluate_TF_LITE(path, test_ds):
    interpreter = tflite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    outputs = []
    labels = []

    dataset = test_ds.unbatch().batch(1)
    for data in dataset:
        print('data', data)
        my_input = np.array(data[0], dtype=np.float32)
        print('My_input', my_input)
        label = np.array(data[1], dtype=np.float32)
        labels.append(label)
        interpreter.set_tensor(input_details[0]["index"], my_input)
        interpreter.invoke()
        my_output = interpreter.get_tensor(output_details[0]["index"])
        outputs.append(my_output[0])

    outputs = np.squeeze(np.array(outputs))
    labels = np.squeeze(np.array(labels))

    error = np.absolute(outputs - labels)
    mean_1 = np.mean(error, axis=1)
    mae = np.mean(mean_1, axis=0)
    temp_MAE = mae[0]
    hum_MAE = mae[1]

    return temp_MAE, hum_MAE


def get_gzipped_model_size(file):
  # Returns size of gzipped model, in bytes.
    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(file)
    return os.path.getsize(zipped_file)


def save_TF_Lite_quant(path, type_model):
    converter = tflite.TFLiteConverter.from_keras_model(path)
    pruned_tflite_model = converter.convert()

    _, pruned_tflite_file = tempfile.mkstemp('.tflite')

    with open(pruned_tflite_file, 'wb') as f:
        f.write(pruned_tflite_model)

    print('Saved pruned TFLite model to:', pruned_tflite_file)
    convert = tflite.TFLiteConverter.from_keras_model(path)
    converter.optimizations = [tflite.Optimize.DEFAULT]
    quantized_and_pruned_tflite_model = converter.convert()

    #_, quantized_and_pruned_tflite_file = tempfile.mkstemp(".tflite")
    if type_model == "a":
        path_saving = "Group18_th_a.tflite"

    elif type_model == "b":
        path_saving = "Group18_th_b.tflite"

    with open(path_saving, 'wb') as f:
        f.write(quantized_and_pruned_tflite_model)

    abs_path_saving = os.path.abspath(path_saving)
    print('Saved quantized and pruned TFLite model to:', abs_path_saving)

    print("Size of gzipped pruned and quantized TFlite model: %.2f bytes" % (
        save_zip_model(quantized_and_pruned_tflite_model,type_model)))

    return abs_path_saving


def main(args):

    type_model = args.version
    # set random seed to get always same value
    seed = 42
    tf.random.set_seed(seed)

    # download and read data
    path = tf.keras.utils.get_file(
        origin="https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip", \
        fname="jena_climate_2009_2016.csv.zip", \
        extract=True, \
        cache_dir=".", cache_subdir="data")

    path, _ = os.path.splitext(path)  # to remove .zip from path
    data = pd.read_csv(path)

    # select humidity and temperature
    sel = [data.columns[2], data.columns[5]]
    temp_hum = data.iloc[:, data.columns.isin(sel)]

    # convert them in float32 numpy array
    temp_hum = temp_hum.to_numpy(dtype=np.float32)

    # split the data into train 0.7, validation 0.1, test 0.2 sets
    n = len(temp_hum)
    train_data = temp_hum[0:int(n * 0.7)]
    val_data = temp_hum[int(n * 0.7):int(n * 0.8)]
    test_data = temp_hum[int(n * 0.8):]

    input_width = 6

    # label options 0 temperature, 1 humidity, 2 both
    label_options = 2

    # hyperparameters:
    if type_model == "a":
        output_width = 3
        batch_size = 512
        chosen_model = "MLP"
        learning_rate_training = 0.01
        n_epochs_training = 2
        n_epochs_pruning = 2

        # to decrease the size of the layers
        scaling_factor = 0.025

        final_sparsity = 0.9


    elif type_model == "b":
        output_width = 9
        batch_size = 32
        chosen_model = "MLP_2"
        learning_rate_training = 0.01
        n_epochs_training = 20
        n_epochs_pruning = 20

        # to decrease the size of the layers
        scaling_factor = 0.03

        final_sparsity = 0.9

    #generate the dataset
    train_ds, val_ds, test_ds = gen_dataset(train_data,val_data,test_data, input_width, output_width, batch_size, label_options)

    # generate the model
    inp = (input_width, label_options)
    out = (output_width, label_options)
    model = gen_model(inp, out, scaling_factor, chosen_model)

    #train the model
    trained_model, summary, history = compile_and_train(model, train_ds, learning_rate_training, n_epochs_training)
    saving_path = saving_model(trained_model)

    #prune the model
    pruned_model = pruning(saving_path,train_ds, n_epochs_pruning, final_sparsity)

    #save as tf lite and quantize
    path_saved_model = save_TF_Lite_quant(pruned_model, type_model)
    temp_MAE, hum_MAE = evaluate_TF_LITE(path_saved_model, test_ds)


    print(f"Temp MAE:{temp_MAE}", f"\nHum MAE:{hum_MAE}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str,
                        choices=['a', 'b'], required=True,
                        help='Which version of the model run')
    args = parser.parse_args()
    main(args)