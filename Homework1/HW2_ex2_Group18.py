import tensorflow as tf
from tensorflow import keras
import os
import argparse
import numpy as np
import tensorflow_model_optimization as tfmot
import tempfile
import tensorflow.lite as tflite
import zlib

def get_dataset():
    path = tf.keras.utils.get_file(\
        origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip", \
        fname="mini_speech_commands.zip", \
        extract=True, \
        cache_dir=".", cache_subdir="data")

    data_dir, _ = os.path.splitext(path)
    commands = np.array(['stop', 'up', 'yes', 'right', 'left', 'no', 'down', 'go'], dtype=str)

    return path, commands


def create_data_partition():
    train_paths = list(np.loadtxt("kws_train_split.txt", dtype = "str"))
    val_paths = list(np.loadtxt("kws_val_split.txt", dtype = "str"))
    test_paths = list(np.loadtxt("kws_test_split.txt", dtype = "str"))
    return train_paths, val_paths, test_paths


class SignalGenerator:
    def __init__(self, labels, sampling_rate, frame_length, frame_step, \
                 num_mel_bins=None, lower_frequency=None, upper_frequency=None, \
                 num_coefficients=None, mfcc=False):
        self.labels = labels
        self.sampling_rate = sampling_rate
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.num_mel_bins = num_mel_bins
        self.lower_frequency = lower_frequency
        self.upper_frequency = upper_frequency
        self.num_coefficients = num_coefficients
        num_spectrogram_bins = (frame_length) // 2 + 1

        if mfcc is True:
            self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                self.num_mel_bins, num_spectrogram_bins, self.sampling_rate,
                self.lower_frequency, self.upper_frequency)
            self.preprocess = self.preprocess_with_mfcc
        else:
            self.preprocess = self.preprocess_with_stft

    def read(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        label = parts[-2]
        label_id = tf.argmax(label == self.labels)
        audio_binary = tf.io.read_file(file_path)
        audio, _ = tf.audio.decode_wav(audio_binary)
        audio = tf.squeeze(audio, axis=1)

        return audio, label_id

    def pad(self, audio):
        zero_padding = tf.zeros([self.sampling_rate] - tf.shape(audio), dtype=tf.float32)
        audio = tf.concat([audio, zero_padding], 0)
        audio.set_shape([self.sampling_rate])

        return audio

    def get_spectrogram(self, audio):
        stft = tf.signal.stft(audio, frame_length=self.frame_length,
                              frame_step=self.frame_step, fft_length=self.frame_length)
        spectrogram = tf.abs(stft)

        return spectrogram

    def get_mfccs(self, spectrogram):
        mel_spectrogram = tf.tensordot(spectrogram,
                                       self.linear_to_mel_weight_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[..., :self.num_coefficients]

        return mfccs

    def preprocess_with_stft(self, file_path):
        audio, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        spectrogram = tf.expand_dims(spectrogram, -1)
        spectrogram = tf.image.resize(spectrogram, [32, 32])

        return spectrogram, label

    def preprocess_with_mfcc(self, file_path):
        audio, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        mfccs = self.get_mfccs(spectrogram)
        mfccs = tf.expand_dims(mfccs, -1)

        return mfccs, label

    def make_dataset(self, files, train, batch_size):
        # files are list of file paths and train is a boolean (true for returning also labels)
        ds = tf.data.Dataset.from_tensor_slices(files)
        ds = ds.map(self.preprocess, num_parallel_calls=4)
        ds = ds.batch(batch_size)
        ds = ds.cache()
        if train is True:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)

        return ds


def gen_model(input_shape, scaling_factor,flag_MFCC, output_shape, choose="MLP"):
    inputs = keras.Input(shape=input_shape)
    if flag_MFCC == False:
        strides = [2, 2]
    else:
        strides = [2, 1]

    if choose == "CNN-2D":
        model = tf.keras.Sequential([tf.keras.layers.Conv2D(input_shape=input_shape, filters=int(128 * scaling_factor),
                                                            kernel_size=[3, 3], strides=strides, use_bias=False,
                                                            name='first_conv1d'),
                                     tf.keras.layers.BatchNormalization(momentum=0.1),
                                     tf.keras.layers.ReLU(),
                                     tf.keras.layers.Conv2D(input_shape=input_shape, filters=int(128 * scaling_factor),
                                                            kernel_size=[3, 3], strides=strides, use_bias=False,
                                                            name='second_conv1d'),
                                     tf.keras.layers.BatchNormalization(momentum=0.1),
                                     tf.keras.layers.ReLU(),
                                     tf.keras.layers.Conv2D(input_shape=input_shape, filters=int(128 * scaling_factor),
                                                            kernel_size=[3, 3], strides=strides, use_bias=False,
                                                            name='third_conv1d'),
                                     tf.keras.layers.BatchNormalization(momentum=0.1),
                                     tf.keras.layers.ReLU(),
                                     tf.keras.layers.GlobalAvgPool2D(),
                                     tf.keras.layers.Dense(output_shape, name='fc')])



    if choose == "DS-CNN":
        model = tf.keras.Sequential([tf.keras.layers.Conv2D(input_shape=input_shape, filters=int(256 * scaling_factor),
                                                            kernel_size=[3, 3], strides=strides, use_bias=False,
                                                            name='first_conv1d'),
                                     tf.keras.layers.BatchNormalization(momentum=0.1),
                                     tf.keras.layers.ReLU(),
                                     tf.keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1],
                                                                     use_bias=False),
                                     tf.keras.layers.Conv2D(input_shape=input_shape, filters=int(256 * scaling_factor),
                                                            kernel_size=[1, 1], strides=[1, 1], use_bias=False,
                                                            name='second_conv1d'),
                                     tf.keras.layers.BatchNormalization(momentum=0.1),
                                     tf.keras.layers.ReLU(),
                                     tf.keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1],
                                                                     use_bias=False, ),
                                     tf.keras.layers.Conv2D(input_shape=input_shape, filters=int(scaling_factor * 128),
                                                            kernel_size=[1, 1], strides=[1, 1], use_bias=False,
                                                            name='third_conv1d'),
                                     tf.keras.layers.BatchNormalization(momentum=0.1),
                                     tf.keras.layers.ReLU(),
                                     tf.keras.layers.GlobalAvgPool2D(),
                                     tf.keras.layers.Dense(output_shape, name='fc')])


    return model


def compile_and_train(model, train_ds, learning_rate, n_epochs=20):
    # then we can compile and print the summary of the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
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


def accuracy(pred, true):
    return sum(1 for x,y in zip(true,pred) if x == y) / float(len(true))


def pruning(saving_path,train_ds, batch_size, epochs=20, final_sparsity=0.8):
    model_new = tf.keras.models.load_model(filepath=saving_path)
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    # model_for_pruning = prune_low_magnitude(model_new)

    pruning_params = {
        "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.3, final_sparsity=final_sparsity, begin_step=len(train_ds) * 5,
            end_step=len(train_ds) * 15)
    }

    model_for_pruning = prune_low_magnitude(model_new, **pruning_params)
    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep()]

    # batch_size = 32
    input_width = 6

    input_shape = [batch_size, input_width, 2]
    model_for_pruning.build(input_shape)

    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    model_for_pruning.compile(
        optimizer="adam",
        loss=loss,
        metrics=metrics)
    model_for_pruning.fit(train_ds,
                          batch_size=batch_size, epochs=epochs, callbacks=callbacks)
    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

    print(model_for_pruning.summary())
    logdir = tempfile.mkdtemp()

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
        my_input = np.array(data[0], dtype=np.float32)
        label = np.array(data[1], dtype=np.int32)
        labels.append(label[0])
        interpreter.set_tensor(input_details[0]["index"], my_input)
        interpreter.invoke()
        my_output = interpreter.get_tensor(output_details[0]["index"])
        my_output = np.argmax(np.squeeze(np.array(my_output)))
        outputs.append(my_output)

    comput_accuracy = accuracy(outputs, labels)

    return comput_accuracy


def save_zip_model(tf_lite_model,type_model ):
    tf_lite_model_dir = f"Group18_kws_{type_model}.tflite.zlib "
    with open (tf_lite_model_dir, "wb") as f:
        tf_lite_compressed = zlib.compress(tf_lite_model)
        f.write(tf_lite_compressed)
    return os.path.getsize(tf_lite_model_dir)


def save_TF_Lite_quant(path, type_model):
    converter = tflite.TFLiteConverter.from_keras_model(path)
    pruned_tflite_model = converter.convert()

    # _, pruned_tflite_file = tempfile.mkstemp('.tflite')

    with open('DS-CNN_C_pruned.tflite', 'wb') as f:
        f.write(pruned_tflite_model)

    # print('Saved pruned TFLite model to:', pruned_tflite_file)
    convert = tflite.TFLiteConverter.from_keras_model(path)
    converter.optimizations = [tflite.Optimize.DEFAULT]
    quantized_and_pruned_tflite_model = converter.convert()

    path_save = 'DS-CNN_C_quantized_and_pruned.tflite'
    # _,quantized_and_pruned_tflite_file = tempfile.mkstemp(".tflite")
    with open(path_save, 'wb') as f:
        f.write(quantized_and_pruned_tflite_model)
    # quantized_and_pruned_tflite_file_path='MLP_quantized_and_pruned.tflite'
    # print('Saved quantized and pruned TFLite model to:', quantized_and_pruned_tflite_file)

    # print("Size of gzipped baseline Keras model: %.2f bytes" % (get_gzipped_model_size(saving_path)))
    print("Size of gzipped pruned and quantized TFlite model: %.2f bytes" % (
        save_zip_model(quantized_and_pruned_tflite_model, type_model)))
    return path_save


def main(args):
    type_model = args.version
    # set random seed to get always same value
    seed = 42
    tf.random.set_seed(seed)

    # define parameters for the signal generator
    sampling_rate = 16000
    frame_length = 256
    frame_step = 128
    freq_mel_high = 4000
    freq_mel_low = 20
    num_MFCC = 10
    flag_MFCC = True

    if type_model == "a":
        mel_bins = 20
        # training
        batch_size = 64
        learning_rate_training = 0.02
        chosen_model = "DS-CNN"
        scaling_factor = 0.9
        n_epochs_train = 30
        # pruning
        n_epochs_pruning = 30
        final_sparsity = 0.8

    if type_model == "b":
        mel_bins = 16
        # training
        batch_size = 32
        learning_rate_training = 0.03
        chosen_model = "DS-CNN"
        scaling_factor = 0.8
        n_epochs_train = 30
        # pruning
        n_epochs_pruning = 30
        final_sparsity = 0.8

    if type_model == "c":
        mel_bins = 16
        # training
        batch_size = 32
        learning_rate_training = 0.03
        chosen_model = "DS-CNN"
        scaling_factor = 0.5
        n_epochs_train = 30
        # pruning
        n_epochs_pruning = 30
        final_sparsity = 0.8

    #generate the dataset
    path, commands = get_dataset()
    train_paths, val_paths, test_paths = create_data_partition()

    # create the signal generator
    SG = SignalGenerator(commands, sampling_rate, frame_length, frame_step, mel_bins, freq_mel_low, freq_mel_high, num_MFCC, flag_MFCC)

    # create the various dataset split
    train_files = tf.convert_to_tensor(train_paths)
    val_files = tf.convert_to_tensor(val_paths)
    test_files = tf.convert_to_tensor(test_paths)

    train_ds = SG.make_dataset(train_files, True, batch_size)
    val_ds = SG.make_dataset(val_files, False, batch_size)
    test_ds = SG.make_dataset(test_files, False, batch_size)

    # generate the model
    inp = (124, 10, 1)
    output_shape = 8
    model = gen_model(inp, scaling_factor, flag_MFCC, output_shape, chosen_model)

    #train the model
    trained_model, summary, history = compile_and_train(model, train_ds, learning_rate_training, n_epochs_train)
    saving_path = saving_model(trained_model)

    #prune the model
    pruned_model = pruning(saving_path,train_ds, batch_size, n_epochs_pruning, final_sparsity)

    #quantize and save the model
    path_saved_model = save_TF_Lite_quant(pruned_model, type_model)

    #evaluate the model
    accuracy_eval = evaluate_TF_LITE(path_saved_model, test_ds)
    os.remove("DS-CNN_C_pruned.tflite")
    os.remove("DS-CNN_C_quantized_and_pruned.tflite")
    print(f"Accuracy: {accuracy_eval}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str,
                        choices=['a', 'b','c'], required=True,
                        help='Which version of the model run')
    args = parser.parse_args()
    main(args)