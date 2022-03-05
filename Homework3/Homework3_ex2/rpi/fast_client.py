import numpy as np
import os
import tensorflow as tf
import tensorflow.lite as tflite
import time
from scipy import signal
from DoSomething import DoSomething
import WAV_enc_dec as wav
import datetime
import sys
import json

class SubScriberFast(DoSomething):
    def notify(self, topic, msg):
        input_json = msg.decode("utf-8")
        input_json = json.loads(input_json)
        size = sys.getsizeof(input_json)
        timecode = input_json["bt"]
        label = input_json["e"][0]["v"]
        pred_slow_string = str(timecode) +" , " + str(label) + " , " + str(size)
        with open("slow_predictions.txt", "w") as f:
            f.write(pred_slow_string)


def get_dataset():
    path = tf.keras.utils.get_file(\
        origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip", \
        fname="mini_speech_commands.zip", \
        extract=True, \
        cache_dir=".", cache_subdir="data")

    data_dir, _ = os.path.splitext(path)
    commands = ["stop","up","yes","right","left","no","down","go"]
    return path, commands


def create_data_partition():
    test_paths = list(np.loadtxt("kws_test_split.txt", dtype = "str"))
    return  test_paths


def accuracy(pred, true):
    
    return sum(1 for x,y in zip(true,pred) if x == y) / float(len(true))


def evaluate_TF_LITE(model_path, test_paths, mel, low, high, sampling_ratio, commands, frame_lenght, frame_step, threshold, sampling_rate, num_mfcc):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    outputs = []
    labels = []
    tot_time = []
    publisher = DoSomething("Publisher Fast")
    publisher.run()
    subscriber = SubScriberFast("Subscriber Fast")
    subscriber.run()
    subscriber.myMqttClient.mySubscribe("/282383/SlowAudio")
    Preprocesser = preprocess(commands,sampling_rate, frame_lenght, frame_step, mel, low, high,num_mfcc, int(sampling_rate * sampling_ratio) )
    tot_size = 0
    for data in test_paths:
        label = np.array(data.split("/")[3])
        label = commands.index(str(label))
        labels.append(label)
        #start measuring the time
        start = time.time()
        audio_binary = tf.io.read_file(data)
        mfcc_input = Preprocesser.preprocess_with_mfcc(audio_binary)
        interpreter.set_tensor(input_details[0]["index"], mfcc_input)
        interpreter.invoke()
        my_output = interpreter.get_tensor(output_details[0]["index"])
        #end measuring the time
        end = time.time()
        fast_prob = tf.nn.softmax(my_output)
        fast_pred = tf.argmax(fast_prob, axis=1)
        output, size_delta = success_seeker_policy(fast_pred,fast_prob, threshold , data, publisher)
        tot_size += size_delta
        outputs.append(output)
        tot_time.append(int((end-start)*1000))

    publisher.end()
    comput_accuracy = accuracy(outputs, labels)
    avg_time = np.mean(tot_time)
    print(f"Collaborative Accuracy: {comput_accuracy*100} %  \nAVG Fast Total Inference Time: {avg_time} ms \nCommunication Cost: {float(tot_size / 1000000)} MB")
    return comput_accuracy


def success_seeker_policy(index, vector, threshold, path, publisher):
    if np.max(vector) < threshold:
        audio_string = wav.audio_to_senML(path)
        timecode = int(datetime.datetime.now().timestamp())
        request = {
            "bn": f"fast_service",
            "bt": timecode,
            "e": [
                {"n": "a", "u": "/", "t": 0, "v": audio_string}
            ]
        }
        request_str = json.dumps(request)
        size_delta = sys.getsizeof(request_str)
        publisher.myMqttClient.myPublish("/282383/FastAudio", request_str)
        waiting = 0
        while waiting < 2:
            time.sleep(1)
            waiting += 1
        #after waiting we can try to read the file in order to get the predicted slow
        with open("slow_predictions.txt") as f:
            line = f.readlines()
        timecode_received = int(line[0].split(",")[0])

        
        if timecode_received == timecode:
            size_delta += int(line[0].split(",")[2])
            return int(line[0].split(",")[1]), size_delta
        else:
            return np.argmax(vector), size_delta
    else:
        size_delta = 0
        return index, size_delta



class preprocess:
    def __init__(self, labels, sampling_rate, frame_length, frame_step,
                 num_mel_bins=None, lower_frequency=None, upper_frequency=None,
                 num_coefficients=None, resampling_rate=None):

        self.labels = labels

        self.sampling_rate = sampling_rate
        self.resampling_rate = resampling_rate

        self.frame_length = frame_length
        self.frame_step = frame_step

        self.lower_frequency = lower_frequency
        self.upper_frequency = upper_frequency

        self.num_mel_bins = num_mel_bins
        self.num_coefficients = num_coefficients

        num_spectrogram_bins = (frame_length) // 2 + 1

        if self.resampling_rate is not None:
            rate = self.resampling_rate

        else:
            rate = self.sampling_rate

        self.num_frames = (rate - self.frame_length) // self.frame_step + 1

        self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            self.num_mel_bins, num_spectrogram_bins, rate,
            self.lower_frequency, self.upper_frequency)
        self.preprocess = self.preprocess_with_mfcc

    def custom_resampling(self, audio):
        audio = signal.resample_poly(
            audio, 1, self.sampling_rate // self.resampling_rate)
        audio = tf.convert_to_tensor(audio, dtype=tf.float32)
        return audio

    def read(self, audio_bytes):
        audio, _ = tf.audio.decode_wav(audio_bytes)
        audio = tf.squeeze(audio, axis=1)

        if self.resampling_rate is not None:
            audio = tf.numpy_function(
                self.custom_resampling, [audio], tf.float32)

        return audio

    def pad(self, audio):
        if self.resampling_rate is not None:
            rate = self.resampling_rate
        else:
            rate = self.sampling_rate
        zero_padding = tf.zeros([rate] - tf.shape(audio), dtype=tf.float32)
        audio = tf.concat([audio, zero_padding], 0)
        audio.set_shape([rate])

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

    def preprocess_with_mfcc(self, audio_bytes):
        audio = self.read(audio_bytes)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        mfccs = self.get_mfccs(spectrogram)
        # Reshaping since only 1 audio at time si given for inference
        mfccs = tf.reshape(
            mfccs, [1, self.num_frames, self.num_coefficients, 1])

        return mfccs


def main():
    #define parameter
    sampling_rate = 16000
    sampling_ratio = 0.5
    frame_length =  int(40*(sampling_rate/1000)*sampling_ratio)
    frame_step = int(20*(sampling_rate/1000)*sampling_ratio)
    freq_mel_high = 4000
    freq_mel_low = 20
    num_MFCC = 10
    mel_bins = 16
    threshold = 0.45

    seed = 42
    tf.random.set_seed(seed)

    path, commands = get_dataset()
    test_paths = create_data_partition()

    model_path = "kws_dscnn_True.tflite"
    accuracy_eval = evaluate_TF_LITE(model_path, test_paths, mel_bins, freq_mel_low, freq_mel_high, sampling_ratio,commands, frame_length, frame_step, threshold, sampling_rate, num_MFCC)

if __name__ == "__main__":
    main()
