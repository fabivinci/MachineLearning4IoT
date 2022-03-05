import tensorflow as tf
import tensorflow.lite as tflite
import time
from scipy import signal
from DoSomething import DoSomething
import WAV_enc_dec as wav
import json


class SubscriberSlow(DoSomething):
    def __init__(self, clientID):
        super().__init__(clientID)
        self.publisher = DoSomething("Slow Publisher")
        self.publisher.run()

    def notify(self, topic, msg):
        msg_string = msg.decode("utf-8")
        msg_json = json.loads(msg_string)
        file_audio_string = msg_json["e"][0]["v"]
        timecode = str(msg_json["bt"])
        saving_path = wav.audio_from_senML(file_audio_string, "file_audio")
        answer_str = predict(saving_path, timecode)

        self.publisher.myMqttClient.myPublish("/282383/SlowAudio", answer_str)


def predict(path_audio, timecode):
    sampling_rate = 16000
    sampling_ratio = 1
    frame_length = 40 * 16
    frame_step = 20 * 16
    freq_mel_high = 4000
    freq_mel_low = 20
    num_mfcc = 10
    mel_bins = 40
    model_path = "kws_dscnn_True.tflite"
    commands = ["stop", "up", "yes", "right", "left", "no", "down", "go"]

    Preprocesser = preprocess(commands, sampling_rate, frame_length, frame_step, mel_bins, freq_mel_low, freq_mel_high,
                              num_mfcc, int(sampling_rate * sampling_ratio))
    audio_binary = tf.io.read_file(path_audio)
    mfcc = Preprocesser.preprocess_with_mfcc(audio_binary)
    pred_label = int(get_label(model_path, mfcc))
    answer = {
        "bn": f"slow_service",
        "bt": timecode,
        "e": [
            {"n": "label", "u": "/", "t": 0, "v": pred_label}
        ]
    }
    answer_str = json.dumps(answer)

    return answer_str


def get_label(model_path, mfcc_input):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], mfcc_input)
    interpreter.invoke()
    my_output = interpreter.get_tensor(output_details[0]["index"])
    fast_prob = tf.nn.softmax(my_output)
    fast_pred = tf.argmax(fast_prob, axis=1)

    return fast_pred


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
    seed = 42
    tf.random.set_seed(seed)

    subscriber = SubscriberSlow("Subscriber Slow")
    subscriber.run()
    subscriber.myMqttClient.mySubscribe("/282383/FastAudio")

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()