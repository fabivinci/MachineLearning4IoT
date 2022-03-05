import base64

def audio_to_senML(audio):
    with open(audio, 'rb') as fd:
        wav_file = fd.read()
    #convert from bytes to string
    wav_base64 = base64.b64encode(wav_file)
    audio_string = wav_base64.decode('utf-8')
    return audio_string

def audio_from_senML(encoded_audio, name):
    bytes_audio = base64.b64decode(encoded_audio)
    path = f"{name}.wav"
    with open(path, "wb") as file:
        file.write(bytes_audio)
    return path
