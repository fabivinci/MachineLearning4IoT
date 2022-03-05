import base64
import argparse


def convert_TFLITE_string(model_path, saving=False):
    with open(model_path, "rb") as fp:
        model_string = fp.read()
    model_base64 = base64.b64encode(model_string)
    if saving == True:
        path = "string_model.txt"
        with open(path, "wb") as file:
            file.write(model_base64)
        return path
    return model_base64


def convert_string_TFLITE(string, saving=False, name=None):
    model = base64.b64decode(string)

    if saving:
        path = f"{name}.tflite"
        with open(path, "wb") as file:
            file.write(model)
        return path
    else:
        return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model', type=str,
                        help='tflite model path')
    parser.add_argument('--original', type=str,
                        help='if tflite convert in string, if string opposite')

    args = parser.parse_args()
    model_path = args.model
    original = args.original
    if original == "tflite":
        convert_TFLITE_string(model_path)
    elif original == "string":
        convert_string_TFLITE(model_path)