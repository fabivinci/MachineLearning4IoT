import requests
import convert_tflite
import json

def main():
    #ADD MODELS
    url = 'http://raspberrypi.local:8080'

    model_names = ['CNN.tflite', 'MLP.tflite']
    for model in model_names:
        model_path = model
        model_name = model.split('.')[0]

        model_base64 = convert_tflite.convert_TFLITE_string(model_path)
        body = {'model': str(model_base64.decode("utf-8")), 'name': model_name}
        body = json.dumps(body)
        r = requests.post(url + "/add", body)

        if r.status_code == 200:
            print(r.text)

        else:
            print('Error:', r.status_code)

    #LIST THE MODELS FOUND
    r = requests.get(url + "/list")
    if r.status_code==200:
        print(r.text)
    else:
        print("Error", r.status_code)


    #PREDICT WITH CNN
    model_name = 'CNN'
    tthres=0.1
    hthres=0.2

    parameters = {"model":model_name,"tthres":tthres, "hthres":hthres}
    r = requests.get(url + "/predict", params = parameters)

if __name__ == "__main__":
    main()
