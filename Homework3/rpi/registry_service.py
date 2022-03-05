import cherrypy
import json
import os
import convert_tflite
import measure_DHT as DHT
import adafruit_dht
from board import D4
import threading
import numpy as np
import predict as PRED 
import tensorflow as tf
from DoSomething import DoSomething
from datetime import datetime


# One class for each service
class AddModel(object):
    exposed = True
    def GET(self, *path, **query):
        pass

    def POST(self, *path, **query):

        # path is a list: contains all the elements of the request path
        # query dictionary contains all the parameters of the request

        saving_path = "models"
        body = cherrypy.request.body.read()
        body = json.loads(body)
        model = body.get('model')
        if model is None:
            raise cherrypy.HTTPError(400, "empty model")
        model_name = body.get('name')
        if model_name is None:
            raise cherrypy.HTTPError(400, "no name provided")

        if not os.path.exists(saving_path):
            # create the directory
            os.makedirs(saving_path)

        # save model
        final_path = os.path.join(saving_path, model_name)
        final_file = open(f'{final_path}.tflite', 'wb')
        bytes_model = convert_tflite.convert_string_TFLITE(model)
        final_file.write(bytes_model)
        final_file.close()
        cherrypy.response.headers["Status"] = "200"
        return f"Model {model_name} saved correctly"


    def PUT(self, *path, **query):
        pass

    def DELETE(self, *path, **query):
        pass

class ListModels(object):
    exposed = True
    def GET(self, *path, **query):
        saving_path = "models"
        output_body = {}

        #exposed = True
        filenames = next(os.walk(saving_path), (None, None, []))[2]  # [] if no file

        final_names = []
        for f in filenames:
            new_f = f.split(".")[0]
            final_names.append(new_f)
        output_body["models"] = final_names
        cherrypy.response.headers["Status"] = "200"

        return json.dumps(output_body)

    def POST(self, *path, **query):
        pass

    def PUT(self, *path, **query):
        pass

    def DELETE(self, *path, **query):
        pass

class Predict(object):
    exposed=True
    def GET(self, *path, **query):

        publisher = DoSomething("publisher 1")  # this create the publisher 1
        publisher.run()

        dht_device = adafruit_dht.DHT11(D4)
        registered_values = []
        ticker = threading.Event()
        WAIT_TIME_SECONDS = 1
        count = 0
        delta = 10
        model_name = query.get('model')
        tthres = float(query.get('tthres'))
        hthres = float(query.get('hthres'))

        interpreter, input_details, output_details = PRED.instantiate_interpreter(model_name)
        mean_train = np.array([9.107597, 75.904076], dtype=np.float32)
        std_train = np.array([8.654227, 16.557089], dtype=np.float32)
        predictions = []
        ground_truth = []

        while not ticker.wait(WAIT_TIME_SECONDS) and count < delta:

            mes_temp, mes_hum,_ = DHT.measure(dht_device)
            now=datetime.now()
            date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
            to_app = [mes_temp, mes_hum]
            registered_values.append(to_app)
            ground_truth.append(to_app)
            if len(registered_values) >= 6:
                last_six = np.array(registered_values[-6:])
                normalized = (last_six - mean_train) / (std_train + 1.e-6)
                pre_model = np.array([list(normalized)], dtype=np.float32)
                pre_model = tf.convert_to_tensor(pre_model, dtype=np.float32)
                
                pred_value = PRED.prediction(interpreter, input_details, output_details, pre_model)
                
                predictions.append(pred_value)
            if len(registered_values) >= 7:
                
                # compute the error
                error_temp = abs(predictions[0][0] - ground_truth[-1][0])
                error_hum = abs(predictions[0][1] - ground_truth[-1][1])
                

                # pack data into SENML+JS0N string
                SENML_body = {
                    "bn": "raspberrypi.local",  # unique identifier
                    "bt": date_time  # timestamp variable
                     }

                if error_temp > tthres:
                    # create MQTT publish with topic Temperature
                    SENML_body['e'] = [{"n": "predicted_temperature", "u": "Cel", "t": 0, "v": float(predictions[0][0])},
                        {"n": "true_temperature", "u": "Cel", "t": 0, "v": float(ground_truth[-1][0])}]
                    SENML_body_temp = json.dumps(SENML_body)
                    publisher.myMqttClient.myPublish("/279918/TemperatureAlert", SENML_body_temp)  # topic e content
                    
                if error_hum > hthres:
                    SENML_body['e']= [{"n": "predicted_humidity", "u": "%RH", "t": 0, "v": float(predictions[0][1])},
                        {"n": "true_humidity", "u": "%RH", "t": 0, "v": float(ground_truth[-1][1])}]
                    SENML_body_hum = json.dumps(SENML_body)
                    # create MQTT publish with topic Humidity
                    publisher.myMqttClient.myPublish("/279918/HumidityAlert", SENML_body_hum)  # topic e content

        publisher.end()

    def POST(self, *path, **query):
        pass

    def PUT(self, *path, **query):
        pass

    def DELETE(self, *path, **query):
        pass


if __name__ == "__main__":
    # copy paste this part to create a REST every time
    
    conf = {"/": {"request.dispatch": cherrypy.dispatch.MethodDispatcher()}}
    cherrypy.tree.mount(AddModel(), "/add", conf)
    cherrypy.tree.mount(ListModels(), "/list", conf)
    cherrypy.tree.mount(Predict(), "/predict", conf)
    cherrypy.config.update({"server.socket_host": "0.0.0.0"})  # to expose the service to every IP
    cherrypy.config.update({"server.socket_port": 8080})
    cherrypy.engine.start()
    cherrypy.engine.block()
