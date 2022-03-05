import time
import json
from DoSomething import DoSomething


class Subscriber(DoSomething):

    def __init__(self, clientID):
        DoSomething.__init__(self, clientID)

    def notify(self, topic, msg):
        input_json = msg.decode('UTF-8')

        input_json = json.loads(input_json)

        #extract all the information form the json
        events = input_json['e'] #remeber that e is a list
        prediction = None
        ground_truth = None
        temperature = False

        for event in events:

            if event['n'].split('_')[0] =='predicted':
                prediction = event['v']


            elif event['n'].split('_')[0]=='true':
                ground_truth = event['v']


            if event['n'].split('_')[1] == 'humidity':
                t_unit = event['u']
                temperature=False

            if event['n'].split('_')[1] == 'temperature':
                t_unit = event['u']
                temperature = True

            timestamp = input_json['bt']


        if temperature == True:

            print(
                f'\n{timestamp} Temperature Alert Actual Temperature={round(prediction,2)}{t_unit} Prediticted Temperature={round(ground_truth,2)}{t_unit}')


        else:

            print(f'\n{timestamp} Humidity Alert Actual Humidity={round(prediction,2)}{t_unit} Prediticted Humidity={round(ground_truth,2)}{t_unit}')

if __name__ == "__main__":
    test = Subscriber("subscriber 1")
    test.run()

    topic1="/279918/TemperatureAlert"
    topic2 ="/279918/HumidityAlert"
    test.myMqttClient.mySubscribe(topic1)
    test.myMqttClient.mySubscribe(topic2)
    while True:
        time.sleep(1)



