from datetime import datetime
import time



def measure(dht_device):
  temperature = None
  humidity = None
  flag = True
  while flag:
    #to handle the sensor errors
    try:
      temperature = dht_device.temperature
      humidity = dht_device.humidity
      i = 6
      flag = False
    except:
      time.sleep(1)
      continue

  now = datetime.now()
  dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

  res = [dt_string,temperature,humidity]
  return  temperature, humidity,dt_string

