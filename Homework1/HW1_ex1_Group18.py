import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
import os
import struct


def build_dataset(filename, output_path, normalize=False):
    header = ["time", "temperature", "humidity"]
    data = pd.read_csv(filename, names=header)
    data = data[:4]

    if normalize == True:
        max_temp = 50
        min_temp = 0
        max_hum = 90
        min_hum = 20
        data["temperature"] = (data["temperature"] - min_temp) / (max_temp - min_temp)
        data["humidity"] = (data["humidity"] - min_hum) / (max_hum - min_hum)
    with tf.io.TFRecordWriter(output_path) as writer:
        for index, row in data.iterrows():
            measure = row.T
            posix = int(time.mktime((measure[0]).timetuple()))
            #date_time_obj = datetime.strptime(measure[0], "%d/%m/%Y %H:%M:%S")
            #unix_time = date_time_obj.timestamp()
            #unix_time_bytes = int(unix_time).to_bytes(4, "big")
            time_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[posix]))

            if normalize == True:
                #with normalization
                #temp_bytes = struct.pack('<f', measure[1])  # little-endian
                #hum_bytes = struct.pack('<f', measure[2])  # little-endian
                temperature_feature = tf.train.Feature(float_list=tf.train.FloatList(value=[measure[1]]))
                umidity_feature = tf.train.Feature(float_list=tf.train.FloatList(value=[measure[2]]))

            elif normalize == False:
                #temp_bytes = measure[1].to_bytes(1, "big")
                #hum_bytes = measure[2].to_bytes(1, "big")
                temperature_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[measure[1]]))
                umidity_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[measure[2]]))


            mapping = {"time": time_feature, \
                       "temperature": temperature_feature, \
                       "humidity": umidity_feature}
            example = tf.train.Example(features=tf.train.Features(feature=mapping))
            writer.write(example.SerializeToString())
    final_size = os.path.getsize(output_path) / 2. ** 10
    print(f" {final_size*1000}B")


#take as input the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--normalize", type=bool, default=False, required=False, help = "Normalization flag, boolean, False default, optional")
parser.add_argument("--input", type=str ,  required=True, help = "Input path")
parser.add_argument("--output", type=str ,  required=True, help = "Output path")


args = parser.parse_args()
norm_flag = args.normalize
file = args.input
output = args.output

build_dataset(filename = file, output_path = output, normalize = norm_flag)
