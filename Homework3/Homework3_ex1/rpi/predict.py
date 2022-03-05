import tensorflow as tf

def prediction(interpreter, input_details, output_details, input_data):

    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()
    my_output = interpreter.get_tensor(output_details[0]['index'])
    return my_output[0][0]

def instantiate_interpreter(model_name):
    interpreter = tf.lite.Interpreter(model_path=f'./models/{model_name}.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


