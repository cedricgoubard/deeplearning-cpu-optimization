from time import process_time, time
import tempfile
import os
from sklearn.metrics import accuracy_score
import tensorflow as tf
import numpy as np
from tensorflow import keras
import tensorflow_model_optimization as tfmot
import pandas as pd
from datetime import datetime
import tensorflow.keras.backend as K
from statistics import mean

def GetSimpleModel(couche1=16,couche2=32,dense=512):
    model = tf.keras.Sequential([
      keras.layers.InputLayer(input_shape=(32, 32,3)),
      keras.layers.Conv2D(couche1, (3, 3), strides=(2, 2), padding="same"),
      keras.layers.LeakyReLU(alpha=0.2),
      keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
      keras.layers.Conv2D(couche2, (3, 3), strides=(2, 2), padding="same"),
      keras.layers.Flatten(),
      keras.layers.Dense(dense, activation='relu'),
      keras.layers.Dense(100),
    ])
    model._name='baseline'
    model.compile(optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],)

    model.fit(x_train, y_train, epochs=3,verbose=0)
    return model

def GetTFLmodel(model):
    tf_lite_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tf_lite_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = tf_lite_converter.convert()
    tflite_model_name = 'TFlite_post_quantModel8bit'
    open(tflite_model_name, "wb").write(tflite_model)

    interpreter = tf.lite.Interpreter(model_path = tflite_model_name)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details() #1
    output_details = interpreter.get_output_details() #16
    return interpreter

def GetParametersNumber(model):
    trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])
    return trainable_count+non_trainable_count

def GetTime(model,parameters):
    nb_params=GetParametersNumber(model)
    predictions=[]
    temps_cpu=[]
    temps_wall=[]
    date=datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    for k in range(nb_test):
        start_cpu,start_wall=process_time(),time()
        pred=model.predict(x_test[k].reshape(1,32,32,3))
        stop_cpu,stop_wall=process_time(),time()
        temps_cpu.append(stop_cpu-start_cpu)
        temps_wall.append(stop_wall-start_wall)
        predictions.append(np.argmax(pred))
    accuracy=accuracy_score(np.array(predictions),y_test[0:nb_test][:,0])
    return mean(temps_cpu),mean(temps_wall),accuracy,date,parameters,nb_params

def GetTFLtime(interpreter,parameters):
    nb_params=None
    date=datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    pred = []
    temps_cpu =[]
    temps_wall=[]
    for i in range(nb_test):  
        start_cpu,start_wall= process_time(),time()

        inp = X_test_numpy[i]
        inp = inp.reshape(1 ,32, 32,3)
        interpreter.set_tensor(0,inp )
        interpreter.invoke()
        tflite_model_predictions = interpreter.get_tensor(16)
        prediction_classes = np.argmax(tflite_model_predictions, axis=1)
        pred.append(prediction_classes[0])

        stop_cpu,stop_wall=process_time(),time()

        temps_wall.append(stop_wall-start_wall)
        temps_cpu.append(stop_cpu-start_cpu)
    accuracy=accuracy_score(np.array(pred),y_test[0:nb_test][:,0])
    return mean(temps_cpu),mean(temps_wall),accuracy,date,parameters,nb_params

def SendData(result,time_cpu,time_wall,accuracy,date,parameters,nb_params):
    result=result.append({'Modèle':model_name,'CPU + Sys time':time_cpu,'Wall Time':time_wall,'Précision':accuracy,'Date':date,'Méthode':method_name,'Paramètres':parameters,'Nb(paramètres)':nb_params}, ignore_index=True)
    return result

nb_test=10000
couche1=64
couche2=128
dense=512
model_name='CNN'
method_name='TFLite'

cifar100 = tf.keras.datasets.cifar100
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

X_test_numpy = np.array(x_test, dtype=np.float32)
y_test_numpy =np.array(y_test, dtype=np.float32)

result=pd.DataFrame(columns=['Modèle','Nb(paramètres)','Date','Méthode','Paramètres','CPU + Sys time','Précision','Wall Time'])

parameters='Baseline'
model=GetSimpleModel()
time_cpu,time_wall,accuracy,date,parameters,nb_params=GetTime(model,parameters)
result=SendData(result,time_cpu,time_wall,accuracy,date,parameters,nb_params)
print('Méthode: ',method_name,'Modèle: ',model_name,'Paramètre: ',parameters,'-> accuracy:',accuracy,'-> time_cpu:',time_cpu,'->time_wall:',time_wall)

parameters='TFlite'
intepreter=GetTFLmodel(model)
time_cpu,time_wall,accuracy,date,parameters,nb_params=GetTFLtime(intepreter,parameters)
result=SendData(result,time_cpu,time_wall,accuracy,date,parameters,nb_params)
print('Méthode: ',method_name,'Modèle: ',model_name,'Paramètre: ',parameters,'-> accuracy:',accuracy,'-> time_cpu:',time_cpu,'->time_wall:',time_wall)

try:    
    results=pd.read_csv('/home/arnaudhureaux/deeplearning-cpu-optimization/outputs/results.csv')
    results=pd.concat((results,result),axis=0).reset_index(drop=True)
    results.to_csv('/home/arnaudhureaux/deeplearning-cpu-optimization/outputs/results.csv',index=False,header=True,encoding='utf-8-sig')
except:
    result.to_csv('/home/arnaudhureaux/deeplearning-cpu-optimization/outputs/results.csv',index=False,header=True,encoding='utf-8-sig')