from time import process_time, time
from statistics import mean
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras.backend as K
from sklearn.metrics import accuracy_score
from datetime import datetime
import pandas as pd
import signal


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
couche1=64
couche2=128
dense=512
iteration=50
timeout_training=900
timeout_predict=120
method_name='Format des inputs/layers'
model_name='CNN'

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)
        
def GetModel(layers_type=None):
    if layers_type==None:
        pass
    else:
        tf.keras.backend.set_floatx(layers_type)
    model = tf.keras.Sequential([   
        tf.keras.Input(shape=(32, 32, 3),dtype=layers_type),
        tf.keras.layers.Conv2D(couche1, (3, 3), strides=(2, 2), padding="same",dtype=layers_type),
        tf.keras.layers.LeakyReLU(alpha=0.2,dtype=layers_type),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same",dtype=layers_type),
        tf.keras.layers.Conv2D(couche2, (3, 3), strides=(2, 2), padding="same",dtype=layers_type),
        tf.keras.layers.Flatten(dtype=layers_type),
        tf.keras.layers.Dense(dense,activation='relu',dtype=layers_type),
        tf.keras.layers.Dense(100,dtype=layers_type)])
    return model 

def GetParametersNumber(model):
    trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])
    return trainable_count+non_trainable_count

def GetTime(model_name,inputs_type,layers_type,iteration,time_out):
    x_train_type = x_train.astype(inputs_type)
    x_test_type = x_test.astype(inputs_type)
    
    start_cpu_t,start_wall_t=process_time(),time()
    model=GetModel(layers_type=layers_type)

    model.compile(optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],)
    try:
        with timeout(seconds=timeout_training):

            nb_params=GetParametersNumber(model)
            date=datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            parameters={'inputs_type':inputs_type,'layers_type':layers_type}
            model.fit(x_train, y_train, epochs=3,verbose=0)
            stop_cpu_t,stop_wall_t=process_time(),time()
            cpu_time=stop_cpu_t-start_cpu_t
            wall_time=stop_wall_t-start_cpu_t
            temps_cpu=[]
            temps_wall=[]
            accuracys=[]
            with timeout(seconds=time_out):
                try:
                    try:
                        for k in range(iteration):
                            start_cpu,start_wall=process_time(),time()
                            y_pred=model.predict(x_test)
                            stop_cpu,stop_wall=process_time(),time()
                            temps_cpu.append(stop_cpu-start_cpu)
                            temps_wall.append(stop_wall-start_wall)
                            accuracys.append(accuracy_score(np.argmax(y_pred,1),y_test))
                        tf.keras.backend.set_floatx(default_value)
                        return mean(temps_cpu),mean(temps_wall),mean(accuracys),date,parameters,nb_params,cpu_time,wall_time
                    except Exception as e:
                        tf.keras.backend.set_floatx(default_value)
                        return mean(temps_cpu),mean(temps_wall),mean(accuracys),date,parameters,nb_params,cpu_time,wall_time
                except:
                    return mean(temps_cpu),mean(temps_wall),mean(accuracys),date,parameters,nb_params,cpu_time,wall_time
    except:
                    return 'timeout','timeout','timeout',date,parameters,nb_params,'timeout','timeout'

def SendFormatResults():       

    layers_types=['float32','float64']
    inputs_types=['int8','int16','int32','float16','float32']

    default_value=tf.keras.backend.floatx()
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

    result=pd.DataFrame(columns=['Modèle','Nb(paramètres)','Date','Méthode','Paramètres','CPU + Sys time','Wall Time','Précision','Training time(cpu)','Training time(wall)'])
    for inputs_type in inputs_types:
        for layers_type in layers_types:
            time_cpu,time_wall,accuracy,date,parameters,nb_params,cpu_time,wall_time=GetTime(model_name,inputs_type,layers_type,iteration=iteration,time_out=timeout_predict)
            result=result.append({'Modèle':model_name,'CPU + Sys time':time_cpu,'Wall time':time_wall,'Précision':accuracy,'Date':date,'Méthode':method_name,'Paramètres':parameters,'Nb(paramètres)':nb_params,'Training time(cpu)':cpu_time,'Training time(wall)':wall_time}, ignore_index=True)
            print('Modèle: ',model_name,'Input: ',inputs_type, '&', 'Layers: ',layers_type,'--> time_cpu:',time_cpu,'->time_wall:',time_wall,'-> accuracy:',accuracy)
    try:    
        results=pd.read_csv('/home/arnaudhureaux/deeplearning-cpu-optimization/outputs/results.csv')
        results=pd.concat((results,result),axis=0).reset_index(drop=True)
        results.to_csv('/home/arnaudhureaux/deeplearning-cpu-optimization/outputs/results.csv',index=False,header=True,encoding='utf-8-sig')
    except:
        result.to_csv('/home/arnaudhureaux/deeplearning-cpu-optimization/outputs/results.csv',index=False,header=True,encoding='utf-8-sig')
        
SendFormatResults()