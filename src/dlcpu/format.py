from time import process_time, time
from statistics import mean
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from datetime import datetime
import pandas as pd
import signal
import os

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

def get_path_outputs():
    return os.path.dirname(os.path.abspath(__file__)).replace('/src/dlcpu','')+'/outputs'

def get_model(layers_type,couche1,couche2,dense):
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

def get_parameters_number(model):
    trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    return trainable_count+non_trainable_count

def get_time(inputs_type,layers_type,iteration,couche1,couche2,dense,train,pred):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    x_train_type = x_train.astype(inputs_type)
    x_test_type = x_test.astype(inputs_type)
    
    start_cpu_t,start_wall_t=process_time(),time()
    model=get_model(layers_type,couche1,couche2,dense)

    model.compile(optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],)
    try:
        with timeout(seconds=1200):
            with tf.device('/'+train+':0'): 
                nb_params=get_parameters_number(model)
                date=datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                parameters={'inputs_type':inputs_type,'layers_type':layers_type}
                model.fit(x_train, y_train, epochs=3,verbose=0)
                stop_cpu_t,stop_wall_t=process_time(),time()
                cpu_time=stop_cpu_t-start_cpu_t
                wall_time=stop_wall_t-start_cpu_t
                temps_cpu=[]
                temps_wall=[]
                accuracys=[]
            with tf.device('/'+pred+':0'): 
                with timeout(seconds=150):
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

def send_format_results(layers_types,
                    inputs_types,
                    couche1,
                    couche2,
                    dense,
                    iteration,
                    train,
                    pred):   
    
    path_output=get_path_outputs()
    default_value=tf.keras.backend.floatx()
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

    result=pd.DataFrame(columns=['Modèle','Nb(paramètres)','Date',
        'Méthode','Paramètres','CPU + Sys time','Wall Time','Précision',
        'Training time(cpu)','Training time(wall)','Train','Pred'])
    for inputs_type in inputs_types:
        for layers_type in layers_types:
            time_cpu,time_wall,accuracy,date,parameters,nb_params,cpu_time,wall_time=get_time(
                                                                                    inputs_type,
                                                                                    layers_type,
                                                                                    iteration,
                                                                                    couche1,couche2,dense,
                                                                                    train,pred)
            result=result.append({'Modèle':'CNN','CPU + Sys time':time_cpu,
                'Wall time':time_wall,'Précision':accuracy,'Date':date,
                'Méthode':'Format des inputs/layers','Paramètres':parameters,'Nb(paramètres)':nb_params,
                'Training time(cpu)':cpu_time,'Training time(wall)':wall_time,
                'Train':train,'Pred':pred},
                 ignore_index=True)
            print('Modèle: ','CNN','Input: ',inputs_type,
             '&', 'Layers: ',layers_type,
             '--> time_cpu:',time_cpu,
             '->time_wall:',time_wall,
             '-> accuracy:',accuracy)
    filename='results.csv'
    try:    
        results=pd.read_csv(path_output+filename)
        results=pd.concat((results,result),axis=0).reset_index(drop=True)
        results.to_csv(path_output+filename,index=False,header=True,encoding='utf-8-sig')
    except:
        result.to_csv(path_output+filename,index=False,header=True,encoding='utf-8-sig')
        
send_format_results(layers_types=[None],#=['float32','float64'],
                    inputs_types=[None],#=['int8','int16','int32','float16','float32'],
                    couche1=32,
                    couche2=32,
                    dense=64,
                    iteration=30,
                    train='CPU',
                    pred='CPU')