from time import process_time, time
import tempfile
import os
from sklearn.metrics import accuracy_score
import tensorflow as tf
import numpy as np
import tensorflow_model_optimization as tfmot
import pandas as pd
from datetime import datetime
import tensorflow.keras.backend as K
from statistics import mean

def get_path_outputs():
    return os.path.dirname(os.path.abspath(__file__)).replace('/src/dlcpu','')+'/outputs'

def get_paramaters_number(model):
    trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])
    return trainable_count+non_trainable_count

def get_simple_model(couche1,couche2,dense,x_train,y_train):
    start_cpu,start_wall=process_time(),time()
    model = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(32, 32,3)),
      tf.keras.layers.Conv2D(couche1, (3, 3), strides=(2, 2), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.2),
      tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
      tf.keras.layers.Conv2D(couche2, (3, 3), strides=(2, 2), padding="same"),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(dense, activation='relu'),
      tf.keras.layers.Dense(100),
    ])
    model._name='baseline'
    model.compile(optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],)

    model.fit(x_train, y_train, epochs=3,verbose=0)
    stop_cpu,stop_wall=process_time(),time()
    cpu_time=stop_cpu-start_cpu
    wall_time=stop_wall-start_wall
    return model,cpu_time,wall_time

def get_pruned_model(model,x_train,y_train):
    start_cpu,start_wall=process_time(),time()
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    batch_size = 1
    epochs = 3
    validation_split = 0.2

    num_images = x_train.shape[0] * (1 - validation_split)
    end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

#     pruning_params = {
#           'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
#                                                                    final_sparsity=0.80,
#                                                                    begin_step=0,
#                                                                    end_step=end_step)
#     }
    pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=0.80,
                                  begin_step=0, end_step=-1, frequency=100)
        }
    
    model_for_pruning = prune_low_magnitude(model, **pruning_params)
    model_for_pruning._name='pruned'
    model_for_pruning.compile(optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],)
    logdir = tempfile.mkdtemp()

    callbacks = [
      tfmot.sparsity.keras.UpdatePruningStep(),
      tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
    ]

    model_for_pruning.fit(x_train, y_train, epochs=3,verbose=0,callbacks=callbacks)
    stop_cpu,stop_wall=process_time(),time()
    cpu_time=stop_cpu-start_cpu
    wall_time=stop_wall-start_wall
    return model_for_pruning,cpu_time,wall_time

def get_time(model,x_test,y_test,iteration):
    nb_params=get_paramaters_number(model)
    accuracys=[]
    temps_cpu=[]
    temps_wall=[]
    date=datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    parameters={'Nom du modèle':model.name}
    for k in range(iteration):
        start_cpu,start_wall=process_time(),time()
        y_pred=model.predict(x_test)
        stop_cpu,stop_wall=process_time(),time()
        temps_cpu.append(stop_cpu-start_cpu)
        temps_wall.append(stop_wall-start_wall)
        accuracys.append(accuracy_score(np.argmax(y_pred,1),y_test))
    return mean(temps_cpu),mean(temps_wall),mean(accuracys),date,parameters,nb_params

def send_pruning_results(couche1,couche2,dense,iteration):
    path=get_path_outputs()
    
    cifar100 = tf.keras.datasets.cifar100
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    baseline,cpu_base,wall_base=get_simple_model(couche1,couche2,dense,x_train,y_train)
    pruned,cpu_prun,wall_prun=get_pruned_model(baseline,x_train,y_train)
    models=[baseline,pruned]
    cpu_times=[cpu_base,cpu_prun]
    wall_times=[wall_base,wall_prun]
    result=pd.DataFrame(columns=['Modèle','Nb(paramètres)','Date',
                                 'Méthode','Paramètres','CPU + Sys time',
                                 'Précision','Wall Time','Training time(cpu)',
                                 'Training time(wall)'])
    for k in range(len(models)):
        time_cpu,time_wall,accuracy,date,parameters,nb_params=get_time(models[k],x_test,y_test,iteration)
        result=result.append({'Modèle':'CNN','CPU + Sys time':time_cpu,
                              'Wall Time':time_wall,'Précision':accuracy,
                              'Date':date,'Méthode':'Pruning','Paramètres':parameters,
                              'Nb(paramètres)':nb_params,'Training time(cpu)':cpu_times[k],
                              'Training time(wall)':wall_times[k]}, ignore_index=True)
        print('Méthode: ','Pruning',
              'Modèle: ','CNN','Paramètre: ',
              parameters['Nom du modèle'],'-> accuracy:',accuracy,
              '-> time_cpu:',time_cpu,'->time_wall:',time_wall)
    filename='results.csv'
    try:    
        results=pd.read_csv(path+filename)
        results=pd.concat((results,result),axis=0).reset_index(drop=True)
        results.to_csv(path+filename,index=False,header=True,encoding='utf-8-sig')
    except:
        result.to_csv(path+filename,index=False,header=True,encoding='utf-8-sig')
        
send_pruning_results(couche1=32,
    couche2=64,
    dense=64,
    iteration=50)