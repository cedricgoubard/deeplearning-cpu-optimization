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

nb_test=10000
couche1=64
couche2=128
dense=512

cifar100 = tf.keras.datasets.cifar100
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

X_test_numpy = np.array(x_test, dtype=np.float32)
y_test_numpy =np.array(y_test, dtype=np.float32)

model_name='CNN'
method_name='TFLite'

def GetParametersNumber(model):
    trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])
    return trainable_count+non_trainable_count

def GetSimpleModel(couche1=16,couche2=32,dense=512):
    start_cpu,start_wall=process_time(),time()
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
    stop_cpu,stop_wall=process_time(),time()
    temps_cpu=stop_cpu-start_cpu
    temps_wall=stop_wall-start_wall
    return model,temps_cpu,temps_wall
def GetSimpleModel4Quantization(couche1=16,couche2=32,dense=512):
    start_cpu,start_wall=process_time(),time()
    model = tf.keras.Sequential([
      keras.layers.InputLayer(input_shape=(32, 32,3)),
      keras.layers.Conv2D(couche1, (3, 3), strides=(2, 2), padding="same"),
#      keras.layers.LeakyReLU(alpha=0.2),
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
    stop_cpu,stop_wall=process_time(),time()
    temps_cpu=stop_cpu-start_cpu
    temps_wall=stop_wall-start_wall
    return model

def GetClusteredModel(model):
    start_cpu,start_wall=process_time(),time()
    cluster_weights = tfmot.clustering.keras.cluster_weights
    CentroidInitialization = tfmot.clustering.keras.CentroidInitialization

    clustering_params = {
      'number_of_clusters': 16,
      'cluster_centroids_init': CentroidInitialization.LINEAR
    }

    # Cluster a whole model
    clustered_model = cluster_weights(model, **clustering_params)

    # Use smaller learning rate for fine-tuning clustered model
    opt = tf.keras.optimizers.Adam(learning_rate=1e-5)

    clustered_model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=opt,
      metrics=['accuracy'])

    clustered_model.fit(x_train, y_train, epochs=3,verbose=0)
    stop_cpu,stop_wall=process_time(),time()
    temps_cpu=stop_cpu-start_cpu
    temps_wall=stop_wall-start_wall
    return clustered_model,temps_cpu,temps_wall

def GetQuantizedModel(model):
    start_cpu,start_wall=process_time(),time()
    quantize_model = tfmot.quantization.keras.quantize_model

    q_aware_model = quantize_model(model)

    q_aware_model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    q_aware_model.fit(x_train, y_train, epochs=3,verbose=0)
    stop_cpu,stop_wall=process_time(),time()
    temps_cpu=stop_cpu-start_cpu
    temps_wall=stop_wall-start_wall
    return q_aware_model,temps_cpu,temps_wall

#ne fonctionne pas avec TFlite
def GetPrunedModel(model):
    start_cpu,start_wall=process_time(),time()
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    batch_size = 1
    epochs = 3
    validation_split = 0.2

    num_images = x_train.shape[0] * (1 - validation_split)
    end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

    pruning_params = {
          'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                   final_sparsity=0.80,
                                                                   begin_step=0,
                                                                   end_step=end_step)
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
    temps_cpu=stop_cpu-start_cpu
    temps_wall=stop_wall-start_wall
    return model_for_pruning,temps_cpu,temps_wall

def GetTFLmodel(model):
    start_cpu,start_wall=process_time(),time()
    tf_lite_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tf_lite_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = tf_lite_converter.convert()
    tflite_model_name = 'TFlite_post_quantModel8bit'
    open(tflite_model_name, "wb").write(tflite_model)

    interpreter = tf.lite.Interpreter(model_path = tflite_model_name)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details() #1
    output_details = interpreter.get_output_details() #16
    stop_cpu,stop_wall=process_time(),time()
    temps_cpu=stop_cpu-start_cpu
    temps_wall=stop_wall-start_wall
    return interpreter
#pour les mod??les classiques
def GetTime(model_input,parameters):
    nb_params=GetParametersNumber(model_input)
    predictions=[]
    temps_cpu=[]
    temps_wall=[]
    date=datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    for k in range(nb_test):
        start_cpu,start_wall=process_time(),time()
        pred=model_input.predict(x_test[k].reshape(1,32,32,3))
        stop_cpu,stop_wall=process_time(),time()
        temps_cpu.append(stop_cpu-start_cpu)
        temps_wall.append(stop_wall-start_wall)
        predictions.append(np.argmax(pred))
    accuracy=accuracy_score(np.array(predictions),y_test[0:nb_test][:,0])
    return mean(temps_cpu),mean(temps_wall),accuracy,date,parameters,nb_params
#pour les mod??les TFlite
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
#pour enregistrer les infos dans result

def SendData(result,time_cpu,time_wall,accuracy,date,parameters,nb_params,cpu_time,wall_time):
    result=result.append({'Mod??le':model_name,'CPU + Sys time':time_cpu,'Wall Time':time_wall,'Pr??cision':accuracy,'Date':date,'M??thode':method_name,'Param??tres':parameters,'Nb(param??tres)':nb_params,'Training time(cpu)':cpu_time,'Training time(wall)':wall_time}, ignore_index=True)
    return result

def SendTFliteResults():

    result=pd.DataFrame(columns=['Mod??le','Nb(param??tres)','Date','M??thode','Param??tres','CPU + Sys time','Pr??cision','Wall Time','Training time(cpu)','Training time(wall)'])

    model,simple_cpu,simple_wall=GetSimpleModel(couche1,couche2,dense)
    #model_pruned=GetPrunedModel(model) #ne fonctionne pas avec TFlite
    model_copy = keras.models.clone_model(model)
    model_clustered,clust_cpu,clust_wall=GetClusteredModel(model_copy)

    model4quantization = GetSimpleModel4Quantization(couche1,couche2,dense)
    model_quantized,quant_cpu,quant_wall=GetQuantizedModel(model4quantization)

    cpu_times=[simple_cpu,simple_cpu+clust_cpu,simple_cpu+quant_cpu]
    wall_times=[simple_wall,simple_wall+clust_wall,simple_wall+quant_wall]
    names=['Baseline','Weight_Clustering','Quantized']
    models=[model,model_clustered,model_quantized]

    for k in range(len(names)):
        time_cpu,time_wall,accuracy,date,parameters,nb_params=GetTime(models[k],names[k])
        result=SendData(result,time_cpu,time_wall,accuracy,date,parameters,nb_params,cpu_times[k],wall_times[k])

    names=['TFlite(baseline)','TFlite(weight clustering)','TFlite(quantization)']
    for k in range(len(names)):
        start_cpu,start_wall=process_time(),time()
        intepreter=GetTFLmodel(models[k])
        time_cpu,time_wall,accuracy,date,parameters,nb_params=GetTFLtime(intepreter,names[k])
        stop_cpu,stop_wall=process_time(),time()
        temps_cpu_l=stop_cpu-start_cpu
        temps_wall_l=stop_wall-start_wall
        result=SendData(result,time_cpu,time_wall,accuracy,date,parameters,nb_params,temps_cpu_l+cpu_times[k],temps_wall_l+wall_times[k])

    try:    
        results=pd.read_csv('/home/arnaudhureaux/deeplearning-cpu-optimization/outputs/results.csv')
        results=pd.concat((results,result),axis=0).reset_index(drop=True)
        results.to_csv('/home/arnaudhureaux/deeplearning-cpu-optimization/outputs/results.csv',index=False,header=True,encoding='utf-8-sig')
    except:
        result.to_csv('/home/arnaudhureaux/deeplearning-cpu-optimization/outputs/results.csv',index=False,header=True,encoding='utf-8-sig')
SendTFliteResults()