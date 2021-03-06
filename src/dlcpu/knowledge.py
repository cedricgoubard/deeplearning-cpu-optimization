from time import process_time, time
from statistics import mean
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import tensorflow.keras.backend as K
from sklearn.metrics import accuracy_score
import pandas as pd
from datetime import datetime

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
couche1=64
couche2=128
dense=512
iteration=50

class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        """ Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            )
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results

def Compile(model):
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model
def Fit(model):
    model.fit(x_train, y_train, epochs=3,verbose=0)
    return model

def GetParametersNumber(model):
    trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])
    return trainable_count+non_trainable_count

def GetTime(model):
    nb_params=GetParametersNumber(model)
    accuracys=[]
    temps_cpu=[]
    temps_wall=[]
    date=datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    parameters={'Nom du mod??le':model.name}
    for k in range(iteration):
        start_cpu,start_wall=process_time(),time()
        y_pred=model.predict(x_test)
        stop_cpu,stop_wall=process_time(),time()
        temps_cpu.append(stop_cpu-start_cpu)
        temps_wall.append(stop_wall-start_wall)
        accuracys.append(accuracy_score(np.argmax(y_pred,1),y_test))
    return mean(temps_cpu),mean(temps_wall),mean(accuracys),date,parameters,nb_params

def GetTeacher(couche1,couche2,dense):
    teacher = keras.Sequential(
        [   keras.Input(shape=(32, 32, 3)),
            layers.Conv2D(couche1*2, (3, 3), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
            layers.Conv2D(couche2*2, (3, 3), strides=(2, 2), padding="same"),
            layers.Flatten(),
            layers.Dense(dense*2,activation='relu'),
            layers.Dense(100),],
        name="teacher",)
    return teacher
def GetStudent(couche1,couche2,dense):
    student = keras.Sequential(
        [   keras.Input(shape=(32, 32, 3)),
            layers.Conv2D(couche1, (3, 3), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
            layers.Conv2D(couche2, (3, 3), strides=(2, 2), padding="same"),
            layers.Flatten(),
            layers.Dense(dense,activation='relu'),
            layers.Dense(100),],
        name="student",)
    return student


def SendKnowledgeResults():
    method_name='Knowledge Distilation'
    model_name='CNN'

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # teacher
    start_cpu,start_wall=process_time(),time()
    teacher=GetTeacher(couche1,couche2,dense)
    Compile(teacher)
    Fit(teacher)
    stop_cpu,stop_wall=process_time(),time()
    teacher_cpu=stop_cpu-start_cpu
    teacher_wall=stop_wall-start_wall

    # student
    start_cpu,start_wall=process_time(),time()
    student=GetStudent(couche1,couche2,dense)
    distiller = Distiller(student=student, teacher=teacher)
    distiller.compile(
        optimizer=keras.optimizers.Adam(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
        student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        distillation_loss_fn=keras.losses.KLDivergence(),
        alpha=0.1,
        temperature=10,
    )
    Fit(distiller)
    stop_cpu,stop_wall=process_time(),time()
    student_cpu=stop_cpu-start_cpu
    student_wall=stop_wall-start_wall

    # baseline
    start_cpu,start_wall=process_time(),time()
    student_scratch = keras.models.clone_model(student)
    student_scratch._name='baseline'
    Compile(student_scratch)
    Fit(student_scratch)
    stop_cpu,stop_wall=process_time(),time()
    baseline_cpu=stop_cpu-start_cpu
    baseline_wall=stop_wall-start_wall

    cpu_time=[student_cpu,teacher_cpu,baseline_cpu]
    wall_time=[student_wall,teacher_wall,baseline_wall]
    models=[distiller.student,teacher,student_scratch]

    result=pd.DataFrame(columns=['Mod??le','Nb(param??tres)','Date','M??thode','Param??tres','CPU + Sys time','Pr??cision','Wall Time','Training time(cpu)','Training time(wall)'])
    for k in range(len(models)):
        time_cpu,time_wall,accuracy,date,parameters,nb_params=GetTime(models[k])
        result=result.append({'Mod??le':model_name,'CPU + Sys time':time_cpu,'Wall Time':time_wall,'Pr??cision':accuracy,'Date':date,'M??thode':method_name,'Param??tres':parameters,'Nb(param??tres)':nb_params,'Training time(cpu)':cpu_time[k],'Training time(wall)':wall_time[k]}, ignore_index=True)
        print('M??thode: ',method_name,'Mod??le: ',model_name,'Param??tre: ',parameters['Nom du mod??le'],'--> time_cpu:',time_cpu,'->time_wall:',time_wall,'-> accuracy:',accuracy)

    try:    
        results=pd.read_csv('/home/arnaudhureaux/deeplearning-cpu-optimization/outputs/results.csv')
        results=pd.concat((results,result),axis=0).reset_index(drop=True)
        results.to_csv('/home/arnaudhureaux/deeplearning-cpu-optimization/outputs/results.csv',index=False,header=True,encoding='utf-8-sig')
    except:
        result.to_csv('/home/arnaudhureaux/deeplearning-cpu-optimization/outputs/results.csv',index=False,header=True,encoding='utf-8-sig')

#SendKnowledgeResults()