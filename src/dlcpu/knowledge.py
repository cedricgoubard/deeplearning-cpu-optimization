from time import process_time, time
from statistics import mean
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from datetime import datetime
import os

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

class Distiller(tf.keras.Model):
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
    
def get_path_outputs():
    return os.path.dirname(os.path.abspath(__file__)).replace('/src/dlcpu','')+'/outputs'
    
def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    return model
def fit(model):
    model.fit(x_train, y_train, epochs=3,verbose=0)
    return model

def get_parameters_number(model):
    trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    return trainable_count+non_trainable_count

def get_time(model,iteration):
    nb_params=get_parameters_number(model)
    accuracys,temps_cpu,temps_wall=[],[],[]
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

def get_teacher(couche1,couche2,dense):
    teacher = tf.keras.Sequential(
        [   tf.keras.Input(shape=(32, 32, 3)),
            tf.keras.layers.Conv2D(couche1*2, (3, 3), strides=(2, 2), padding="same"),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
            tf.keras.layers.Conv2D(couche2*2, (3, 3), strides=(2, 2), padding="same"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(dense*2,activation='relu'),
            tf.keras.layers.Dense(100),],
        name="teacher",)
    return teacher
def get_student(couche1,couche2,dense):
    student = tf.keras.Sequential(
        [   tf.keras.Input(shape=(32, 32, 3)),
            tf.keras.layers.Conv2D(couche1, (3, 3), strides=(2, 2), padding="same"),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
            tf.keras.layers.Conv2D(couche2, (3, 3), strides=(2, 2), padding="same"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(dense,activation='relu'),
            tf.keras.layers.Dense(100),],
        name="student",)
    return student

def send_knowledge_results(couche1,couche2,dense,iteration):
    path=get_path_outputs()
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    # teacher
    start_cpu,start_wall=process_time(),time()
    teacher=get_teacher(couche1,couche2,dense)
    compile_model(teacher)
    fit(teacher)
    stop_cpu,stop_wall=process_time(),time()
    teacher_cpu=stop_cpu-start_cpu
    teacher_wall=stop_wall-start_wall
    # student
    start_cpu,start_wall=process_time(),time()
    student=get_student(couche1,couche2,dense)
    distiller = Distiller(student=student, teacher=teacher)
    distiller.compile(
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        distillation_loss_fn=tf.keras.losses.KLDivergence(),
        alpha=0.1,
        temperature=10,
    )
    fit(distiller)
    stop_cpu,stop_wall=process_time(),time()
    student_cpu=stop_cpu-start_cpu
    student_wall=stop_wall-start_wall
    # baseline
    start_cpu,start_wall=process_time(),time()
    student_scratch = tf.keras.models.clone_model(student)
    student_scratch._name='baseline'
    compile_model(student_scratch)
    fit(student_scratch)
    stop_cpu,stop_wall=process_time(),time()
    baseline_cpu=stop_cpu-start_cpu
    baseline_wall=stop_wall-start_wall
    cpu_time=[student_cpu,teacher_cpu,baseline_cpu]
    wall_time=[student_wall,teacher_wall,baseline_wall]
    models=[distiller.student,teacher,student_scratch]

    result=pd.DataFrame(columns=['Modèle','Nb(paramètres)','Date',
                                 'Méthode','Paramètres','CPU + Sys time',
                                 'Précision','Wall Time','Training time(cpu)',
                                 'Training time(wall)'])
    for k in range(len(models)):
        time_cpu,time_wall,accuracy,date,parameters,nb_params=get_time(models[k],iteration)
        result=result.append({'Modèle':'CNN','CPU + Sys time':time_cpu,'Wall Time':time_wall,
                              'Précision':accuracy,'Date':date,'Méthode':'Knowledge Distilation',
                              'Paramètres':parameters,'Nb(paramètres)':nb_params,
                              'Training time(cpu)':cpu_time[k],'Training time(wall)':wall_time[k]}, 
                             ignore_index=True)
        print('Méthode: ','Knowledge Distilation',
              'Modèle: CNN','Paramètre: ',
              parameters['Nom du modèle'],'--> time_cpu:',
              time_cpu,'->time_wall:',
              time_wall,'-> accuracy:',accuracy)
    filename='results.csv'
    try:    
        results=pd.read_csv(path+filename)
        results=pd.concat((results,result),axis=0).reset_index(drop=True)
        results.to_csv(path+filename,index=False,header=True,encoding='utf-8-sig')
    except:
        result.to_csv(path+filename,index=False,header=True,encoding='utf-8-sig')

send_knowledge_results(
    couche1=32,
    couche2=64,
    dense=64,
    iteration=30)