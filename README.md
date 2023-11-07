# Deep Neural Network for Malaria Infected Cell Recognition

## AIM

To develop a deep neural network for Malaria infected cell recognition and to analyze the performance.

## PROBLEM STATEMENT:
The problem at hand is the automatic classification of red blood cell images into two categories: parasitized and uninfected. Malaria-infected red blood cells, known as parasitized cells, contain the Plasmodium parasite, while uninfected cells are healthy and free from the parasite. The goal is to build a convolutional neural network (CNN) model capable of accurately distinguishing between these two classes based on cell images.

Traditional methods of malaria diagnosis involve manual inspection of blood smears by trained professionals, which can be time-consuming and error-prone. Automating this process using deep learning can significantly speed up diagnosis, reduce the workload on healthcare professionals, and improve the accuracy of detection.

Our dataset comprises 27,558 cell images, evenly split between parasitized and uninfected cells. These images have been meticulously collected and annotated by medical experts, making them a reliable source for training and testing our deep neural network.

### Dataset:
![193736032-b5847f1f-f002-4edc-912a-eaf48444f1b0](https://github.com/Aashima02/malaria-cell-recognition/assets/93427086/2c515772-eb1e-4103-ae2c-e89240d50f0a)

## Neural Network Model:
![image](https://github.com/Aashima02/malaria-cell-recognition/assets/93427086/50778a7c-7b29-4c2b-abc1-85c4151fc60a)


## DESIGN STEPS

1. We begin by importing the necessary Python libraries, including TensorFlow for deep learning, data preprocessing tools, and visualization libraries.

2. To leverage the power of GPU acceleration, we configure TensorFlow to allow GPU processing, which can significantly speed up model training.

3. We load the dataset, consisting of cell images, and check their dimensions. Understanding the image dimensions is crucial for setting up the neural network architecture.

4. We create an image generator that performs data augmentation, including rotation, shifting, rescaling, and flipping. Data augmentation enhances the model's ability to generalize and recognize malaria-infected cells in various orientations and conditions.

5. We design a convolutional neural network (CNN) architecture consisting of convolutional layers, max-pooling layers, and fully connected layers. The model is compiled with appropriate loss and optimization functions.

6. We split the dataset into training and testing sets, and then train the CNN model using the training data. The model learns to differentiate between parasitized and uninfected cells during this phase.

7. We visualize the training and validation loss to monitor the model's learning progress and detect potential overfitting or underfitting.

8. We evaluate the trained model's performance using the testing data, generating a classification report and confusion matrix to assess accuracy and potential misclassifications.

9. We demonstrate the model's practical use by randomly selecting and testing a new cell image for classification.

## PROGRAM

### Import necessary libraries:
python
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf


### Allow GPU Processing:
```python
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True # dynamically grow the memory used on the GPU
config.log_device_placement = True # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)
%matplotlib inline


### Read the images:
python
my_data_dir = './dataset/cell_images'
os.listdir(my_data_dir)
test_path = my_data_dir+'/test/'
train_path = my_data_dir+'/train/'

os.listdir(train_path)
len(os.listdir(train_path+'/uninfected/'))
len(os.listdir(train_path+'/parasitized/'))
os.listdir(train_path+'/parasitized')[0]

para_img= imread(train_path+
                 '/parasitized/'+
                 os.listdir(train_path+'/parasitized')[0])
plt.imshow(para_img)


### Check Image Dimensions:
python
# Checking the image dimensions
dim1 = []
dim2 = []
for image_filename in os.listdir(test_path+'/uninfected'):
    img = imread(test_path+'/uninfected'+'/'+image_filename)
    d1,d2,colors = img.shape
    dim1.append(d1)
    dim2.append(d2)

sns.jointplot(x=dim1,y=dim2)
image_shape = (130,130,3)


### Image Generator:
python
image_gen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees
                               width_shift_range=0.10, # Shift the pic width by a max of 5%
                               height_shift_range=0.10, # Shift the pic height by a max of 5%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.1, # Shear means cutting away part of the image (max 10%)
                               zoom_range=0.1, # Zoom in by 10% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value)

image_gen.flow_from_directory(train_path)
image_gen.flow_from_directory(test_path)


### Generate the model & compile:
python
model = models.Sequential()
model.add(keras.Input(shape=(image_shape)))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu',))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu',))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(128))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

batch_size = 16


### Fit the model:
python
train_image_gen = image_gen.flow_from_directory(train_path,target_size=image_shape[:2],
                              color_mode='rgb',batch_size=batch_size,class_mode='binary')
train_image_gen.batch_size
len(train_image_gen.classes)
train_image_gen.total_batches_seen
test_image_gen = image_gen.flow_from_directory(test_path,target_size=image_shape[:2],
                              color_mode='rgb',batch_size=batch_size,
                              class_mode='binary',shuffle=False)
train_image_gen.class_indices
results = model.fit(train_image_gen,epochs=10,validation_data=test_image_gen)
model.save('cell_model.h5')


### Plot graphs:
python
losses = pd.DataFrame(model.history.history)
losses[['loss','val_loss']].plot()
model.metrics_names


### Metrics Evaluation:
python
model.evaluate(test_image_gen)
pred_probabilities = model.predict(test_image_gen)
test_image_gen.classes
predictions = pred_probabilities > 0.5
print(classification_report(test_image_gen.classes,predictions))
confusion_matrix(test_image_gen.classes,predictions)


### Check for new image:
python
list_dir=["Un Infected","parasitized"]
dir_=(rnd.choice(list_dir))
p_img=imread(train_path+'/'+dir_+'/'+os.listdir(train_path+'/'+dir_)[rnd.randint(0,100)])
img  = tf.convert_to_tensor(np.asarray(p_img))
img = tf.image.resize(img,(130,130))
img=img.numpy()
pred=bool(model.predict(img.reshape(1,130,130,3))<0.5 )
plt.title("Model prediction: "+("Parasitized" if pred  else "Un Infected")
			+"\nActual Value: "+str(dir_))
plt.axis("off")
plt.imshow(img)
plt.show()
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![image](https://github.com/Aashima02/malaria-cell-recognition/assets/93427086/55f7d2f1-3ec2-4c39-b218-bcf7c8a2ea83)


### Classification Report

![image](https://github.com/Aashima02/malaria-cell-recognition/assets/93427086/2c4ed8de-e78a-46e1-b532-2520d584c142)


### Confusion Matrix

![image](https://github.com/Aashima02/malaria-cell-recognition/assets/93427086/ad118a7b-cb80-448b-abc5-e4a744dc9e08)

### New Sample Data Prediction

![image](https://github.com/Aashima02/malaria-cell-recognition/assets/93427086/5ed5a3cf-c82b-440e-a89c-295eb26d0a6f)

## RESULT:
The model's performance is evaluated through training and testing, and it shows potential for assisting healthcare professionals in diagnosing malaria more efficiently and accurately.
