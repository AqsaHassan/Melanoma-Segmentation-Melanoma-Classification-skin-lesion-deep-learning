import os
from glob import glob

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import Model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
#from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping


#%matplotlib inline
import matplotlib.pyplot as plt

import os
from glob import glob
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import Model, Input
from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet201

#from keras.optimizers import Adam

base_skin_dir='/home/aqsah/projects/isic2018/isic2018'

skin_df = pd.read_csv(os.path.join(base_skin_dir, 'HAM10000_metadata')) # load $
skin_df.head()

X_test = np.load("/home/aqsah/projects/isic2018/isic2018/segment/256_192_test.npy")
y_test = np.load("/home/aqsah/projects/isic2018/isic2018/segment/test_labels.npy")
X_train = np.load("/home/aqsah/projects/isic2018/isic2018/segment/256_192_train.npy")
y_train = np.load("/home/aqsah/projects/isic2018/isic2018/segment/train_labels.npy")
X_val = np.load("/home/aqsah/projects/isic2018/isic2018/segment/256_192_val.npy")
y_val = np.load("/home/aqsah/projects/isic2018/isic2018/segment/val_labels.npy")

X_train.shape, X_val.shape

y_train.shape, y_val.shape

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

X_val.shape, X_test.shape, y_val.shape, y_test.shape
input_shape = X_val[0,:,:,:].shape
model_input = Input(shape=input_shape)
#############################
inception = InceptionV3(input_shape=input_shape, input_tensor=model_input, include_top=False, weights=None)
for layer in inception.layers:
    layer.trainable = True
inception_last_layer = inception.get_layer('mixed10')
print('last layer output shape:', inception_last_layer.output_shape)
inception_last_output = inception_last_layer.output

# Flatten the output layer to 1 dimension
x_inception = layers.GlobalMaxPooling2D()(inception_last_output)
# Add a fully connected layer with 512 hidden units and ReLU activation
x_inception = layers.Dense(512, activation='relu')(x_inception)
# Add a dropout rate of 0.7
x_inception = layers.Dropout(0.5)(x_inception)
# Add a final sigmoid layer for classification
x_inception = layers.Dense(7, activation='softmax')(x_inception)

# Configure and compile the model

inception_model = Model(model_input, x_inception)
optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
#inception_model.compile(loss='categorical_crossentropy',
#              optimizer=optimizer,
#             metrics=['accuracy'])

inception_model.compile(loss='categorical_crossentropy', 
              optimizer=optimizer,metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])


inception_model.load_weights("SegInceptionV3.h5")
inception_model.summary()
##############################
denseNet = DenseNet201(input_shape=input_shape, input_tensor=model_input, include_top=False, weights=None)
for layer in denseNet.layers:
    layer.trainable = True
denseNet_last_layer = denseNet.get_layer('relu')
print('last layer output shape:', denseNet_last_layer.output_shape)
denseNet_last_output = denseNet_last_layer.output
# Flatten the output layer to 1 dimension
x_denseNet = layers.GlobalMaxPooling2D()(denseNet_last_output)
# Add a fully connected layer with 512 hidden units and ReLU activation
x_denseNet = layers.Dense(512, activation='relu')(x_denseNet)
# Add a dropout rate of 0.7
x_denseNet = layers.Dropout(0.5)(x_denseNet)
# Add a final sigmoid layer for classification
x_denseNet = layers.Dense(7, activation='softmax')(x_denseNet)

# Configure and compile the model

denseNet_model = Model(model_input, x_denseNet)
optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
denseNet_model.compile(loss='categorical_crossentropy', 
              optimizer=optimizer,metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
denseNet_model.load_weights("DenseNetFull.h5")
denseNet_model.summary()

######################
def ensemble(models, model_input):
    outputs = [model.outputs[0] for model in models]
    y = layers.Average()(outputs)
    model = Model(model_input, y, name='ensemble')
    return model
ensemble_model = ensemble([denseNet_model, inception_model], model_input)
ensemble_model.compile(loss='categorical_crossentropy', 
              optimizer=optimizer,metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])


X_test = np.load("/home/aqsah/projects/isic2018/isic2018/256_192_test.npy")
y_test = np.load("/home/aqsah/projects/isic2018/isic2018/test_labels.npy")
y_test = to_categorical(y_test)

#loss_val, acc_val = ensemble_model.evaluate(X_val, y_val, verbose=1)
loss_val, acc_val, prec_val, recall_val = ensemble_model.evaluate(X_val, y_val, verbose=1)
print("Validation: accuracy = %f  ;  loss_v = %f; prec_val = %f ; recall_val = %f" % (acc_val, loss_val, prec_val, recall_val))
#Validation: accuracy = 0.888027  ;  loss_v = 0.399892
#loss_test, acc_test = ensemble_model.evaluate(X_test, y_test, verbose=1)
loss_test, acc_test, prec_test, recall_test = ensemble_model.evaluate(X_test, y_test, verbose=1)
#print("Test: accuracy = %f  ;  loss = %f" % (acc_test, loss_test))
print("Test: accuracy = %f  ;  loss = %f; precision = %f; recall = %f" % (acc_test, loss_test, prec_test, recall_test))

# Retrieve a list of accuracy results on training and test data
# sets for each training epoch
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Retrieve a list of list results on training and test data
# sets for each training epoch
loss = history.history['loss']
val_loss = history.history['val_loss']

# Get number of epochs
epochs = range(len(acc))

# Plot training and validation accuracy per epoch
plt.plot(epochs, acc, label = "training")
plt.plot(epochs, val_acc, label = "validation")
plt.legend(loc="upper left")
plt.title('Training and validation accuracy')
plt.savefig("SegmentedInceptionV3Acc.png")



# Plot training and validation loss per epoch
plt.figure()

plt.plot(epochs, loss, label = "training")
plt.plot(epochs, val_loss, label = "validation")
plt.legend(loc="upper right")
plt.title('Training and validation loss')
plt.savefig("SegmentedInceptionV3Loss.png")
