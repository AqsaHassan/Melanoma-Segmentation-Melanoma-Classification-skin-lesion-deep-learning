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

from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import Model
from keras.applications.densenet import DenseNet201
#from keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam
#from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ReduceLROnPlateau
import keras.backend as K

# %matplotlib inline
import matplotlib.pyplot as plt
#from google.colab import drive

#drive.mount('/content/drive')

# Set the ROOT_DIR variable to the root directory of the HAMdataset
#ROOT_DIR = '/content/drive/MyDrive/HAMdataset'
#assert os.path.exists(ROOT_DIR), 'ROOT_DIR does not exist. Did you forget to read the instructions above? ;)'

#base_skin_dir = "/Users/Hoang/Machine_Learning/skin_cancer/skin-cancer-mnist-ham10000"

base_skin_dir='/home/aqsah/projects/isic2018/isic2018'

skin_df = pd.read_csv(os.path.join(base_skin_dir, 'HAM10000_metadata')) # load in the data
skin_df.head()

X_train = np.load("/home/aqsah/projects/isic2018/isic2018/256_192_train.npy")
y_train = np.load("/home/aqsah/projects/isic2018/isic2018/train_labels.npy")
X_val = np.load("/home/aqsah/projects/isic2018/isic2018/256_192_val.npy")
y_val = np.load("/home/aqsah/projects/isic2018/isic2018/val_labels.npy")

X_train.shape, X_val.shape

y_train.shape, y_val.shape

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

y_train.shape, y_val.shape

"""**Load the Pretrained Model**"""
pre_trained_model = InceptionV3(input_shape=(192, 256, 3), include_top=False, weights="imagenet")
for layer in pre_trained_model.layers:
    print(layer.name)
    layer.trainable = False
    
print(len(pre_trained_model.layers))

last_layer = pre_trained_model.get_layer('mixed10')
print('last layer output shape:', last_layer.output_shape)
last_output = last_layer.output


"""**Define MODEL**"""
# Flatten the output layer to 1 dimension
x = layers.GlobalMaxPooling2D()(last_output)
# Add a fully connected layer with 512 hidden units and ReLU activation
x = layers.Dense(512, activation='relu')(x)
# Add a dropout rate of 0.7
x = layers.Dropout(0.5)(x)
# Add a final sigmoid layer for classification
x = layers.Dense(7, activation='softmax')(x)

# Configure and compile the model

model = Model(pre_trained_model.input, x)
optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
#model.compile(loss='categorical_crossentropy',
#              optimizer=optimizer,
#             metrics=['accuracy'])

#model.summary()

model.compile(loss='categorical_crossentropy', 
              optimizer=optimizer,metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

model.summary()


train_datagen = ImageDataGenerator(rotation_range=60, width_shift_range=0.2, height_shift_range=0.2,
                                   shear_range=0.2, zoom_range=0.2, fill_mode='nearest')

train_datagen.fit(X_train)

val_datagen = ImageDataGenerator()
val_datagen.fit(X_val)

batch_size = 64
epochs = 3
history = model.fit_generator(train_datagen.flow(X_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = val_datagen.flow(X_val, y_val),
                              verbose = 1, steps_per_epoch=(X_train.shape[0] // batch_size), 
                              validation_steps=(X_val.shape[0] // batch_size))


for layer in pre_trained_model.layers:
    layer.trainable = True
optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#model.compile(loss='categorical_crossentropy',
#              optimizer=optimizer,
#              metrics=['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, 
                                            min_lr=0.000001, cooldown=2)
model.summary()

batch_size = 32
epochs = 20
history = model.fit_generator(train_datagen.flow(X_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = val_datagen.flow(X_val, y_val),
                              verbose = 1, steps_per_epoch=(X_train.shape[0] // batch_size),
                              validation_steps=(X_val.shape[0] // batch_size),
                              callbacks=[learning_rate_reduction])

#loss_val, acc_val = model.evaluate(X_val, y_val, verbose=1)
loss_val, acc_val, prec_val, recall_val = model.evaluate(X_val, y_val, verbose=1)
print("Validation: accuracy = %f  ;  loss_v = %f; prec_val = %f ; recall_val = %f" % (acc_val, loss_val, prec_val, recall_val))
#print("Validation: accuracy = %f  ;  loss_v = %f" % (acc_val, loss_val))


X_test = np.load("/home/aqsah/projects/isic2018/isic2018/256_192_test.npy")
y_test = np.load("/home/aqsah/projects/isic2018/isic2018/test_labels.npy")
y_test = to_categorical(y_test)
#loss_test, acc_test = model.evaluate(X_test, y_test, verbose=1)
loss_test, acc_test, prec_test, recall_test = model.evaluate(X_test, y_test, verbose=1)
#print("Test: accuracy = %f  ;  loss = %f" % (acc_test, loss_test))
print("Test: accuracy = %f  ;  loss = %f; precision = %f; recall = %f" % (acc_test, loss_test, prec_test, recall_test))
model.save("InceptionV3.h5")

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

