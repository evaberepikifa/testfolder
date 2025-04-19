import pandas as pd
# from tqdm import tqdm
import numpy as np
from glob import glob
import cv2
import os
import shutil
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
import random
#import torchsummary
from PIL import Image
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
import tensorflow
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.applications.densenet import DenseNet169
from tensorflow.keras import layers, models, optimizers

#Force to use available GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        #TensorFlow to use only the first GPU (optional)
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("✅ GPU is available and in use.")
    except RuntimeError as e:
        print("RuntimeError while setting GPU config:", e)
else:
    print("No GPU found. Running on CPU.")

# Set seeds
Seed = 42
os.environ['PYTHONHASHSEED'] = str(Seed)
random.seed(Seed)
np.random.seed(Seed)
tensorflow.random.set_seed(Seed)

#Pre-processing 
df = pd.read_csv('Chest_xray_Corona_Metadata.csv')
df.head()

#discard stress-smoking pneumonia images
df = df[df.Label_1_Virus_category != "Stress-Smoking"]

#create target variables
targets = [
    (df['Label'] == "Normal"),
    (df['Label'] == "Pnemonia") & (df['Label_1_Virus_category'] == "Virus"),
    (df['Label'] == "Pnemonia") & (df['Label_1_Virus_category'] == "bacteria")
]

values = ['Normal', 'Pnemonia-virus', 'Pnemonia-bacteria']
default = 'NaN'
df['target'] = np.select(targets, values, default=default)

chest_df = pd.DataFrame(df)
chest_df.shape

#check counts of target variables
target_counts = chest_df['target'].value_counts()
print(target_counts)

#split the dataframe
train = chest_df[chest_df['Dataset_type'] == 'TRAIN']
test = chest_df[chest_df['Dataset_type'] == 'TEST']

y_train = train['target']
y_test = test['target']


#Augment data
batch_size = 32

train['target'] = train['target'].astype(str)
test['target'] = test['target'].astype(str)


# ✅ Split the DataFrame

test_df = test
train_df, val_df= train_test_split(
    train,
    test_size=0.2,
    stratify=train['target'],
    random_state=42

)

#Create data generators
train_gen = ImageDataGenerator(
    rescale=1/255,
    zoom_range=0.1,
    rotation_range=20,
    width_shift_range=0.1,
    shear_range=0.1,
    samplewise_center=True,
    samplewise_std_normalization=True
)

val_gen = ImageDataGenerator(
    rescale=1/255,
    samplewise_center=True,
    samplewise_std_normalization=True
)


# Get the directory 
current_dir = os.getcwd()
train_dir = os.path.join(current_dir, 'Coronahack-Chest-XRay-Dataset', 'Coronahack-Chest-XRay-Dataset', 'train')

train_flow = train_gen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_dir,
    x_col="X_ray_image_name",
    y_col='target',
    batch_size=32,
    shuffle=True,
    seed=Seed,
    class_mode='categorical',
    target_size=(224, 224)
)

val_dir = os.path.join(current_dir, 'Coronahack-Chest-XRay-Dataset', 'Coronahack-Chest-XRay-Dataset', 'train')

val_flow = val_gen.flow_from_dataframe(
    dataframe=val_df,
    directory=val_dir,
    x_col="X_ray_image_name",
    y_col='target',
    batch_size=32,
    shuffle=False,
    class_mode='categorical',
    target_size=(224, 224)
)

test_dir = os.path.join(current_dir, 'Coronahack-Chest-XRay-Dataset', 'Coronahack-Chest-XRay-Dataset', 'test')

test_flow = val_gen.flow_from_dataframe(
    dataframe=test_df,
    directory=test_dir,
    x_col="X_ray_image_name",
    y_col='target',
    batch_size=32,
    shuffle=False,
    class_mode='categorical',
    target_size=(224, 224)
)

#early stopping  & define epochs
early_stopping_callbacks = tensorflow.keras.callbacks.EarlyStopping(patience = 15, restore_best_weights = True, verbose = 1)
epochs = 20

#Using DenseNet169 and adding changes to Model Architcture

base_model = DenseNet169(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,

  layers.Flatten(),
  layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001)),
  layers.BatchNormalization(),
  layers.Activation('relu'),
  layers.Dropout(0.3),

  layers.Dense(256, kernel_regularizer=regularizers.l2(0.0001)),
  layers.BatchNormalization(),
  layers.Activation('relu'),
  layers.Dropout(0.4),

  layers.Dense(128, kernel_regularizer=regularizers.l2(0.0001)),
  layers.BatchNormalization(),
  layers.Activation('relu'),
  layers.Dropout(0.3),

  layers.Dense(64, kernel_regularizer=regularizers.l2(0.0001)),
  layers.BatchNormalization(),
  layers.Activation('relu'),
  layers.Dropout(0.3),

  layers.Dense(3, activation='softmax')
 
  ])

model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.Adam(learning_rate=1e-4),
    metrics=['accuracy']
)

History= model.fit(train_flow, epochs=epochs, validation_data=test_flow,callbacks=[early_stopping_callbacks])

test_loss, test_acc = model.evaluate(test_flow)
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)

#Saved Model 
model.save("xray_classifier_v2_88acc.h5")



