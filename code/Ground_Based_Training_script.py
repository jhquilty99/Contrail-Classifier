import numpy as np
import keras
import tensorflow as tf
import scikit-learn
import os
import Pillow as PIL

from keras.applications import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Dataset

from sklearn.utils import class_weight

from os import listdir
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

ml_client = MLClient.from_config()

# Gets the datastore
datastore = ml_client.datastores.get("contrail_datastore")

# Creates a file dataset
dataset = Dataset.File.from_files(path=[(datastore, "CCSN_v2"), 
                                        (datastore, "CLASA"), 
                                        (datastore, "Roboflow"), 
                                        (datastore, "Singapore Data Swimcat")])

# Registers the file dataset
dataset = dataset.register(workspace=ml_client.workspace, 
                           name='ContrailImages',
                           description='This dataset contains images of various clouds and contrails for image classification')

# Specifies a compute target
compute_target = ml_client.compute_targets.get("cpu-cluster")

# Mounts the dataset
with dataset.mount(compute_target) as mount_context:
    mounted_path = mount_context.mount_point

    def image_loader(image_dir, dictionary):
        # List to hold all image data
        images = list()
        # List to hold image classifications (1,0)
        classes = list()
        # List to hold folders
        folders = list()
        # Specify the common size to resize all images
        common_size = (400, 400)
        # Iterate over each image in the directory
        for folder in os.listdir(image_dir):
            # Only open files with the specified filetype extension (e.g., ".png" or ".jpg")
            for filename in os.listdir(os.path.join(image_dir, folder)):
                if filename.endswith('.jpg'):
                    # Open each image file
                    img_path = os.path.join(image_dir, folder, filename)
                    img = Image.open(img_path)
                    # Resize image to the common size
                    img = ImageOps.fit(img, common_size, Image.Resampling.LANCZOS)
                    # Append the image data to your list
                    images.append(img)
                    classes.append(dictionary[folder])
                    folders.append(image_dir)
        # Now the 'images' list contains all the images in the image_dir as PIL Image objects, with the labels in the 'classes' list
        return np.array([np.array(image) for image in images]), np.array(classes), np.array(folders)

    image_dir = os.path.join(mounted_path, "Roboflow")
    robo_dictionary = {
        'Contrail':1,
        'No_contrail':0
    }
    images_robo, classes_robo, folder_robo = image_loader(image_dir, robo_dictionary)

    image_dir = os.path.join(mounted_path, "CCSN_v2")
    ccsn_dictionary = {
        'Ct':1,
        'Ac':0, 'Sc':0, 'Ns':0, 'Cu':0, 'Ci':0, 'Cc':0, 'Cb':0, 'As':0, 'Cs':0, 'St':0
    }
    images_ccsn, classes_ccsn, folder_ccsn = image_loader(image_dir, ccsn_dictionary)

    image_dir = os.path.join(mounted_path, "CLASA")
    clasa_dictionary = {
        'Contrail':1,
        'Cirrus':0
    }
    images_clasa, classes_clasa, folder_clasa = image_loader(image_dir, clasa_dictionary)

    image_dir = os.path.join(mounted_path, "Singapore Data Swimcat")
    singapore_dictionary = {
        'A-sky':0,
        'B-pattern':0,
        'C-thick-dark':0,
        'D-thick-white':0,
        'E-veil':0
    }
    images_singapore, classes_singapore, folder_singapore = image_loader(image_dir, singapore_dictionary)

    # Merge all the folders into two lists: one containing images, and the other has the labels
    images_all = np.concatenate([images_robo, images_ccsn, images_clasa, images_singapore])
    classes_all = np.concatenate([classes_robo, classes_ccsn, classes_clasa, classes_singapore])
    folders_all = np.concatenate([folder_robo, folder_ccsn, folder_clasa, folder_singapore])

    X_train, X_test, y_train, y_test = train_test_split(images_all, classes_all, test_size=0.2, random_state=42)

    # Import the pre-built ResNet50 model
    base_model = ResNet50(weights='imagenet', include_top=False)

    # Add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # Add a fully-connected layer
    x = Dense(1024, activation='relu')(x)

    # Add a logistic layer for binary classification
    predictions = Dense(1, activation='sigmoid')(x)

    # This is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # First: train only the top layers (which were randomly initialized)
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    X_train_tensor = tf.convert_to_tensor([np.array(img) for img in X_train])
    y_train_tensor = tf.convert_to_tensor(y_train)
    X_test_tensor = tf.convert_to_tensor([np.array(img) for img in X_test])
    y_test_tensor = tf.convert_to_tensor(y_test)

    # Balancing the Data
    # Calculate class weights
    class_weights = class_weight.compute_class_weight(
        class_weight = 'balanced',
        classes = np.unique(y_train),
        y = y_train
    )

    # Convert class_weights to a dictionary to include it in model.fit()
    class_weights = dict(enumerate(class_weights))

    # Fitting the model
    model.fit(
        x = X_train_tensor,
        y = y_train_tensor,
        validation_data = (X_test_tensor, y_test_tensor),
        steps_per_epoch= len(X_train) / 32, 
        epochs=32, 
        class_weight = class_weights
)