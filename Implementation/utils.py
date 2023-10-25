import os
import random
import numpy as np
import cv2
import tensorflow as tf
from keras.applications.resnet_v2 import preprocess_input, decode_predictions
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.initializers import random_uniform, glorot_uniform, constant, identity
from tensorflow.python.framework.ops import EagerTensor


def load_data(train_data_path, split_ratio=0.8, random_seed=0, shuffle=True, save=True):
    """
    Input: Takes path to training data folder as input
    Output: Returns X_train, X_val, y_train, y_val as numpy array(s) in this order
            For save = True, it also saves these numpy arrays in .npy format in current working directory
    Note: Annotations, bounding boxes are read but ignored for now, to be dealt with later on
        **All path formatting is done with Ubuntu 22.04 formatting in mind; subject to modifications with change in OS**
    """
    bleeding_path = train_data_path + "/bleeding"
    non_bleeding_path = train_data_path + "/non-bleeding"

    # Define the paths to the images and annotations
    bleeding_images_path = bleeding_path + "/Images"
    bleeding_annotations_path = bleeding_path + "/Annotations"
    bleeding_bounding_boxes_path = bleeding_path + "/Bounding boxes/YOLO_TXT"
    non_bleeding_images_path = non_bleeding_path + "/images"
    non_bleeding_annotations_path = non_bleeding_path + "/annotation"

    # Bleeding
    # Images paths
    image_list_bleeding_orig = os.listdir(bleeding_images_path)
    image_list_bleeding_orig = list(np.sort(image_list_bleeding_orig))
    image_list_bleeding = [bleeding_images_path + '/' + i for i in image_list_bleeding_orig]
    # Masks paths
    mask_list_bleeding_orig = os.listdir(bleeding_annotations_path)
    mask_list_bleeding_orig = list(np.sort(mask_list_bleeding_orig))
    mask_list_bleeding = [bleeding_annotations_path + '/' + i for i in mask_list_bleeding_orig]

    # Non-Bleeding
    # Images paths
    image_list_non_bleeding_orig = os.listdir(non_bleeding_images_path)
    image_list_non_bleeding_orig = list(np.sort(image_list_non_bleeding_orig))
    image_list_non_bleeding = [non_bleeding_images_path + '/' + i for i in image_list_non_bleeding_orig]
    # Masks paths
    mask_list_non_bleeding_orig = os.listdir(non_bleeding_annotations_path)
    mask_list_non_bleeding_orig = list(np.sort(mask_list_non_bleeding_orig))
    mask_list_non_bleeding = [bleeding_annotations_path + '/' + i for i in mask_list_non_bleeding_orig]

    # Train-test split
    random.seed(random_seed)
    random_indices = random.sample(range(len(image_list_bleeding)), len(image_list_bleeding))
    n_train_samples = int(len(image_list_bleeding) * split_ratio)
    image_list_bleeding_train = image_list_bleeding[:n_train_samples]
    image_list_bleeding_val = image_list_bleeding[n_train_samples:]
    image_list_non_bleeding_train = image_list_non_bleeding[:n_train_samples]
    image_list_non_bleeding_val = image_list_non_bleeding[n_train_samples:]

    # Load the images in numpy arrays
    # We use float16 for better memory management for the cost of accuracy
    X_train_b = load_images(image_list_bleeding_train).astype(np.float16)
    X_val_b = load_images(image_list_bleeding_val).astype(np.float16)
    X_train_nb = load_images(image_list_non_bleeding_train).astype(np.float16)
    X_val_nb = load_images(image_list_non_bleeding_val).astype(np.float16)

    num_train_samples = X_train_b.shape[0] + X_train_nb.shape[0]
    num_val_samples = X_val_b.shape[0] + X_val_nb.shape[0]

    # Create labels for the data
    y_train = np.concatenate((np.ones(num_train_samples // 2), np.zeros(num_train_samples // 2)))
    y_val = np.concatenate((np.ones(num_val_samples // 2), np.zeros(num_val_samples // 2)))

    shuffle_indices = np.arange(num_train_samples)

    if shuffle:
        # Shuffle the data and labels (if necessary)
        np.random.seed(random_seed)
        np.random.shuffle(shuffle_indices)

    X_train = np.concatenate((X_train_b, X_train_nb), axis=0)[shuffle_indices]
    X_val = np.concatenate((X_val_b, X_val_nb), axis=0)
    y_train = y_train[shuffle_indices].astype(np.float16)
    y_val = y_val.astype(np.float16)

    if save:
        np.save('X_train.npy', X_train)
        np.save('X_val.npy', X_val)
        np.save('y_train.npy', y_train)
        np.save('y_val.npy', y_val)

    return X_train, X_val, y_train, y_val


def load_images(image_list):
    """
    Input: Takes a list of paths to the images
    Output: Returns a single numpy array with all the image data in the list of image paths
    """
    images = []
    for img_path in image_list:
        img = cv2.imread(img_path)  # Load the image using OpenCV
        # Convert from BGR to RGB (In certain cases, OpenCV reads the images
        # in the BGR format instead of the standard RGB format)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (224, 224))  # Resize the image as required
        img = img / 255.0  # Normalize the pixel values
        images.append(img)

    return np.array(images)


def identity_block(X, f, filters, training=True, initializer=random_uniform):
    """
    Implementation of the identity block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    training -- True: Behave in training mode
                False: Behave in inference mode
    initializer -- to set up the initial weights of a layer. Equals to random uniform initializer

    Returns:
    X -- output of the identity block, tensor of shape (m, n_H, n_W, n_C)
    """

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters=F1, kernel_size=1, strides=(1, 1), padding='valid', kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X, training=training)  # Default axis
    X = Activation('relu')(X)

    ### START CODE HERE
    ## Second component of main path
    ## Set the padding = 'same'
    X = Conv2D(filters=F2, kernel_size=f, strides=(1, 1), padding='same', kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X, training=training)
    X = Activation('relu')(X)

    ## Third component of main path
    ## Set the padding = 'valid'
    X = Conv2D(filters=F3, kernel_size=1, strides=(1, 1), padding='valid', kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X, training=training)

    ## Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, s=2, training=True, initializer=glorot_uniform):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    s -- Integer, specifying the stride to be used
    training -- True: Behave in training mode
                False: Behave in inference mode
    initializer -- to set up the initial weights of a layer. Equals to Glorot uniform initializer,
                   also called Xavier uniform initializer.

    Returns:
    X -- output of the convolutional block, tensor of shape (m, n_H, n_W, n_C)
    """

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####

    # First component of main path glorot_uniform(seed=0)
    X = Conv2D(filters=F1, kernel_size=1, strides=(s, s), padding='valid', kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X, training=training)
    X = Activation('relu')(X)

    ### START CODE HERE

    ## Second component of main path (≈3 lines)
    X = Conv2D(filters=F2, kernel_size=f, strides=(1, 1), padding='same', kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X, training=training)
    X = Activation('relu')(X)

    ## Third component of main path (≈2 lines)
    X = Conv2D(filters=F3, kernel_size=1, strides=(1, 1), padding='valid', kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X, training=training)

    ##### SHORTCUT PATH ##### (≈2 lines)
    X_shortcut = Conv2D(filters=F3, kernel_size=1, strides=(s, s), padding='valid',
                        kernel_initializer=initializer(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3)(X_shortcut, training=training)

    ### END CODE HERE

    # Final step: Add shortcut value to main path (Use this order [X, X_shortcut]),
    # and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X