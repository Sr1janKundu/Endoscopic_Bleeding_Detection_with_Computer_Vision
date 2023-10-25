import tensorflow as tf
from keras import Input
from keras.layers import (InputLayer, Add, Dense, Activation, ZeroPadding2D, BatchNormalization,
                          Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,
                          GlobalAveragePooling2D, Dropout)
from keras.initializers import random_uniform, glorot_uniform, constant, identity
from keras.models import Sequential, Model, load_model
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
import utils


def create_model1(image_shape):
    """
    Creates a simple Convolutional Model
    Input Shape:
    Architecture:

    """
    cnn = Sequential()
    # Input Layer
    cnn.add(InputLayer(input_shape=image_shape, name='Input_Layer'))

    # 1st Conv + Maxpooling
    cnn.add(Conv2D(32, (5, 5), padding='same', activation='relu', name='C1'))
    cnn.add(MaxPooling2D(pool_size=(2, 2), name='MP1'))
    # Dropout
    cnn.add(Dropout(0.1, name='DropOut1'))

    # 2nd Conv + Maxpooling
    cnn.add(Conv2D(64, (5, 5), activation='relu', name='C2'))
    cnn.add(MaxPooling2D(pool_size=(2, 2), name='MP2'))
    # Dropout
    cnn.add(Dropout(0.2, name='DropOut2'))

    # 3rd Conv + Maxpooling
    cnn.add(Conv2D(128, (5, 5), activation='relu', name='C3'))
    cnn.add(MaxPooling2D(pool_size=(2, 2), name='MP3'))
    # Dropout
    cnn.add(Dropout(0.3, name='DropOut3'))

    # Converting 3D feature to 1D feature Vector
    cnn.add(Flatten(name='Dense1'))

    # Fully Connected Layer
    cnn.add(Dense(128, activation='relu', name='Dense2'))

    # Add the final output layer
    cnn.add(Dense(units=1, activation='sigmoid', name='Sigmoid_Output'))

    return cnn


def create_model2(input_shape):
    """
    Creates a transfer learning model based on MobileNetV2
    Arguments: image_shape -- Image width, height, no. of channels as a set
    Returns: tf.keras.model
    """
    # Import the mobilenet_v2 model architecture
    base_model = MobileNetV2(input_shape=(224, 224, 3),
                             include_top=False,
                             alpha=1.0,
                             weights='imagenet')
    # freeze the base model by making it non-trainable
    base_model.trainable = False

    # Initiate the model
    inputs = tf.keras.Input(shape=input_shape)
    x = utils.data_augmenter()(inputs)
    x = preprocess_input(x)

    # Add Base Mobilenet_v2 Model
    x = base_model(x, training=False)

    # Add the new Binary classification layers
    # Use global avg pooling to summarize the info in each channel
    x = GlobalAveragePooling2D()(x)

    # Include dropout with probability of 0.2 to avoid overfitting
    x = Dropout(0.2)(x)

    # Use a prediction layer with one neuron (as a binary classifier only needs one)
    prediction_layer = Dense(1)

    # Add the final output layer
    outputs = prediction_layer(x)

    model = tf.keras.Model(inputs, outputs)

    return model


def create_resnet50(input_shape, classes):
    """
    Stage-wise implementation of the architecture of the popular ResNet50:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> FLATTEN -> DENSE

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = utils.convolutional_block(X, f=3, filters=[64, 64, 256], s=1)
    X = utils.identity_block(X, 3, [64, 64, 256])
    X = utils.identity_block(X, 3, [64, 64, 256])

    ### START CODE HERE

    # Use the instructions above in order to implement all of the Stages below
    # Make sure you don't miss adding any required parameter

    ## Stage 3 (≈4 lines)
    # `convolutional_block` with correct values of `f`, `filters` and `s` for this stage
    X = utils.convolutional_block(X, f=3, filters=[128, 128, 512], s=2)
    # the 3 `identity_block` with correct values of `f` and `filters` for this stage
    X = utils.identity_block(X, 3, [128, 128, 512])
    X = utils.identity_block(X, 3, [128, 128, 512])
    X = utils.identity_block(X, 3, [128, 128, 512])

    # Stage 4 (≈6 lines)
    # add `convolutional_block` with correct values of `f`, `filters` and `s` for this stage
    X = utils.convolutional_block(X, f=3, filters=[256, 256, 1024], s=2)
    # the 5 `identity_block` with correct values of `f` and `filters` for this stage
    X = utils.identity_block(X, 3, [256, 256, 1024])
    X = utils.identity_block(X, 3, [256, 256, 1024])
    X = utils.identity_block(X, 3, [256, 256, 1024])
    X = utils.identity_block(X, 3, [256, 256, 1024])
    X = utils.identity_block(X, 3, [256, 256, 1024])

    # Stage 5 (≈3 lines)
    # add `convolutional_block` with correct values of `f`, `filters` and `s` for this stage
    X = utils.convolutional_block(X, f=3, filters=[512, 512, 2048], s=2)
    # the 2 `identity_block` with correct values of `f` and `filters` for this stage
    X = utils.identity_block(X, 3, [512, 512, 2048])
    X = utils.identity_block(X, 3, [512, 512, 2048])

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D()(X)"
    X = AveragePooling2D((2, 2))(X)

    ### END CODE HERE

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', kernel_initializer=glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs=X_input, outputs=X)

    return model
