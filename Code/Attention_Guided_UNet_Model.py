import os
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K
from keras.losses import binary_crossentropy
from keras import Model
from tf_notification_callback import TelegramCallback

from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, PReLU, UpSampling2D, concatenate , Reshape, Dense, Permute, MaxPool2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Activation, add, GaussianNoise, BatchNormalization, multiply

telegram_callback = TelegramCallback(bot_token= 'INSERT YOUR OWN TELEGRAM BOT_TOKEN ID', 
                                    chat_id='INSERT YOUR OWN CHAT ID', 
                                    modelName='Model', 
                                    loss_metrics=['loss', 'val_loss'], 
                                    acc_metrics=['dsc', 'val_dsc'], 
                                    getSummary=False)

# Define constants & Setting directories for Train Test Split
SEED = 909
BATCH = 2
BATCH_SIZE_TRAIN = 2
BATCH_SIZE_TEST = 2

# Size of 2D slice 512 by 512 downsize to 256 by 256 (OOM issue)
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMG_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

data_dir = '/home/tester/kong/slices/'

data_dir_train = os.path.join(data_dir, 'training')
# The images should be stored under: "data/slices/training/image/image"
data_dir_train_image = os.path.join(data_dir_train, 'image')
# The images should be stored under: "data/slices/training/mask/image"
data_dir_train_mask = os.path.join(data_dir_train, 'mask')

data_dir_test = os.path.join(data_dir, 'test')
# The images should be stored under: "data/slices/test/image/image"
data_dir_test_image = os.path.join(data_dir_test, 'image')
# The images should be stored under: "data/slices/test/mask/image"
data_dir_test_mask = os.path.join(data_dir_test, 'mask')

# Image Data Generator to Training Set with Data Augmentation

def create_segmentation_generator_train(img_path, msk_path, BATCH_SIZE):
    data_gen_args = dict(rescale=1./255,
                         featurewise_center=True,
                         featurewise_std_normalization=True,
                         rotation_range=90,
                         width_shift_range=0.2,      
                         height_shift_range=0.2,
                         zoom_range=0.3
                        )
    datagen = ImageDataGenerator(**data_gen_args)
    
    # datagen.flow_from_directory() is very specific on file structure, check on keras documentation
    
    img_generator = datagen.flow_from_directory(img_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH), class_mode=None, color_mode='grayscale', batch_size=BATCH_SIZE, seed=SEED)
    
    msk_generator = datagen.flow_from_directory(msk_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH), class_mode=None, color_mode='grayscale', batch_size=BATCH_SIZE, seed=SEED)
    return zip(img_generator, msk_generator)

# Create Train & Test Generator
train_generator = create_segmentation_generator_train(data_dir_train_image, data_dir_train_mask, BATCH_SIZE_TRAIN)

# Image Data Generator to Test Set without Data Augmentation
# Remember not to perform any image augmentation in the test generator!
def create_segmentation_generator_test(img_path, msk_path, BATCH_SIZE):
    data_gen_args = dict(rescale=1./255)
    datagen = ImageDataGenerator(**data_gen_args)
    
    # datagen.flow_from_directory() is very specific on file structure, check on keras documentation

    img_generator = datagen.flow_from_directory(img_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH), class_mode=None, color_mode='grayscale', batch_size=BATCH_SIZE, seed=SEED)
    msk_generator = datagen.flow_from_directory(msk_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH), class_mode=None, color_mode='grayscale', batch_size=BATCH_SIZE, seed=SEED)
    return zip(img_generator, msk_generator)

test_generator = create_segmentation_generator_test(data_dir_test_image, data_dir_test_mask, BATCH_SIZE_TEST)

# Parameters Placeholders

NUM_TRAIN = 30094
NUM_TEST = 12891
NUM_OF_EPOCHS = 100

LR = 3e-4

def display(display_list):
    plt.figure(figsize=(15,15)) 
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]), cmap='gray')
    plt.show()

def show_dataset(datagen, num=1):
    for i in range(0,num):
        image,mask = next(datagen)
        display([image[0], mask[0]])

show_dataset(train_generator, 10) # num will display the number of sets as set by the parameter; default is 1

# U-Net model
def unet_model(input_shape, modified_unet=True, start_channel=64, 
               number_of_levels=4, inc_rate=2, output_channels=1):
    """
    Builds UNet model
    
    Parameters
    ----------
    input_shape : tuple
        Shape of the input data (height, width, channel)
    modified_unet : bool
        Whether to use modified UNet or the original UNet
    learning_rate : float
        Learning rate for the model. The default is 0.01.
    start_channel : int
        Number of channels of the first conv. The default is 64.
    number_of_levels : int
        The depth size of the U-structure. The default is 3.
    inc_rate : int
        Rate at which the conv channels will increase. The default is 2.
    output_channels : int
        The number of output layer channels. The default is 4
    saved_model_dir : str
        If spesified, the model weights will be loaded from this path. The default is None.
    Returns
    -------
    model : keras.model
        The created keras model with respect to the input parameters
    """

        
    input_layer = Input(shape=input_shape, name='the_input_layer')

    if modified_unet:
        x = GaussianNoise(0.01, name='Gaussian_Noise')(input_layer)
        x = Conv2D(32, 3, padding='same')(x)
        x = level_block_modified(x, start_channel, number_of_levels, inc_rate)
        x = BatchNormalization(axis = -1)(x)
        x = PReLU(shared_axes=[1, 2])(x)
    else: 
        x = level_block(input_layer, start_channel, number_of_levels, inc_rate)

    x            = Conv2D(output_channels, 1, padding='same')(x)
    output_layer = Activation('sigmoid')(x)
    
    model        = keras.Model(inputs = input_layer, outputs = output_layer)
    
    return model


def se_block(x, ratio=16):
    
    """
    creates a squeeze and excitation block
    https://arxiv.org/abs/1709.01507
    
    Parameters
    ----------
    x : tensor
        Input keras tensor
    ratio : int
        The reduction ratio. The default is 16.
    Returns
    -------
    x : tensor
        A keras tensor
    """
 

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = x.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(x)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([x, se])
    return x


def level_block(x, dim, level, inc):
    
    if level > 0:
        m = conv_layers(x, dim)
        x = MaxPool2D(pool_size=(2, 2))(m)
        x = level_block(x,int(inc*dim), level-1, inc)
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(dim, 2, padding='same')(x)
        m = concatenate([m,x])
        x = conv_layers(m, dim)
    else:
        x = conv_layers(x, dim)
    return x


def level_block_modified(x, dim, level, inc):
    
    if level > 0:
        m = res_block(x, dim, encoder_path=True)##########
        x = Conv2D(int(inc*dim), 2, strides=2, padding='same')(m)
        x = level_block_modified(x, int(inc*dim), level-1, inc)

        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(dim, 2, padding='same')(x)

        m = concatenate([m,x])
        m = se_block(m, 8)
        x = res_block(m, dim, encoder_path=False)
    else:
        x = res_block(x, dim, encoder_path=True) #############
    return x


def conv_layers(x, dim):

    x = Conv2D(dim, 3, padding='same')(x)
    x = Activation("relu")(x)

    x = Conv2D(dim, 3, padding='same')(x)
    x = Activation("relu")(x)

    return x

def res_block(x, dim, encoder_path=True):

    m = BatchNormalization(axis = -1)(x)
    m = PReLU(shared_axes = [1, 2])(m)
    m = Conv2D(dim, 3, padding='same')(m)

    m = BatchNormalization(axis = -1)(m)
    m = PReLU(shared_axes = [1, 2])(m)
    m = Conv2D(dim, 3, padding='same')(m)

    if encoder_path:
        x = add([x, m])
    else:
        x = Conv2D(dim, 1, padding='same', use_bias=False)(x)
        x = add([x,m])
    return  x

input_shape = (256, 256, 1)

model = unet_model(input_shape, modified_unet=True, start_channel=32, 
               number_of_levels=3, inc_rate=2, output_channels=1)
model.summary()

# Evaluation Metrics

epsilon = 1e-5

def dsc(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dsc(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

EPOCH_STEP_TRAIN = NUM_TRAIN//BATCH
EPOCH_STEP_TEST = NUM_TEST//BATCH

if NUM_TRAIN % BATCH != 0:
    EPOCH_STEP_TRAIN += 1
if NUM_TEST % BATCH != 0:
    EPOCH_STEP_TEST += 1

opt = tf.keras.optimizers.Nadam(LR)
metrics = [dsc]
model.compile(loss=bce_dice_loss, optimizer=opt, metrics=metrics)

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False),
    telegram_callback
]

import requests
access_token = 'INSERT YOUR OWN ACCESS TOKEN'
chat_id = 'INSERT YOUR OWN CHAT ID'

def telegram_bot_sendtext(bot_message):
    bot_token = access_token
    bot_chatID = chat_id
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message

    response = requests.get(send_text)

    return response.json()
    

msg = telegram_bot_sendtext("Model Training in Progress")

# Fit Model
history = model.fit_generator(generator=train_generator, 
                    steps_per_epoch=EPOCH_STEP_TRAIN, 
                    validation_data=test_generator, 
                    validation_steps=EPOCH_STEP_TEST,
                   epochs=NUM_OF_EPOCHS,
                   callbacks=callbacks)

# Training Done Send Message to telegram
msg_train = telegram_bot_sendtext("Model has completed training.")

# list all data in history
print(history.history.keys())

# summarize history for loss
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.subplot(1, 2, 1)
plt.plot(epochs, loss, 'bo', label = 'Training Loss')
plt.plot(epochs, val_loss, 'b', label = 'Test Loss')
plt.title('Attention Guided U-Net with Custom Loss with Data Augmentation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
plt.savefig('/home/tester/kong/Attention_Guided_U-Net_with_Custom_Loss_with_DataAugment_Loss.png')

# summarize history for DICE
dsc = history.history['dsc']
val_dsc = history.history['val_dsc']
plt.subplot(1, 2, 2)
plt.plot(epochs, dsc, 'bo', label = 'Training DICE')
plt.plot(epochs, val_dsc, 'b', label = 'Test DICE')
plt.title('Attention Guided U-Net with Custom Loss with Data AUgmentation DICE Score')
plt.ylabel('DICE Score')
plt.xlabel('Epoch')
plt.legend()
plt.show()
plt.savefig('/home/tester/kong/Attention_Guided_U-Net_with_Custom_Loss_with_DataAugment_DICE.png')

# Save the pre-trained model 

model.save('/home/tester/kong/Attention_Guided_UNet_FullDataset_CustomLoss_with_DataAugmentation')

# Model Saved. Send Message to telegram
msg_save = telegram_bot_sendtext("Trained Model is saved. Ending Session.")
