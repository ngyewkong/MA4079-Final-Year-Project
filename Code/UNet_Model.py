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
import keras.backend as K
from tf_notification_callback import TelegramCallback

telegram_callback = TelegramCallback(bot_token='INSERT YOUR OWN TELEGRAM BOT_TOKEN ID', 
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
    data_gen_args = dict(rescale=1./255
                         #featurewise_center=True,
                         #featurewise_std_normalization=True,
                         #rotation_range=90,
                         #width_shift_range=0.2,      
                         #height_shift_range=0.2,
                         #zoom_range=0.3
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

def unet(n_levels, initial_features=32, n_blocks=2, kernel_size=3, pooling_size=2, in_channels=1, out_channels=1):
    inputs = keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, in_channels))
    x = inputs
    
    # convolution parameters, activation 'rectified linear unit', same padding to ensure output shape is the same as input shape
    convpars = dict(kernel_size=kernel_size, activation='relu', padding='same')
    
    # downsampling
    skips = {}
    for level in range(n_levels):
        for _ in range(n_blocks):
            x = keras.layers.Conv2D(initial_features * 2 ** level, **convpars)(x)
        if level < n_levels - 1:
            skips[level] = x
            x = keras.layers.MaxPool2D(pooling_size)(x)
            
    # upsampling
    for level in reversed(range(n_levels-1)):
        x = keras.layers.Conv2DTranspose(initial_features * 2 ** level, strides=pooling_size, **convpars)(x)
        x = keras.layers.Concatenate()([x, skips[level]])
        for _ in range(n_blocks):
            x = keras.layers.Conv2D(initial_features * 2 ** level, **convpars)(x)
            
    # output
    activation = 'sigmoid' if out_channels == 1 else 'softmax'
    x = keras.layers.Conv2D(out_channels, kernel_size=1, activation=activation, padding='same')(x)
    
    unet_model = keras.Model(inputs=[inputs], outputs=[x], name=f'UNET-L{n_levels}-F{initial_features}')
    return unet_model
    
model = unet(4)
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

def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    smooth = 1
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)

def weighted_bce(y_true, y_pred):
    weights = (y_true * 59.) + 1.
    bce = K.binary_crossentropy(y_true, y_pred)
    weighted_bce = K.mean(bce * weights)
    return weighted_bce

def bce_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()
    loss = bce(y_true, y_pred)
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
access_token = '1695094498:AAFfpt-dlsXAtg5KtExj98nqTV8aq5bjNCA'
chat_id = '-492316955'

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
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('U-Net from Scratch with BCE Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('/home/tester/kong/U-Net_from_Scratch_with_BCEDiceLoss_Loss.png')

# summarize history for DICE
plt.plot(history.history['dsc'])
plt.plot(history.history['val_dsc'])
plt.title('U-Net from Scratch with BCE Loss DICE Score')
plt.ylabel('DICE Score')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('/home/tester/kong/U-Net_from_Scratch_with_BCEDiceLoss_DICE.png')

# Save the pre-trained model 

model.save('/home/tester/kong/UNetScratch_FullDataset_CustomLoss')

# Model Saved. Send Message to telegram
msg_save = telegram_bot_sendtext("Trained Model is saved. Ending Session.")
