import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization
from tensorflow.keras.layers import UpSampling2D, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras import backend as K
from tf_notification_callback import TelegramCallback

telegram_callback = TelegramCallback(bot_token='1695094498:AAFfpt-dlsXAtg5KtExj98nqTV8aq5bjNCA', 
                                    chat_id='-492316955', 
                                    modelName='Model', 
                                    loss_metrics=['loss', 'val_loss'], 
                                    acc_metrics=['dsc', 'val_dsc'], 
                                    getSummary=False)


# Define constants & Setting directories for Train Test Split
SEED = 909
BATCH_SIZE_TRAIN = 2
BATCH_SIZE_TEST = 2
BATCH_SIZE = 2


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
def to_grayscale_then_rgb(image):
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.grayscale_to_rgb(image)
    return image

def create_segmentation_generator_train(img_path, msk_path, BATCH_SIZE):
    data_gen_args = dict(rescale=1./255,
                    featurewise_center=True,
                    featurewise_std_normalization=True,
                    rotation_range=90,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    zoom_range=0.3
                        )
    
    datagen = ImageDataGenerator(**data_gen_args, preprocessing_function=to_grayscale_then_rgb)
    
    # datagen.flow_from_directory() is very specific on file structure, check on keras documentation

    img_generator = datagen.flow_from_directory(img_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH), batch_size=BATCH_SIZE, class_mode=None, color_mode='rgb', seed=SEED)
    
    msk_generator = datagen.flow_from_directory(msk_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH), batch_size=BATCH_SIZE, class_mode=None, color_mode='rgb', seed=SEED)
    return zip(img_generator, msk_generator)

    # Create Train & Test Generator
train_generator = create_segmentation_generator_train(data_dir_train_image, data_dir_train_mask, BATCH_SIZE_TRAIN)

# Image Data Generator to Test Set without Data Augmentation
# Remember not to perform any image augmentation in the test generator!

def create_segmentation_generator_test(img_path, msk_path, BATCH_SIZE):
    data_gen_args = dict(rescale=1./255)
    datagen = ImageDataGenerator(**data_gen_args, preprocessing_function=to_grayscale_then_rgb)
    
    # datagen.flow_from_directory() is very specific on file structure, check on keras documentation

    img_generator = datagen.flow_from_directory(img_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH), class_mode=None, color_mode='rgb', batch_size=BATCH_SIZE, seed=SEED)
    msk_generator = datagen.flow_from_directory(msk_path, target_size=IMG_SIZE, class_mode=None, color_mode='rgb', batch_size=BATCH_SIZE, seed=SEED)
    return zip(img_generator, msk_generator)

test_generator = create_segmentation_generator_test(data_dir_test_image, data_dir_test_mask, BATCH_SIZE_TEST)

# Hyperparameters
IMAGE_SIZE = 256

NUM_TRAIN = 30094
NUM_TEST = 12891

EPOCHS = 100
BATCH = 2
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

show_dataset(train_generator, 10)

# Define Model

def model(in_channels=3, out_channels=3):
    inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, in_channels), name="input_image")
    
    encoder = MobileNetV2(input_tensor=inputs, weights='imagenet', include_top=False, alpha=0.35)
    # Set Trainable to False to freeze the encoder layers
    encoder.trainable = False
    skip_connection_names = ["input_image", "block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu"]
    encoder_output = encoder.get_layer("block_13_expand_relu").output
    
    f = [16, 32, 48, 64]
    x = encoder_output
    for i in range(1, len(skip_connection_names)+1, 1):
        x_skip = encoder.get_layer(skip_connection_names[-i]).output
        x = UpSampling2D((2, 2))(x)
        x = Concatenate()([x, x_skip])
        
        x = Conv2D(f[-i], (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        
        x = Conv2D(f[-i], (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        
    x = Conv2D(3, (1, 1), padding="same")(x)
    x = Activation("sigmoid")(x)
    
    model = Model(inputs, x)
    return model

model = model()
model.summary()

# Evaluation Metrics

from keras.losses import binary_crossentropy

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
access_token = '1695094498:AAFfpt-dlsXAtg5KtExj98nqTV8aq5bjNCA'
chat_id = '-492316955'

def telegram_bot_sendtext(bot_message):
    
    bot_token = access_token
    bot_chatID = chat_id
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message

    response = requests.get(send_text)

    return response.json()
    

msg = telegram_bot_sendtext("Model Training in Progress")

history = model.fit_generator(generator=train_generator, 
                    steps_per_epoch=EPOCH_STEP_TRAIN, 
                    validation_data=test_generator, 
                    validation_steps=EPOCH_STEP_TEST,
                   epochs=EPOCHS,
                   callbacks=callbacks)

# Training Done Send Message to telegram
msg_train = telegram_bot_sendtext("Model has completed training.")

# list all data in history
print(history.history.keys())

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('U-Net with MobileNet Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('/home/tester/kong/UNetMobileNetV2_with_Custom_Loss_NoFreeze_Loss.png')

# summarize history for DICE
plt.plot(history.history['dsc'])
plt.plot(history.history['val_dsc'])
plt.title('U-Net with MobileNet DICE Score')
plt.ylabel('DICE Score')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('/home/tester/kong/UNetMobileNetV2_with_Custom_Loss_NoFreeze_DICE.png')

def show_prediction(datagen, num=1):
    for i in range(0,num):
        image, mask = next(datagen) # Everytime will load another slice when function is called
        pred_mask = model.predict(image) 
        display([image[0], mask[0], pred_mask[0]])

show_prediction(test_generator, 10) # num will show number of slices to be displayed

# Save the pre-trained model 

model.save('/home/tester/kong/UNetMobileV2_FullData_CustomLoss_TF_NoFreeze_DataAugmentation')

# Model Saved. Send Message to telegram
msg_save = telegram_bot_sendtext("Trained Model is saved. Ending Session.")

