"""
    Source file name: architectures_segmentation.py  
    
    Description: this file contains the code to perform data augmentation, neural network training and testing, annotation assistance
    
    Copyright (C) <2020>  <Vito Paolo Pastore, Matteo Moro, Francesca Odone>
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 3 of the License.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import os
import sys
import random
import warnings
import copy
import wx
import cv2
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
import pandas as pd
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input,Conv2D, Conv2DTranspose,Dropout, Lambda,MaxPooling2D,concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.compat.v1.keras import backend as K
import threading


class unet():

    def __init__(self,architecture, backbone,image_net,single_labels,colors,address,annotation_file,image_folder,annotation_folder,bodyparts,BATCH_SIZE = 5,train_flag=1,annotation_assistance = 0,lr = 0.001,loss = "Weighted Categorical_cross_entropy",markersize=13):

        self.lr_drop = 10


        self.loss_function = loss

        self.markerSize = int(markersize)

        self.BATCH_SIZE = int(BATCH_SIZE)  # the higher the better
        self.annotation_file = os.path.join(address,annotation_file)
        self.bodyparts = bodyparts
        self.colors = colors
        self.error = 0
        self.single_labels = single_labels
        self.architecture = architecture
        self.backbone = backbone
        self.image_net = image_net
        self.annotation_assistance = annotation_assistance
        self.learning_rate = float(lr)
        self.train_flag = train_flag
        self.num_bodyparts = len(self.bodyparts)
        self.image_folder = image_folder
        self.annotation_folder = annotation_folder
        self.address = address

        try:
            file = open(self.address + os.sep + 'annotation_options.txt')
            self.options = file.readlines()
        except:
            wx.MessageBox('Error in reading the annotation_options, please check the existence or choose'
                          ' annotation_options\n '
                          'annotation_options reading error '
                          , 'Error!', wx.OK | wx.ICON_ERROR)
            return
        try:
            configuration_file = self.address + os.sep + 'file_configuration.txt'
            file = open(configuration_file)
            self.pref = file.readlines()
        except:
            wx.MessageBox('Error in reading the  configuration file, please check the address or existence'
                          'configuration file \n '
                          'configuration file_error'
                          , 'Error!', wx.OK | wx.ICON_ERROR)
            self.error = 1
            return

        self.scorer = self.options[1][:-1]

        self.number_videos = len(self.pref)-2

        for i in range(0, self.number_videos):
            ind = self.pref[2+i].rfind(os.sep)
            if self.pref[2+i][ind+1:-1] in self.annotation_file:
                self.name_video_list = self.pref[2+i][ind+1:-1]
                break
        try:
            self.dataFrame = pd.read_pickle(self.annotation_file)
        except:
            wx.MessageBox('Annotations reading error\n '
                          'Annotation not found'
                          , 'Error!', wx.OK | wx.ICON_ERROR)
            self.error = 1
            return

        self.annotated = np.where(np.bitwise_and((np.isnan(self.dataFrame.iloc[:, 0].values) == False),
                                                     self.dataFrame.iloc[:, 0].values > 0) == True)[0]
        if train_flag==1:
            self.train()

        if self.error!=1:
            self.test()



    def weighted_categorical_crossentropy(self,weights):

        weights = K.variable(weights)

        def loss(y_true, y_pred):
            # scale predictions so that the class probas of each sample sum to 1
            y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
            # clip to prevent NaN's and Inf's
            y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
            # calc
            loss = y_true * K.log(y_pred) * weights
            loss = -K.sum(loss, -1)
            return loss

        return loss

    def train(self):

        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        # config = tf.ConfigProto(gpu_options=gpu_options)
        session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True,per_process_gpu_memory_fraction=0.6))
        session = tf.Session(config=session_config)
    
        addresss = self.address
  


        warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
        seed = 42

        np.random.seed(10)

        print('Getting and resizing train images and masks ... ')
        sys.stdout.flush()
        counter = 0

        self.IMG_WIDTH = 288  # for faster computing on kaggle
        self.IMG_HEIGHT = 288  # for faster computing on kaggle

        counter = 0

        files_original_name = list()


        #self.num_bodyparts =1

        if len(self.annotated)==0:
            wx.MessageBox('Did you save your annotation?\n '
                          'No annotation found in your file, please save and re-run'
                          , 'Error!', wx.OK | wx.ICON_ERROR)
            self.error = 1
            return

        for i in range(0, len(self.annotated)):
            files_original_name.append(self.dataFrame[self.dataFrame.columns[0]]._stat_axis[self.annotated[i]][7:])

        img = imread(self.image_folder + os.sep + files_original_name[0])


        IMG_CHANNELS = len(np.shape(img))
        self.IMG_CHANNELS = IMG_CHANNELS

        self.file_name_for_prediction_confidence = files_original_name;

        X_train = np.zeros((len(self.annotated), self.IMG_HEIGHT, self.IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
        Y_train = np.zeros((len(self.annotated), self.IMG_HEIGHT, self.IMG_WIDTH, self.num_bodyparts + 1), dtype=np.int)
        New_train = np.zeros((len(self.annotated), self.IMG_HEIGHT, self.IMG_WIDTH), dtype=np.int)

        for l in range(0,len(self.annotated)):
            img = imread(self.image_folder + os.sep + files_original_name[l])

            # mask_ = np.zeros((np.shape(img)[0],np.shape(img)[1],self.num_bodyparts))
            mask_ = np.zeros((np.shape(img)[0],np.shape(img)[1],self.num_bodyparts))
            img = resize(img, (self.IMG_HEIGHT, self.IMG_WIDTH), mode='constant', preserve_range=True)

            X_train[counter] = img


            for j in range(0,self.num_bodyparts):
                mask_single_label = np.zeros((mask_.shape[0], mask_.shape[1]))

                #if annotation was assisted, x is negative

                points = np.asarray([self.dataFrame[self.dataFrame.columns[j*2]].values[self.annotated[l]],
                                     self.dataFrame[self.dataFrame.columns[j*2 + 1]].values[self.annotated[l]]], dtype=float)
                points = np.abs(points)

                if  np.isnan(points[0]):
                    continue

                cv2.circle(mask_single_label, (int(round((points[0] * (2 ** 4)))), int(round(points[1] * (2 ** 4)))),
                           int(round(self.markerSize * (2 ** 4))), (255, 255, 255), thickness=-1, shift=4)
                mask_[:,:,j] = mask_single_label

            mask_ = resize(mask_, (self.IMG_HEIGHT, self.IMG_WIDTH), mode='constant', preserve_range=True)
            a,mask_ = cv2.threshold(mask_,200,255,cv2.THRESH_BINARY)
            mask_=mask_/255.0
            temp = np.sum(mask_,axis=2)

            for j in range(0, self.num_bodyparts):
                New_train[counter] = New_train[counter] + mask_[:, :, j]* (j + 1)

            # alternative method to build the ground truth
            # temp = temp + 1
            # temp[temp == 0] = 1
            # temp[temp > 1] = 0
            # Y_train[counter, :, :,1:] = mask_
            # Y_train[counter,:,:,0] = temp
            counter += 1
            #

        try:
            Y_train = tf.keras.utils.to_categorical(New_train, num_classes=self.num_bodyparts + 1)
        except:
            wx.MessageBox('two or more labels are overlapping!\n '
                          'Check annotation or re-perform the labeling operation'
                          , 'Error!', wx.OK | wx.ICON_ERROR)
            self.error = 1
            return

        counter = 0
        
        from segmentation_models import get_preprocessing

        self.processer = get_preprocessing(self.backbone)

        X_train = self.processer(X_train)

        print('Done!')

        from tensorflow.keras.preprocessing import image

        # Creating the training Image and Mask generator
        image_datagen = image.ImageDataGenerator(shear_range=0.5, rotation_range=50, zoom_range=0.2, width_shift_range=0.2,
                                                 height_shift_range=0.2, fill_mode='reflect')
        mask_datagen = image.ImageDataGenerator(shear_range=0.5, rotation_range=50, zoom_range=0.2, width_shift_range=0.2,
                                                height_shift_range=0.2, fill_mode='reflect')

        # Keep the same seed for image and mask generators so they fit together

        image_datagen.fit(X_train[:int(X_train.shape[0] * 0.9)], augment=True, seed=seed)
        mask_datagen.fit(Y_train[:int(Y_train.shape[0] * 0.9)], augment=True, seed=seed)

        x = image_datagen.flow(X_train[:int(X_train.shape[0] * 0.9)], batch_size=self.BATCH_SIZE, shuffle=True, seed=seed)
        y = mask_datagen.flow(Y_train[:int(Y_train.shape[0] * 0.9)], batch_size=self.BATCH_SIZE, shuffle=True, seed=seed)

        # Creating the validation Image and Mask generator
        image_datagen_val = image.ImageDataGenerator()
        mask_datagen_val = image.ImageDataGenerator()

        image_datagen_val.fit(X_train[int(X_train.shape[0] * 0.9):], augment=True, seed=seed)
        mask_datagen_val.fit(Y_train[int(Y_train.shape[0] * 0.9):], augment=True, seed=seed)

        x_val = image_datagen_val.flow(X_train[int(X_train.shape[0] * 0.9):], batch_size=self.BATCH_SIZE, shuffle=True,
                                       seed=seed)
        y_val = mask_datagen_val.flow(Y_train[int(Y_train.shape[0] * 0.9):], batch_size=self.BATCH_SIZE, shuffle=True, seed=seed)

        train_generator = zip(x, y)
        val_generator = zip(x_val, y_val)

        from segmentation_models import Unet,PSPNet,Linknet,FPN
        from segmentation_models.losses import CategoricalFocalLoss
        from segmentation_models.utils import set_trainable
        import segmentation_models
        from tensorflow.keras.optimizers import RMSprop,SGD


        if self.architecture=='Linknet':

            self.model = Linknet(self.backbone, classes = self.num_bodyparts + 1, activation='softmax',encoder_weights=self.image_net,input_shape = (self.IMG_WIDTH,self.IMG_HEIGHT,self.IMG_CHANNELS))

        elif  self.architecture=='unet':

            self.model = Unet(self.backbone, classes = self.num_bodyparts + 1, activation='softmax',encoder_weights=self.image_net,input_shape = (self.IMG_WIDTH,self.IMG_HEIGHT,self.IMG_CHANNELS))
        elif  self.architecture=='PSPnet':
            self.model = PSPNet(self.backbone, classes = self.num_bodyparts + 1, activation='softmax',encoder_weights=self.image_net,input_shape = (self.IMG_WIDTH,self.IMG_HEIGHT,self.IMG_CHANNELS))

        elif self.architecture=='FPN':
            self.model = FPN(self.backbone, classes = self.num_bodyparts + 1, activation='softmax',encoder_weights=self.image_net,input_shape = (self.IMG_WIDTH,self.IMG_HEIGHT,self.IMG_CHANNELS))

        weights = np.zeros((1, self.num_bodyparts + 1), dtype=float)
        weight = 1.0 / self.num_bodyparts
        num_zeros = 1
        while (weight * 10 < 1):
            weight = weight * 10
            num_zeros += 1
        weight = int(weight * 10) / np.power(10, num_zeros)
        weights[0, 1:] = weight
        weights[0, 0] = 1 - np.sum(weights[0, 1:])
        weights = weights[0]

        if self.loss_function=="Weighted Categorical_cross_entropy":
            loss = self.weighted_categorical_crossentropy(weights)
        else:
            loss = segmentation_models.losses.DiceLoss(class_weights=weights)
        metric = segmentation_models.metrics.IOUScore(class_weights=weights,per_image=True)
        self.model.compile(optimizer = RMSprop(lr=self.learning_rate), loss=loss,metrics=[metric])
        earlystopper = EarlyStopping(patience=6, verbose=1)
        #
        checkpointer = ModelCheckpoint(os.path.join(self.address , 'Unet.h5'), verbose=1, save_best_only=True)
        reduce_lr = keras.callbacks.LearningRateScheduler(self.lr_scheduler)

        #
        # model.fit_generator(train_generator, validation_data=val_generator, validation_steps=10, steps_per_epoch=50,
        #                                epochs=2, callbacks=[earlystopper, checkpointer],verbose=1)
        # model.load_weights(self.address + 'Temp_weights.h5')

        # set_trainable(model)
        #
        self.model.fit_generator(train_generator, validation_data=val_generator, steps_per_epoch=20,validation_steps=5,
                                       epochs=100, callbacks=[earlystopper, checkpointer, reduce_lr], verbose=1)

    def test(self):
        import segmentation_models
        from segmentation_models import Unet
        session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.6))
        session = tf.Session(config=session_config)
        weights = np.zeros((1, self.num_bodyparts + 1), dtype=float)
        weight = 1.0 / self.num_bodyparts
        num_zeros = 1
        while (weight * 10 < 1):
            weight = weight * 10
            num_zeros += 1
        weight = int(weight * 10) / np.power(10, num_zeros)
        weights[0, 1:] = weight
        weights[0, 0] = 1 - np.sum(weights[0, 1:])
        weights = weights[0]
        if self.loss_function=="Weighted Categorical_cross_entropy":
            loss = self.weighted_categorical_crossentropy(weights)
        else:
            loss = segmentation_models.losses.DiceLoss(class_weights=weights)

        metric = segmentation_models.metrics.IOUScore(class_weights=weights,per_image=True)

        model = load_model(os.path.join(self.address , 'Unet.h5'),
                           custom_objects={'loss': loss, 'dice_loss': segmentation_models.losses.DiceLoss,
                                          'iou_score':metric})

        OUTPUT = os.path.join(self.address, self.name_video_list + '_prediction')

        try:
            os.mkdir(OUTPUT)

        except:
            pass
        files = os.listdir(self.image_folder)
        files_original_name = list()

        self.annotated = np.where(np.bitwise_and((np.isnan(self.dataFrame.iloc[:, 0].values) == False),
                                                 self.dataFrame.iloc[:, 0].values > 0) == True)[0]

        # if len(self.annotated)==0:
        #     wx.MessageBox('Did you save your annotation?\n '
        #                   'No annotation found in your file, please save and re-run'
        #                   , 'Error!', wx.OK | wx.ICON_ERROR)
        #     self.error = 1
        #     return

        for i in range(0, len(self.annotated)):
            files_original_name.append(files[self.annotated[i]])
        #files = np.unique(files_original_name)
        img = imread(self.image_folder + os.sep + files[0])

        IMG_CHANNELS = len(np.shape(img))
        self.IMG_CHANNELS = IMG_CHANNELS
        self.IMG_WIDTH = 288
        self.IMG_HEIGHT = 288
        counter = 0

        if self.annotation_assistance==0:



            self.X_test = np.zeros((len(files)- len(self.annotated), self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS), dtype=np.uint8)

            self.testing_index = list()

            for l in range(0, len(files)):
                if l not in self.annotated:
                    self.testing_index.append(l)
                    img = imread(self.image_folder + os.sep + files[l])[:, :, :self.IMG_CHANNELS]
                    img = resize(img, (self.IMG_HEIGHT, self.IMG_WIDTH), mode='constant', preserve_range=True)
                    self.X_test[counter] = img
                    counter += 1

            self.testing_index = np.asarray(self.testing_index)

        else:
            if os.path.isfile(os.path.join(os.path.dirname(self.annotation_file), self.name_video_list + '_index_annotation.txt')):
                self.pref_ann = open(os.path.join(os.path.dirname(self.annotation_file),self.name_video_list + '_index_annotation_auto.txt'), 'r')
            temporary = self.pref_ann.readlines()
            for i in range(0, len(temporary)):
                temporary[i] = temporary[i][:-1]
            temporary = np.asarray(temporary)
            self.frame_selected_for_annotation = temporary.astype(int)
            self.frame_selected_for_annotation=np.sort(self.frame_selected_for_annotation)

            if len(self.frame_selected_for_annotation)==0:
                filename = self.ask()
                if filename=="":
                    return
                else:
                    num_frame_automatic = int(filename)
                    if num_frame_automatic>len(files):
                        wx.MessageBox('Error! Number of frames must be lower than the total number of frames\n '
                                      'Input error'
                                      , 'Error!', wx.OK | wx.ICON_ERROR)
                    my_list = list(range(0,
                                         len(files)))  # list of integers from 1 to end                # adjust this boundaries to fit your needs
                    random.shuffle(my_list)
                    self.frame_selected_for_annotation = my_list[0:num_frame_automatic]
                    output = open(os.path.join(os.path.dirname(self.annotation_file),
                                                      self.name_video_list + '_index_annotation_auto.txt'), 'w')
                    for fra in self.frame_selected_for_annotation:
                        output.writelines(str(fra))
                        output.writelines('\n')
                    output.close()


            self.X_test = np.zeros((len(self.frame_selected_for_annotation), self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS), dtype=np.uint8)

            for l in range(0,len(files)):
                if l in self.frame_selected_for_annotation:
                    img = imread(self.image_folder + os.sep + files[l])[:, :, :self.IMG_CHANNELS]
                    img = resize(img, (self.IMG_HEIGHT, self.IMG_WIDTH), mode='constant', preserve_range=True)
                    self.X_test[counter] = img
                    counter += 1

        #X_test = X_test / 255.0
        seed = 42
        # Creating the training Image and Mask generator
        # image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        # # Keep the same seed for image and mask generators so they fit together
        # image_datagen.fit(self.X_test, augment=False,seed=seed)
        # x = image_datagen.flow(self.X_test, batch_size=1, shuffle=False, seed=seed)

        from segmentation_models import get_preprocessing

        self.processer = get_preprocessing(self.backbone)

        self.X_test = self.processer(self.X_test)


        x = self.X_test
        preds_test = model.predict(x, verbose=0)
        # Threshold predictions
        # preds_train_t = (preds_train > 0.5).astype(np.uint8)
        # K.clear_session()
        # preds_val_t = (preds_val > 0.5).astype(np.uint8)
        #preds_test_t = (preds_test > 0.5).astype(np.uint8)
        preds_test_t = preds_test
        # Create list of upsampled test masks
        preds_test_upsampled = []
        address_single_labels = os.path.join(self.address, self.name_video_list + '_Single_labels')
        self.annotated = np.where(np.bitwise_and((np.isnan(self.dataFrame.iloc[:, 0].values) == False),
                                                 self.dataFrame.iloc[:, 0].values > 0) == True)[0]
        try:
            os.mkdir(address_single_labels)
        except:
            pass

        if self.single_labels == 'Yes':

            if self.annotation_assistance == 1:
                self.testing_index = self.frame_selected_for_annotation
            # PREDICT AND SAVE SINGLE LABELS IMAGES
            for i in range(0, len(preds_test)):
                img = imread(self.image_folder + os.sep + files[self.testing_index[i]])
                sizes_test = np.shape(img)[:-1]
                for j in range(0, self.num_bodyparts + 1):
                    preds_test_upsampled = resize(np.squeeze(preds_test_t[i, :, :, j]),
                                                  (sizes_test[0], sizes_test[1]),
                                                  mode='constant', preserve_range=True)
                    preds_test_upsampled = (preds_test_upsampled * 255).astype(int)
                    cv2.imwrite(address_single_labels + os.sep + ("{:02d}".format(j)) + files[self.testing_index[i]], preds_test_upsampled)

        #let us create a dataframe to store prediction confidence

        imlist = os.listdir(self.image_folder)
        self.index = np.sort(imlist)
        a = np.empty((len(self.index), 3,))
        self.dataFrame3 = None
        a[:] = np.nan
        self.scorer = 'user'
        for bodypart in self.bodyparts:
            index = pd.MultiIndex.from_product([[self.scorer], [bodypart], ['x', 'y', 'confidence']],
                                               names=['scorer', 'bodyparts', 'coords'])
            frame = pd.DataFrame(a, columns=index, index=imlist)
            self.dataFrame3 = pd.concat([self.dataFrame3, frame], axis=1)
        num_columns = len(self.dataFrame3.columns)


        # if this is a regular testing, all of the images (but the annotated ones) have to be analyzed
        if self.annotation_assistance == 0:
            for i in range(0, len(preds_test)):

                results = np.zeros((self.num_bodyparts*2))
                results_plus_conf = np.zeros((self.num_bodyparts*3))

                #here to update dataframe with annotations
                img = imread(self.image_folder + os.sep + files[self.testing_index[i]])
                sizes_test = np.shape(img)[:-1]
                for j in range(0,self.num_bodyparts+1):
                    if j==0: continue

                    preds_test_upsampled = resize(np.squeeze(preds_test_t[i,:,:,j]),
                                                       (sizes_test[0], sizes_test[1]),
                                                       mode='constant', preserve_range=True)
                    preds_test_upsampled= (preds_test_upsampled * 255).astype(np.uint8)
                    results[(j - 1) * 2:(j - 1) * 2 + 2]= self.prediction_to_annotation(preds_test_upsampled)
                    results_plus_conf[(j - 1) * 3:(j - 1) * 3 + 3] = self.compute_confidence(preds_test_upsampled)
                    self.plot_annotation(img,results,files[self.testing_index[i]],OUTPUT)
                    self.dataFrame[self.dataFrame.columns[(j - 1) * 2]].values[self.testing_index[i]] = -results[(j - 1) * 2]
                    self.dataFrame[self.dataFrame.columns[(j - 1) * 2 + 1]].values[self.testing_index[i]] = results[(j - 1) * 2 + 1]
                    self.dataFrame3[self.dataFrame3.columns[(j - 1) * 3]].values[self.testing_index[i]] = -results_plus_conf[(j - 1) * 3]
                    self.dataFrame3[self.dataFrame3.columns[(j - 1) * 3 + 1]].values[self.testing_index[i]] = results_plus_conf[
                        (j - 1) * 3 + 1]
                    self.dataFrame3[self.dataFrame3.columns[(j - 1) * 3 + 2]].values[self.testing_index[i]] = results_plus_conf[
                        (j - 1) * 3 + 2]

        else:
            #if annotation assistance is requested, we only want to annotate the random frames extracted by the user
            for i in range(0, len(preds_test)):
                results = np.zeros((self.num_bodyparts * 2))
                results_plus_conf = np.zeros((self.num_bodyparts * 3))
                # here to update dataframe with annotations
                img = imread(self.image_folder + files[self.frame_selected_for_annotation[i]])
                sizes_test = np.shape(img)[:-1]
                for j in range(0, self.num_bodyparts + 1):
                    if j == 0: continue

                    preds_test_upsampled = resize(np.squeeze(preds_test_t[i, :, :, j]),
                                                  (sizes_test[0], sizes_test[1]),
                                                  mode='constant', preserve_range=True)
                    preds_test_upsampled = (preds_test_upsampled * 255).astype(np.uint8)
                    results[(j - 1) * 2:(j - 1) * 2 + 2] = self.prediction_to_annotation(preds_test_upsampled)
                    results_plus_conf[(j - 1) * 3:(j - 1) * 3 + 3] = self.compute_confidence(preds_test_upsampled)
                    self.plot_annotation(img, results, files[self.frame_selected_for_annotation[i]], OUTPUT)
                    self.dataFrame[self.dataFrame.columns[(j - 1) * 2]].values[self.frame_selected_for_annotation[i]] = -results[(j - 1) * 2]
                    self.dataFrame[self.dataFrame.columns[(j - 1) * 2 + 1]].values[self.frame_selected_for_annotation[i]] = results[(j - 1) * 2 + 1]
                    self.dataFrame3[self.dataFrame3.columns[(j - 1) * 3]].values[self.frame_selected_for_annotation[i]] = -results_plus_conf[(j - 1) * 3]
                    self.dataFrame3[self.dataFrame3.columns[(j - 1) * 3 + 1]].values[self.frame_selected_for_annotation[i]] = results_plus_conf[
                        (j - 1) * 3 + 1]
                    self.dataFrame3[self.dataFrame3.columns[(j - 1) * 3 + 2]].values[self.frame_selected_for_annotation[i]] = results_plus_conf[
                        (j - 1) * 3 + 2]

        self.dataFrame.to_pickle(self.annotation_file)
        self.dataFrame.to_csv(os.path.join(self.annotation_file + ".csv"))
        try:
            self.dataFrame3.to_csv(os.path.join(self.address, self.annotation_file + "_with_confidence.csv"))
        except:
            wx.MessageBox('Error in writing the results file'
                          'file not accessible\n '
                          'Error in writing'
                          , 'Error!', wx.OK | wx.ICON_ERROR)
    def lr_scheduler(self,epoch):
        return self.learning_rate * (0.5 ** (epoch // self.lr_drop))

    def prediction_to_annotation(self,annotation):
        # compute_corresponding_annotation_point
       # annotation = cv2.cvtColor(annotation, cv2.COLOR_BGR2GRAY)
        thresh, annotation = cv2.threshold(annotation, 0.99, 1, cv2.THRESH_BINARY)
        contour, hierarchy = cv2.findContours(annotation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contour != []:
            max_area = 0
            i_max = []
            for cc in contour:
                mom = cv2.moments(cc)
                area = mom['m00']
                if area > max_area:
                    max_area = area
                    i_max = cc


            if len(i_max)==0:

                xc = 1
                yc = -1
            else:
                center = cv2.moments(i_max)

                xc = center['m10'] / center['m00']
                yc = center['m01'] / center['m00']

        else:
            #maybe one joint is missing, but the other were correctly identified
            # self.dataFrame[self.dataFrame.columns[(i - 1) * 2]].values[j] = -1
            # self.dataFrame[self.dataFrame.columns[(i - 1) * 2 + 1]].values[j] = -1
            xc= 1
            yc=-1

        return xc,yc

        # self.statusbar.SetStatusText("File saved")
        # MainFrame.updateZoomPan(self)
        #
        # copyfile(os.path.join(self.filename + ".csv"), os.path.join(self.filename + "_MANUAL.csv"))
        # self.dataFrame.to_csv(os.path.join(self.filename + ".csv"))
        # copyfile(os.path.join(self.filename + ".csv"), os.path.join(self.filename + "_MANUAL.csv"))
        # self.dataFrame.to_pickle(self.filename)  # where to save it, usually as a .pkl
        # wx.PyCommandEvent(wx.EVT_BUTTON.typeId, self.load.GetId())

    def compute_confidence(self,annotation):
        # compute_corresponding_annotation_point
        # annotation = cv2.cvtColor(annotation, cv2.COLOR_BGR2GRAY)
        confidence_image = copy.copy(annotation)
        raw_image = copy.copy(annotation)
        confidence_image[np.where(confidence_image!=0)]=1;

        mask = np.zeros_like(confidence_image)
        thresh, annotation = cv2.threshold(annotation, 0.99, 1, cv2.THRESH_BINARY)
        contour, hierarchy = cv2.findContours(annotation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contour != []:
            max_area = 0
            i_max = []
            for cc in contour:
                mom = cv2.moments(cc)
                area = mom['m00']
                if area > max_area:
                    max_area = area
                    i_max = cc

            if len(i_max) == 0:
                xc = 1
                yc = -1
                confidence = 0
            else:

                center = cv2.moments(i_max)
                cv2.drawContours(mask, [i_max], -1, (255, 255, 255), -1)

                mask[np.where(mask != 0)] = 1;

                mean_values = np.multiply(confidence_image, mask)
                confidence = np.mean(raw_image[np.where(mean_values != 0)])
                confidence = confidence/255.0



                xc = center['m10'] / center['m00']
                yc = center['m01'] / center['m00']

        else:
            # maybe one joint is missing, but the other were correctly identified
            # self.dataFrame[self.dataFrame.columns[(i - 1) * 2]].values[j] = -1
            # self.dataFrame[self.dataFrame.columns[(i - 1) * 2 + 1]].values[j] = -1
            xc =  1
            yc = -1
            confidence = 0

        return xc, yc, confidence



    def plot_annotation(self,image,points,name,OUTPUT):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for i in range(0, len(self.bodyparts)):
            if not np.isnan(points[i * 2] and points[i * 2 + 1]):
                cv2.circle(image, (int(round((points[i * 2] * (2 ** 4)))), int(round(points[i * 2 + 1] * (2 ** 4)))),
                           int(round(self.markerSize * (2 ** 4))), self.colors[i]*255, thickness=-1, shift=4)
            cv2.imwrite(os.path.join(OUTPUT, name),
                        image)

    def ask(self,parent=None, message='Attention:\n you did not input the number of frames to automatically detect in previous interface \n Please, fill the following textbox with such number or cancel'):
        default_value = ""
        dlg = wx.TextEntryDialog(parent, message, value=default_value)
        dlg.ShowModal()
        result = dlg.GetValue()
        dlg.Destroy()
        return result