"""
    Source file name: training.py  
    
    Description: this file contains the code to call the training module and perform neural network training and testing 
    
    Code adapted and modified from: 
    Alexander Mathis, Pranav Mamidanna, Kevin M Cury, Taiga Abe, Venkatesh N Murthy,Mackenzie Weygandt Mathis, and Matthias Bethge. Deeplabcut: markerless pose estimation
    of user-defined body parts with deep learning. Nature neuroscience, 21(9):1281{1289}, 2018.
    
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


from VisionTool.architectures_segmentation import *
import os
import wx
import matplotlib.pyplot as plt

class training ():
    def __init__(self,address,file_annotation,image_folder,annotation_folder,bodyparts,train_flag,annotation_assistance):
        self.address = address 
        self.train_flag = train_flag
        self.bodyparts = bodyparts
        self.annotation_assistance = annotation_assistance
        self.file_annotation = file_annotation
        self.image_folder = image_folder
        self.error = 0
        self.annotation_folder = annotation_folder
        self.file_preferences = os.path.join(self.address,'Architecture_Preferences.txt')
        self.preferences_file =  os.path.join(self.address,'annotation_options.txt')
        try:
            file = open(self.preferences_file)
            self.pref = file.readlines()
        except:
            wx.MessageBox('Error in reading the network configuration file, please check the existence or choose'
                          'preferences again \n '
                          'Preferences_file_error'
                          , 'Error!', wx.OK | wx.ICON_ERROR)
            self.error
            return
        try:
            reading_pointer = open(self.file_preferences)
        except:
            wx.MessageBox(
                'Please, update the NN evaluation preferences using the correspondent button',
                'User instructions', wx.OK | wx.ICON_INFORMATION)
            self.error=-1
            return
        reading = reading_pointer.readlines()
        self.imagenet = reading[5][:-1]
        self.single_labels = reading[7][:-1]
        self.learning_rate = reading[9][:-1]
        self.loss = reading[11][:-1]

        self.map = self.pref[7][:-1]
        try:
            self.colormap = plt.get_cmap(self.map)
        except:
            wx.MessageBox(self.map + 'is not recognized as a valid map \n Settings color map to Pastel2'
                          , 'Error!', wx.OK | wx.ICON_ERROR)
            self.colormap = plt.get_cmap('inferno')
            self.error = -1
            return

        self.colormap = self.colormap.reversed()
        colorIndex = np.linspace(0, 255, len(bodyparts))
        colorIndex = colorIndex.astype(int)
        self.colors = self.colormap(colorIndex)

        if self.imagenet == 'Yes':
            self.imagenet = 'imagenet'
        else:
            self.imagenet = 'None'


        self.architecture = reading[1][:-1]
        self.backbone = reading[3][:-1]
        self.train()
        

            #to be added
        file.close()
        reading_pointer.close()
    def train(self):
        #loading of annotation
            unet(architecture = self.architecture, backbone = self.backbone,image_net = self.imagenet,single_labels=self.single_labels,
                 colors = self.colors,address = self.address,annotation_file = self.file_annotation,
                 image_folder = self.image_folder,
                 annotation_folder = self.annotation_folder,bodyparts= self.bodyparts,
                 train_flag = self.train_flag,annotation_assistance = self.annotation_assistance,lr=self.learning_rate,loss=self.loss)

        
        
