"""
    Source file name: annotation.py  
    
    Description: this file contains the code to open the GUI to perform annotation 
    
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




import wx
import os
import sys
from VisionTool import opening_toolbox
import cv2
from VisionTool.Frame_Extraction import *
from VisionTool.Interface_net import *
from VisionTool.training import *
from shutil import copyfile
import pandas as pd

class Label_frames(wx.Panel):
    """
    """
    def __init__(self, parent,gui_size,cfg):
        """Constructor"""
        wx.Panel.__init__(self, parent=parent)
        self.parent = parent
        # variable initilization
        self.import_from_deeplabcut_flag = 0
        self.parent_frame = parent
        self.method = "automatic"
        self.config = cfg
        # design the panel
        self.ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        self.CONFIG_PATH = os.path.dirname(os.path.join(self.ROOT_DIR, 'configuration.conf') ) # requires `import os`


        sizer = wx.GridBagSizer(5, 5)
        logo = os.path.join( self.CONFIG_PATH , 'malga.png')

        self.address = os.path.dirname(self.config)


        sizer_1 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_2 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_3 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_4 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_5 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_6 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_7 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_8 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_9 = wx.BoxSizer(wx.HORIZONTAL)



        sb_annotation = wx.StaticBox(self, label="Annotation options")

        sb = wx.StaticBox(self, label="Video_Length")
        boxsizer = wx.StaticBoxSizer(sb,wx.VERTICAL)
        sb2 = wx.StaticBox(self, label="Frames selection")
        boxsizer10 = wx.StaticBoxSizer(sb2, wx.VERTICAL)
        s22 = wx.StaticBox(self,label="Video names list")
        boxsizer2 = wx.StaticBoxSizer(s22,wx.VERTICAL)
        s33 = wx.StaticBox(self,label="Frame extraction preferences")
        boxsizer3 = wx.StaticBoxSizer(s33,wx.VERTICAL)
        s34 = wx.StaticBox(self,label="Options")
        boxsizer4 = wx.StaticBoxSizer(s34,wx.HORIZONTAL)

        boxsizer_annotation_and_length = wx.StaticBoxSizer(sb_annotation, wx.HORIZONTAL)



        img = wx.StaticBitmap(self, bitmap=wx.Bitmap(logo), pos=(100, 200))
        # self.sel_config = wx.FilePickerCtrl(self, path="",style=wx.FLP_USE_TEXTCTRL,message="Choose the config.yaml file", wildcard="config.yaml")
        self.help_button = wx.Button(self, label='Help')
        sizer_5.Add(self.help_button,flag=wx.RIGHT, border=10)
        self.help_button.Bind(wx.EVT_BUTTON, self.help_function)

        self.check = wx.Button(self, label="NN evaluation preferences")
        sizer_5.Add(self.check, flag=wx.RIGHT, border=10)
        self.check.Bind(wx.EVT_BUTTON, self.check_labelF)
        self.check.Enable(True)
        self.ok = wx.Button(self, label="Label Frames")
        sizer_5.Add(self.ok, flag=wx.RIGHT, border=10)


        self.textbox_ann = wx.TextCtrl(self, value="0", size=(50, -1))
        #self.lblname2 = wx.StaticText(self, label="End Frame", pos=(20, 60))
        s_ann = wx.StaticBox(self,label="Number of frames to annotate")
        sizer_7.Add(s_ann,flag=wx.RIGHT, border=67)
        sizer_7.Add(self.textbox_ann,flag=wx.RIGHT, border=10)

        self.textbox_automatic_ann = wx.TextCtrl(self, value="0", size=(50, -1))

        s_ann2 = wx.StaticBox(self, label="Number of frames to automatic annotate")
        sizer_8.Add(s_ann2, flag=wx.RIGHT, border=10)
        sizer_8.Add(self.textbox_automatic_ann, flag=wx.RIGHT, border=10)


        self.import_from_deeplabcut = wx.Button(self, label="Import from DeepLabCut")
        sizer_9.Add(self.import_from_deeplabcut, flag=wx.RIGHT, border=10)
        self.import_from_deeplabcut.Bind(wx.EVT_BUTTON, self.import_from_deeplabcut_method)




        self.train = wx.Button(self, label="Start training")
        sizer_5.Add(self.train, flag=wx.RIGHT, border=10)
        self.train.Bind(wx.EVT_BUTTON, self.check_and_train)
        self.test = wx.Button(self, label="Start testing")
        self.test.Bind(wx.EVT_BUTTON, self.check_and_test)

        sizer_5.Add(self.test, flag=wx.RIGHT, border=10)
        self.does_annotation_exist = wx.CheckBox(self, label='Partial or complete Annotation is available')
        self.does_annotation_exist.Enable(False)
        sizer_6.Add(self.does_annotation_exist, flag=wx.BOTTOM)
        self.ok.Bind(wx.EVT_BUTTON, self.label_frames)
        list_options = ['Uniform','K-means']
        #self.Extract_frames_options = wx.CheckListBox(self, -1, (400, 25), wx.DefaultSize, list_options)
        self.Extract_frames_options = wx.CheckListBox(self, -1, (400, 25), wx.DefaultSize, list_options)

        self.Bind(wx.EVT_CHECKLISTBOX, self.EvtCheckListBox, self.Extract_frames_options)
        self.Extract_frames_options.Check(0,True)
        self.Extract_frames_options.SetSelection(0)
        self.index = 0
        #self.lblname = wx.StaticText(self, label="Start frame", pos=(20, 25))
        self.textbox1 = wx.TextCtrl(self, value="0", size=(50, -1))
        #self.lblname2 = wx.StaticText(self, label="End Frame", pos=(20, 60))
        s1 = wx.StaticBox(self,label="Start Frame")
        # s33 = wx.StaticBox(self, label="Number of frames to annotate")
        s2 = wx.StaticBox(self,label="End Frame")
        sizer_2.Add(s1, 0, wx.LEFT|wx.TOP, border=6)
        sizer_2.Add(self.textbox1, 0, wx.LEFT|wx.TOP, border=10)
        # sizer_2.Add(s33, 1, wx.LEFT|wx.TOP, 20)
        sizer_3.Add(s2, 0, wx.LEFT, border=10)
        self.textbox2 = wx.TextCtrl(self, value="0", size=(50, -1))
        sizer_3.Add(self.textbox2, 0, wx.LEFT, border=10)
        boxsizer.Add(sizer_2, 1, wx.EXPAND,0)
        boxsizer.Add(sizer_3, 1, wx.EXPAND, 0)
        boxsizer10.Add(sizer_7,1,wx.EXPAND, 0)
        boxsizer10.Add(sizer_8,1,wx.EXPAND, 0)
        boxsizer10.Add(sizer_9, 1, wx.EXPAND, 0)
        boxsizer_annotation_and_length.Add(boxsizer, wx.EXPAND, 0)
        boxsizer_annotation_and_length.Add(boxsizer10, wx.EXPAND, 0)

        #sizer.Add(self.textbox2, pos=(200, 50))
        #sizer.AddGrowableCol(2)

        #add the list of videos

        self.video_list_text = self.get_video_list()
        self.video_list_with_address = self.get_video_list()
        for i in range(0,len(self.video_list_text)):
            self.video_list_text[i] = self.video_list_text[i][self.find(self.video_list_text[i], os.sep)[-1] + 1:-1]
        self.video_list_text.append('Analyze_all')
        self.video_list = wx.CheckListBox(self, -1, (400, 25), wx.DefaultSize)

        sizer_1.Add(self.video_list,1, wx.TOP | wx.BOTTOM, 20)
        sizer_4.Add(self.Extract_frames_options, 1, wx.TOP | wx.EXPAND, 20)

        boxsizer2.Add(sizer_1, 1, wx.EXPAND,0)

        boxsizer3.Add(sizer_4, 1, wx.EXPAND,0)

        boxsizer4.Add(boxsizer2, 1, wx.EXPAND, 0)
        boxsizer4.Add(boxsizer3, 1, wx.EXPAND, 0)

        self.video_list.Set(self.video_list_text)
        self.Update()
        self.Bind(wx.EVT_CHECKLISTBOX, self.EvtCheckListBox_video, self.video_list)
        self.video_list.Check(0,True)
        self.video_list.SetSelection(0)
        self.index_video = 0
        sizer.Add(boxsizer4, pos=(2, 0), span=(1, 5), flag=wx.EXPAND |wx.LEFT, border=10)
        #sizer.Add(boxsizer, pos=(0, 0), span=(1, 5), flag=wx.LEFT, border=10)
        sizer.Add(boxsizer_annotation_and_length, pos=(0, 0), span=(1, 5), flag=wx.LEFT, border=10)

        sizer.Add(img,pos=(5,0),span=(1, 5), flag=wx.EXPAND |wx.ALIGN_CENTER, border=10)
        sizer.Add(sizer_5, pos=(8, 0), span=(1, 5), flag=wx.BOTTOM|wx.ALIGN_CENTER, border=10)
        sizer.Add(sizer_6, pos=(9, 0), span=(1, 5), flag=wx.EXPAND | wx.BOTTOM, border=10)

        sizer.AddGrowableCol(2)
        #sizer.AddGrowableRow(2)
        self.sizer = sizer
        self.SetSizerAndFit(self.sizer)
        #initial selection
        self.read_videos_for_length()
        self.check_existence_annoatation()
        self.Layout()
        self.Update()
     
    def find(self,s, ch):
        return [i for i, ltr in enumerate(s) if ltr == ch]

    def check_and_train(self,event):
        self.address_proj = os.path.dirname(self.config)

        self.file_preferences = self.address_proj + os.sep + 'Architecture_Preferences.txt'


        if not os.path.isfile(self.file_preferences):
            wx.MessageBox('First, select the preferences for the estimation', 'Preferences missing',
                          wx.OK | wx.ICON_INFORMATION)
            return

        if not self.does_annotation_exist.GetValue() and self.index_video!=len(self.video_list_text)-1:
            wx.MessageBox('No annotation found!', 'Annotation missing', wx.OK | wx.ICON_INFORMATION)
            return

        if self.index_video!=len(self.video_list_text)-1:
            self.name = 'Extracted_frames_' + self.video_list_with_address[self.index_video][self.find(self.video_list_with_address[self.index_video], os.sep)[-1] + 1:-1]
            training(address = self.address_proj,file_annotation = self.filename[:-4],image_folder = self.address + os.sep + self.name + os.sep,annotation_folder = os.path.join(self.address,self.name),bodyparts = self.bodyparts,train_flag = 1,annotation_assistance=0,multi_video=0)
        else:
            self.name = ""
            training(address = self.address_proj,file_annotation = self.filename[:-4],image_folder = self.address + os.sep + self.name + os.sep,annotation_folder = os.path.join(self.address,self.name),bodyparts = self.bodyparts,train_flag = 1,annotation_assistance=0,multi_video=1)


    def check_and_test(self,event):

        self.address_proj = os.path.dirname(self.config)

        self.file_preferences = self.address_proj + os.sep + 'Architecture_Preferences.txt'
        if not os.path.isfile(self.file_preferences):
            wx.MessageBox('First, select the preferences for the estimation', 'Preferences missing', wx.OK | wx.ICON_INFORMATION)
            return
        if not self.does_annotation_exist.GetValue() and self.index_video!=len(self.video_list_text)-1:
            wx.MessageBox('No annotation found!', 'Annotation missing', wx.OK | wx.ICON_INFORMATION)
            return
        else:
            if not os.path.isfile(self.address + os.sep + 'Unet.h5'):
                wx.MessageBox('No Trained network found!', 'Trained model missing', wx.OK | wx.ICON_INFORMATION)
                return
            else:
                dlg = wx.MessageDialog(None,
                                       "Do you want to proceed?",
                                       "Found existing trained architecture!", wx.YES_NO | wx.ICON_WARNING)
                result = dlg.ShowModal()
                if result == wx.ID_NO:
                    return

                else:

                    if self.index_video != len(self.video_list_text) - 1:
                        self.name = 'Extracted_frames_' + self.video_list_with_address[self.index_video][
                                                          self.find(self.video_list_with_address[self.index_video],
                                                                    os.sep)[-1] + 1:-1]
                        training(address=self.address_proj, file_annotation=self.filename[:-4],
                                 image_folder=self.address + os.sep + self.name + os.sep,
                                 annotation_folder=os.path.join(self.address, self.name),
                                 bodyparts=self.bodyparts, train_flag=0, annotation_assistance=0,multi_video=0)
                    else:
                        self.name = ""
                        training(address=self.address_proj, file_annotation=self.filename[:-4],
                                 image_folder=self.address + os.sep + self.name + os.sep,
                                 annotation_folder=os.path.join(self.address, self.name),
                                 bodyparts=self.bodyparts, train_flag=0, annotation_assistance=0, multi_video=1)

    def import_from_deeplabcut_method(self,event):

        with wx.FileDialog(self, "Select deeplabcut annotation csv file",
                           wildcard='select configuration files (*.csv)|*.csv',
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:

            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return  # the user changed their mind

            pathname = fileDialog.GetPaths()

            preferences_file = os.path.dirname(self.config) + os.sep + 'annotation_options.txt'
            if not os.path.isfile(preferences_file):
                wx.MessageBox(
                    'Error in reading the annotation options'
                    'First, choose your preferences in annotation using the button Label Frames \n '
                    'Preferences_file_error'
                    , 'Error!', wx.OK | wx.ICON_ERROR)
                return
            file = open(preferences_file)
            self.pref = file.readlines()
            self.scorer = self.pref[1]
            self.bodyparts = self.pref[9:]
            self.title_video = self.video_list_with_address[self.index_video][
                               self.find(self.video_list_with_address[self.index_video], os.sep)[-1] + 1:-1]
            try:
                self.filename = self.address + os.sep + "Annotation_" + self.title_video + '_' + self.scorer[:-1] + '.csv'
                copyfile(pathname[0],pathname[0][:-4] + 'original' + '.csv')
                os.rename(pathname[0], self.filename)
            except:
                wx.MessageBox(
                    'Error in renaming DLC file'
                    'Check permission and file existence'
                    , 'Error!', wx.OK | wx.ICON_ERROR)
            try:
                dat = pd.read_csv(self.filename, header=None)
                dat.to_pickle(self.filename[:-4])
                self.dataFrame = pd.read_pickle(self.filename[:-4])
                self.dataFrame.sort_index(inplace=True)
                self.dataFrame.to_pickle(self.filename[:-4])
                self.annotated = []
                digit = int(np.log10(int(self.textbox2.GetValue()))) + 1
                for i in range(3, len(self.dataFrame.iloc[:, 0])):
                    self.annotated.append(self.dataFrame.iloc[i, 0][-4 - digit:-4])
                self.import_from_deeplabcut_flag = 1
            except:
                pass


    def Get_annotation_for_deeplabcut_compat(self):
        return self.annotated


    def update_Text(self,text,index):
        if index==1:
            self.textbox1.SetValue(text)
        else:
            self.textbox2.SetValue(text)

    def get_Text(self,index):
        if index==1:
            return self.textbox1.GetValue()
        elif index==2:
            return self.textbox2.GetValue()
        elif index ==3:
            return self.textbox_ann.GetValue()
        else:
            return self.textbox_automatic_ann.GetValue()

    def EvtCheckListBox(self, event):

        self.index = event.GetSelection()

        for i in range(0,2):
            self.Extract_frames_options.Check(i, False)
        self.Extract_frames_options.Check(self.index, True)
        self.Extract_frames_options.SetSelection(self.index)

        # so that (un)checking also selects (moves the highlight)

    def EvtCheckListBox_video(self, event):

        self.index_video = event.GetSelection()

        for i in range(0,len(self.video_list_text)):
            self.video_list.Check(i, False)
        self.video_list.Check(self.index_video, True)
        self.video_list.SetSelection(self.index_video)

        if self.index_video==len(self.video_list_text)-1:
            self.textbox1.Enable(False)
            self.textbox2.Enable(False)
        else:
            self.textbox1.Enable(True)
            self.textbox2.Enable(True)
            self.read_videos_for_length()
            self.check_existence_annoatation()


    def read_videos_for_length(self):
        self.cap = cv2.VideoCapture(self.video_list_with_address[self.index_video][:-1])
        success, image = self.cap.read()
        count = 0


        while success:
            success, image = self.cap.read()
            count += 1

        self.cap.release()
        self.update_Text(str(count-1),2)
        # so that (un)checking also selects (moves the highlight)

    def help_function(self,event):

        help_text = "Use this interface to open the videos you want to label"
        wx.MessageBox(help_text,'Help',wx.OK | wx.ICON_INFORMATION)


    def check_existence_annoatation(self):
        preferences_file = os.path.dirname(self.config)  + os.sep + 'annotation_options.txt'
        if not os.path.isfile(preferences_file):
            self.does_annotation_exist.SetValue(False)
            return
        file = open(preferences_file)
        self.pref = file.readlines()
        self.scorer = self.pref[1]
        self.bodyparts = self.pref[9:]
        self.title_video = self.video_list_with_address[self.index_video][
                           self.find(self.video_list_with_address[self.index_video], os.sep)[-1] + 1:-1]

        self.filename = self.address  + os.sep + "Annotation_" + self.title_video + '_' + self.scorer[:-1] + '.csv'

        if os.path.isfile(self.filename):
            self.does_annotation_exist.SetValue(True)
        else:
            self.does_annotation_exist.SetValue(False)


    def check_labelF(self,event):


        Open_interface.show(None,self,(700,700),self.address)


    def label_frames(self,event):
        if wx.MessageBox(
                "Proceeding with frame extraction.\n"
                "Do you want to proceed with the analysis of selected video", "Confirm",
                wx.YES_NO | wx.YES_DEFAULT, self) == wx.YES:
            f_object = extract_Frames(self,self.video_list_with_address,self.index_video,self,self.index,self.config,self.import_from_deeplabcut_flag)
            if f_object.error == 1:
                return
            opening_toolbox.show(self,self.video_list_with_address,self.index_video,self.config,self.index, imtypes=['*.png'],)


    def get_video_list(self):
        self.config_file = open(self.config, "r")
        a = self.config_file.readlines()[2:]
        self.config_file.close()
        return a

    def destroy(self):
        self.Destroy()
