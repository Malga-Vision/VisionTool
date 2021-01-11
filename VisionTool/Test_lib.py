"""
    Source file name: test_label_video.py
    
    Description: this file contains the code to test video labeling
    
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
import VisionTool.opening_toolbox
import cv2
from VisionTool.Frame_Extraction import *
from VisionTool.Interface_net import *
from VisionTool.New_project_features import *
from VisionTool.training import *
from shutil import copyfile
import pandas as pd
import VisionTool.opening_toolbox
from VisionTool.annotation import *
from shutil import copyfile
from VisionTool.create_new_project  import *

class test ():

    def __init__(self):
        self.ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        self.CONFIG_PATH = os.path.dirname(os.path.join(self.ROOT_DIR, 'configuration.conf'))  # requires `import os`
        self.upload_existent = 0
        self.app = wx.App()
        self.app.MainLoop()

    def new_project_routine(self):
        initiate = routine()
        self.address = initiate.new_project()
        self.config_file_text = os.path.join(self.address, "file_configuration.txt")
        pass

    def open_project(self):
        initiate = routine()
        self.address = initiate.open_existing_project()
        self.config_file_text = os.path.join(self.address, "file_configuration.txt")

    def Load_Videos(self):

        if self.address == "":

            error = wx.MessageBox("you need to create a new project or open an existing one, first!")

        else:
            try:
                self.config_file = open(os.path.join(self.address, "file_configuration.txt"), "r")
            except:
                wx.MessageBox('Error in reading the configuration file, file not accessible \n '
                              'Configuration_file_error'
                              , 'Error!', wx.OK | wx.ICON_ERROR)
                return
            a = self.config_file.readlines()
            self.config_file.close()
            if len(a) > 1:
                if wx.MessageBox(
                        "The project already contains videos.\n"
                        "Do you want to add new videos?", "Confirm",
                        wx.YES_NO | wx.NO_DEFAULT, None) == wx.NO:
                            pass
                else:
                    self.upload_new_Videos()
            else:
                self.upload_new_Videos()


    def upload_new_Videos(self):

        with wx.FileDialog(None, "Select videos",
                           wildcard='select video files (*.mp4;*.avi;*.mpeg4)|*.mp4;*.avi;*.mpeg4|''mp4 files (*.mp4)|*.mp4|'
                                    'avi files (*.avi)|*.avi|' 'mpeg4 (*.mpeg4)|*.mpeg4',
                           defaultDir  = self.CONFIG_PATH + os.sep,style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST | wx.FD_MULTIPLE) as fileDialog:

            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return  # the user changed their mind

            pathname = fileDialog.GetPaths()

            # videos = os.listdir(dlg.GetPath())
            # if len(videos)==0:
            #     error = wx.MessageBox("Warning","Attention, no valid video has been selected")
            try:
                self.config_file = open(os.path.join(self.address, "file_configuration.txt"), "r")
            except:
                wx.MessageBox('Error in reading the configuration file, file not accessible \n '
                              'Configuration_file_error'
                              , 'Error!', wx.OK | wx.ICON_ERROR)
                return
            a = self.config_file.readlines()
            self.config_file.close()

            try:
                self.config_file = open(os.path.join(self.address, "file_configuration.txt"), "a")
            except:
                wx.MessageBox('Error in reading the configuration file, file not accessible \n '
                              'Configuration_file_error'
                              , 'Error!', wx.OK | wx.ICON_ERROR)
                return
            if len(a) > 1:
                for i in range(0, len(pathname)):
                    self.config_file.writelines(pathname[i] + '\n')
            else:
                self.config_file.writelines("\nVideo\n")

                for i in range(0, len(pathname)):
                    self.config_file.writelines(pathname[i] + '\n')

        self.config_file.close()


    def get_video_list(self):
        self.config_file = open(self.config, "r")
        a = self.config_file.readlines()[2:]
        self.config_file.close()
        return a


    def annotate(self, index_video, config,num_annotated = 0, num_auto_annotated = 0):


        self.config = config
        # uniform
        self.annotation = os.path.dirname(self.config) + os.sep + 'annotation_options.txt'
        self.video_list_with_address = self.get_video_list()
        self.index_video = index_video
        index_type = 0
        self.name = 'Extracted_frames_' + self.video_list_with_address[self.index_video][
                                          self.find(self.video_list_with_address[self.index_video], os.sep)[-1] + 1:-1]

        if not os.path.isfile(os.path.dirname(self.config) + os.sep + 'annotation_options.txt'):

            Frame_selection(os.path.dirname(self.config) + os.sep + 'annotation_options.txt')
        else:
            if wx.MessageBox(
                    "A set of preferences has already been saved.\n"
                    "Do you want to change your preferences?", "Confirm",
                    wx.YES_NO | wx.YES_DEFAULT, None) == wx.YES:
                os.remove(os.path.dirname(self.config) + os.sep + 'annotation_options.txt')
                Frame_selection(self, os.path.dirname(self.config) + os.sep + 'annotation_options.txt')

        if index_type == 0:
            self.read_videos_for_length(num_annotated,num_auto_annotated)




    def read_videos_for_length(self,num_annotated=0,num_auto_annotated=0):

        self.cap = cv2.VideoCapture(self.video_list_with_address[self.index_video][:-1])
        success, image = self.cap.read()
        count = 0
        while(success):
           success, image = self.cap.read()
           count+=1

        if num_annotated==0:
            num_annotated = int(count*0.70)
        if num_auto_annotated==0:
            num_auto_annotated = int(count*0.15)

        self.cap.release()

        if not os.path.isdir(self.address + os.sep + self.name):
            os.mkdir(self.address + os.sep + self.name)


        if not self.upload_existent:
        # get preferences in textbox annotation panel
            num_frame_annotated = num_annotated
            num_frame_automatic = num_auto_annotated
            start = 0
            end = count

            self.address = os.path.dirname(self.config)

            if not os.path.isdir(self.address + os.sep + self.name):
                os.mkdir(self.address + os.sep + self.name)


                my_list = list(range(start,
                                     end))  # list of integers from 1 to 99                # adjust this boundaries to fit your needs
                random.shuffle(my_list)
                if num_frame_annotated != 0 or self.import_from_deeplabcut_flag == 1:
                    if num_frame_annotated < (end - start):
                            frames = my_list[0:num_frame_annotated]
                            frames_annotated = my_list[num_frame_annotated:num_frame_automatic + num_frame_annotated]
                    address = os.path.dirname(self.config)
                    p = open(os.path.join(address, '_index_annotation.txt'), 'w')
                    p2 = open(os.path.join(address, '_index_annotation_auto.txt'), 'w')

                    for i in frames:
                        p.writelines(str(i))
                        p.writelines('\n')

                    for i in frames_annotated:
                        p2.writelines(str(i))
                        p2.writelines('\n')

                    p.close()
                    p2.close()

            else:
                permission = wx.MessageBox(
                    "Frame already extracted\n"
                    "Do you want to extract frames again? (The procedure will delete previous frames", "Confirm",
                    wx.YES_NO | wx.NO_DEFAULT, None)
                if permission == 2:
                    shutil.rmtree(self.address + os.sep + self.name)
                    try:
                        os.mkdir(self.address + os.sep + self.name)

                    except:
                        os.mkdir(self.address + os.sep + self.name)

                    num_frame_annotated = int(num_annotated)
                    num_frame_automatic = int(num_auto_annotated)

                    my_list = list(range(start,
                                         end))  # list of integers from 1 to 99                # adjust this boundaries to fit your needs
                    random.shuffle(my_list)
                    if num_frame_annotated != 0 or self.import_from_deeplabcut_flag == 1:
                        if num_frame_annotated < (end - start):
                                frames = my_list[0:num_frame_annotated]
                                frames_annotated = my_list[num_frame_annotated:num_frame_automatic + num_frame_annotated]
                        address = os.path.dirname(self.config)
                        p = open(os.path.join(address, '_index_annotation.txt'), 'w')
                        p2 = open(os.path.join(address, '_index_annotation_auto.txt'), 'w')

                        for i in frames:
                            p.writelines(str(i))
                            p.writelines('\n')

                        for i in frames_annotated:
                            p2.writelines(str(i))
                            p2.writelines('\n')

                        p.close()
                        p2.close()





                else:
                    return
        start = 0
        end = count
        self.frames_id = np.asarray(range(start, end, 1)).astype(int)
        progress = wx.ProgressDialog("extraction in progress", "please wait", maximum=100, parent=None,
                                     style=wx.PD_SMOOTH | wx.PD_AUTO_HIDE)


        self.cap = cv2.VideoCapture(self.video_list_with_address[self.index_video][:-1])
        success, image = self.cap.read()
        count = 1
        cv2.imwrite(self.address + os.sep + self.name + os.sep + 'frame_' + ("{:04d}".format(count)) + '.png',
                    image)
        while success and count < start:
            count += 1
            success, image = self.cap.read()
        while success and count >= start and count < end:
            progress.Update(int(count / len(self.frames_id)) * 100)
            cv2.imwrite(self.address + os.sep + self.name + os.sep + 'frame_' + ("{:04d}".format(count-1)) + '.png',
                        image)
            success, image = self.cap.read()
            count += 1
        self.cap.release()

        progress.Destroy()

        self.cap.release()

        self.view_annotation()

    def find(self, s, ch):
        return [i for i, ltr in enumerate(s) if ltr == ch]


    def view_annotation(self):

        opening_toolbox.show(None, self.video_list_with_address, self.index_video, self.config, 0,
                             imtypes=['*.png'], )

    def preferences_annotation(self):
        Open_interface.show(None, self, (700, 700), self.address)


    def check_and_train(self):

        self.address_proj = os.path.dirname(self.config)

        self.file_preferences = self.address_proj + os.sep + 'Architecture_Preferences.txt'
        self.name = 'Extracted_frames_' + self.video_list_with_address[self.index_video][self.find(self.video_list_with_address[self.index_video], os.sep)[-1] + 1:-1]
        if not os.path.isfile(self.file_preferences):
            wx.MessageBox('First, select the preferences for the estimation', 'Preferences missing', wx.OK | wx.ICON_INFORMATION)
            return

        files = os.listdir(self.address)
        for i in range(0, len(files)):
            if '.csv' in files[i] and 'with_confidence' not in files[i]:
                self.does_annotation_exist = 1
                self.filename = files[i]
            if 'annotation_options' in files[i]:
                preferences_file = os.path.dirname(self.config) + os.sep + 'annotation_options.txt'
                pref_ann = open(preferences_file, 'r')
                temporary = pref_ann.readlines()
                self.scorer = temporary[1]
                self.bodyparts = temporary[9:]

        if not self.does_annotation_exist:
            wx.MessageBox('No annotation found!', 'Annotation missing', wx.OK | wx.ICON_INFORMATION)
            return





        else:
            training(address = self.address,file_annotation = self.filename[:-4],image_folder = self.address + os.sep + self.name + os.sep,annotation_folder = os.path.join(self.address,self.name + 'annotation'),bodyparts = self.bodyparts,train_flag = 1,annotation_assistance=0)
            pass

    def check_and_test(self):



        self.address_proj = os.path.dirname(self.config)

        self.file_preferences = self.address_proj + os.sep + 'Architecture_Preferences.txt'
        self.name = 'Extracted_frames_' + self.video_list_with_address[self.index_video][self.find(self.video_list_with_address[self.index_video], os.sep)[-1] + 1:-1]
        if not os.path.isfile(self.file_preferences):
            wx.MessageBox('First, select the preferences for the estimation', 'Preferences missing', wx.OK | wx.ICON_INFORMATION)
            return
        files = os.listdir(self.address)
        for i in range(0, len(files)):
            if '.csv' in files[i] and 'with_confidence' not in files[i]:
                self.does_annotation_exist = 1
                self.filename = files[i]
            if 'annotation_options' in files[i]:
                preferences_file = os.path.dirname(self.config) + os.sep + 'annotation_options.txt'
                pref_ann = open(preferences_file, 'r')
                temporary = pref_ann.readlines()
                self.scorer = temporary[1]
                self.bodyparts = temporary[9:]
        del self.app
        self.app = wx.App()
        self.app.MainLoop()
        if not self.does_annotation_exist:
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
                    training(address=self.address_proj, file_annotation=self.filename[:-4],
                             image_folder=self.address + os.sep + self.name + os.sep,
                             annotation_folder=os.path.join(self.address, self.name , 'annotation'),
                             bodyparts=self.bodyparts, train_flag=0,annotation_assistance=0)

    def load_testing_annotation(self,config):
        self.upload_existent = 1
        self.config = config
        self.dataFrame = pd.read_pickle(os.path.join(self.CONFIG_PATH , 'Annotation_sample_video.avi_test'))
        self.dataFrame.to_pickle(os.path.join(os.path.dirname(self.config),'Annotation_sample_video.avi_test'))
        copyfile(os.path.join(self.CONFIG_PATH , 'Annotation_sample_video.avi_test.csv'),os.path.join(os.path.dirname(self.config) , 'Annotation_sample_video.avi_test.csv'))
        copyfile(os.path.join(self.CONFIG_PATH , 'sample_video.avi_index_annotation.txt'),os.path.join(os.path.dirname(self.config) , 'sample_video.avi_index_annotation.txt'))
        copyfile(os.path.join(self.CONFIG_PATH , 'sample_video.avi_index_annotation_auto.txt'),os.path.join(os.path.dirname(self.config) , 'sample_video.avi_index_annotation_auto.txt'))
        copyfile(os.path.join(self.CONFIG_PATH , 'annotation_options.txt'),os.path.join(os.path.dirname(self.config) , 'annotation_options.txt'))
