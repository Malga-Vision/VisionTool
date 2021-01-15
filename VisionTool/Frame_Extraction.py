"""
    Source file name: Frame_Extraction.py  
    
    Description: this file contains the code to extract frames to proceed with manual annotation 
    
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
from VisionTool.annotation import *
import cv2
import numpy as np
import shutil
from VisionTool.New_project_features import *
import random


class extract_Frames(wx.Panel):

    def __init__(self, parent, video_list_with_address, index_video, annotation_preferences, index_type, config,
                 import_from_deeplabcut_flag):

        wx.Panel.__init__(self, parent=parent)
        self.config = config
        self.error = 0
        self.import_from_deeplabcut_flag = import_from_deeplabcut_flag
        # uniform
        self.annotation = annotation_preferences
        self.video_list_with_address = video_list_with_address
        self.index_video = index_video
        if self.index_video ==len(self.video_list_with_address):
            wx.MessageBox(
                'Annotation interface is only available if a video is selected\n'
                'Please, de-select \'Analyze_all\' and select a video to open annotation interface\n '
                'User Input Error'
                , 'Error!', wx.OK | wx.ICON_ERROR)
            self.error = 1
            return
        self.name = 'Extracted_frames_' + self.video_list_with_address[self.index_video][
                                          self.find(self.video_list_with_address[self.index_video], os.sep)[-1] + 1:-1]

        if not os.path.isfile(os.path.join(os.path.dirname(self.config),  'annotation_options.txt')):

            Frame_selection(self, os.path.join(os.path.dirname(self.config), 'annotation_options.txt'))
        else:
            if wx.MessageBox(
                    "A set of preferences has already been saved.\n"
                    "Do you want to change your preferences?", "Confirm",
                    wx.YES_NO | wx.YES_DEFAULT, self) == wx.YES:
                os.remove(os.path.dirname(self.config) + '//annotation_options.txt')
                Frame_selection(self, os.path.dirname(self.config) + '//annotation_options.txt')
            else:
                file = open( os.path.join(os.path.dirname(self.config),  'annotation_options.txt'))
                self.pref = file.readlines()
                self.scorer = self.pref[1]

        if index_type == 0:
            self.uniform()
            self.read_videos_for_length()
        else:
            self.uniform()
            self.read_videos_for_length()
            #self.k_means() will be implemented


    def find(self, s, ch):
        return [i for i, ltr in enumerate(s) if ltr == ch]

    def uniform(self):
        pass

    def k_means(self):
        pass

    def manual(self):
        pass

    def read_videos_for_length(self):

        self.cap = cv2.VideoCapture(self.video_list_with_address[self.index_video][:-1])
        success, image = self.cap.read()
        count = 0
        # get preferences in textbox annotation panel
        start = int(self.annotation.get_Text(1))
        end = int(self.annotation.get_Text(2))

        self.address = os.path.dirname(self.config)

        if not os.path.isdir(self.address + os.sep + self.name):
            os.mkdir(self.address + os.sep + self.name)

            num_frame_annotated = int(self.annotation.get_Text(3))
            num_frame_automatic = int(self.annotation.get_Text(4))

            my_list = list(range(start,
                                 end))  # list of integers from 1 to end                # adjust this boundaries to fit your needs
            random.shuffle(my_list)

            if num_frame_annotated < (end - start):
                if self.import_from_deeplabcut_flag:
                    frames = self.annotation.Get_annotation_for_deeplabcut_compat()
                else:
                    frames = my_list[0:num_frame_annotated]

                if num_frame_automatic + num_frame_annotated<(end - start):
                    frames_annotated = my_list[num_frame_annotated:num_frame_automatic + num_frame_annotated]
                else:
                    wx.MessageBox('Please enter a number of frames to automatically annotate <= to the total number of selected frames (end-start)\n'
                                  'Number of frame to annotate too high\n '
                                  'User Input Error'
                                  , 'Error!', wx.OK | wx.ICON_ERROR)
                    self.error = 1

                    return

            else:
                wx.MessageBox('Please enter a number of frames to annotate <= to the total number of selected frames (end-start)\n'
                              'Number of frame to annotate too high\n '
                              'User Input Error'
                              , 'Error!', wx.OK | wx.ICON_ERROR)
                self.error = 1
                return
            address = os.path.dirname(self.annotation.config)
            p = open(os.path.join(address,self.video_list_with_address[self.index_video][
                                          self.find(self.video_list_with_address[self.index_video], os.sep)[-1] + 1:-1]+  '_index_annotation.txt'), 'w')
            p2 = open(os.path.join(address, self.video_list_with_address[self.index_video][
                                          self.find(self.video_list_with_address[self.index_video], os.sep)[-1] + 1:-1] + '_index_annotation_auto.txt'), 'w')

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
                wx.YES_NO | wx.NO_DEFAULT, self)
            if permission == 2:
                shutil.rmtree(self.address + os.sep + self.name)

                try:
                    os.remove(self.address  + os.sep + "Annotation_" + self.video_list_with_address[self.index_video][
                                              self.find(self.video_list_with_address[self.index_video], os.sep)[-1] + 1:-1] + '_' + self.scorer[:-1] + '.csv')
                    os.remove(self.address + os.sep + "Annotation_" + self.video_list_with_address[self.index_video][
                                          self.find(self.video_list_with_address[self.index_video], os.sep)[-1] + 1:-1] + '_' + self.scorer[:-1])
                except:
                    pass
                os.mkdir(self.address + os.sep + self.name)

                num_frame_annotated = int(self.annotation.get_Text(3))
                num_frame_automatic = int(self.annotation.get_Text(4))

                my_list = list(range(start,
                                     end))  # list of integers from 1 to 99                # adjust this boundaries to fit your needs
                random.shuffle(my_list)

                if num_frame_annotated < (end - start):
                    if self.import_from_deeplabcut_flag:
                        frames = self.annotation.Get_annotation_for_deeplabcut_compat()
                    else:
                        frames = my_list[0:num_frame_annotated]


                    if num_frame_automatic + num_frame_annotated < (end - start):
                        frames_annotated = my_list[num_frame_annotated:num_frame_automatic + num_frame_annotated]
                    else:
                        wx.MessageBox(
                            'Please enter a number of frames to automatically annotate <= to the total number of selected frames (end-start)\n'
                            'Number of frame to annotate too high\n '
                            'User Input Error'
                            , 'Error!', wx.OK | wx.ICON_ERROR)
                        self.error = 1

                        return

                else:
                    wx.MessageBox(
                        'Please enter a number of frames to annotate <= to the total number of selected frames (end-start)\n'
                        'Number of frame to annotate too high\n '
                        'User Input Error'
                        , 'Error!', wx.OK | wx.ICON_ERROR)
                    self.error = 1
                    return

                address = os.path.dirname(self.annotation.config)
                p = open(os.path.join(address, self.video_list_with_address[self.index_video][
                                      self.find(self.video_list_with_address[self.index_video], os.sep)[-1] + 1:-1] + '_index_annotation.txt'), 'w')
                p2 = open(os.path.join(address, self.video_list_with_address[self.index_video][
                                      self.find(self.video_list_with_address[self.index_video], os.sep)[-1] + 1:-1] + '_index_annotation_auto.txt'), 'w')

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

        self.frames_id = np.asarray(range(start, end, 1)).astype(int)
        progress = wx.ProgressDialog("extraction in progress", "please wait", maximum=100, parent=self,
                                     style=wx.PD_SMOOTH | wx.PD_AUTO_HIDE)

        self.processSents()
        while success and count < start:
            count += 1
            success, image = self.cap.read()
		
        while success and count >= start and count < end:
            cv2.imwrite(self.address + os.sep + self.name + os.sep + 'frame_' + ("{:04d}".format(count)) + '.png',
                        image)
            progress.Update(int(count / len(self.frames_id)) * 100)
            count += 1
            success, image = self.cap.read()

           

        progress.Destroy()

        wx.Yield()

        self.cap.release()

    def processSents(self):
        wx.Yield()
