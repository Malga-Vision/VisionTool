import wx
import os
from VisionTool.annotation import *
import cv2
import numpy as np
import shutil
from VisionTool.New_project_features import *
import random


class extract_Frames(wx.Panel):

    def __init__(self,parent,video_list_with_address,index_video,annotation_preferences,index_type,config,import_from_deeplabcut_flag):

        wx.Panel.__init__(self, parent=parent)
        self.config = config
        self.import_from_deeplabcut_flag = import_from_deeplabcut_flag
        # uniform
        self.annotation = annotation_preferences
        self.video_list_with_address = video_list_with_address
        self.index_video = index_video
        self.name = 'Extracted_frames_' + self.video_list_with_address[self.index_video][self.find(self.video_list_with_address[self.index_video], '\\')[-1] + 1:-1]

        if not os.path.isfile(os.path.dirname(self.config) + '//annotation_options.txt'):

            Frame_selection(self, os.path.dirname(self.config) + '//annotation_options.txt')
        else:
            if wx.MessageBox(
                    "A set of preferences has already been saved.\n"
                    "Do you want to change your preferences?", "Confirm",
                    wx.YES_NO | wx.YES_DEFAULT, self) == wx.YES:
                os.remove(os.path.dirname(self.config)+ '//annotation_options.txt')
                Frame_selection(self,os.path.dirname(self.config)+ '//annotation_options.txt')



        if index_type == 0:
            self.uniform()
            self.read_videos_for_length()
        elif index_type == 1:
            self.k_means()
        else:
            self.manual()

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
        #get preferences in textbox annotation panel
        start = int(self.annotation.get_Text(1))
        end = int(self.annotation.get_Text(2))

        self.address = os.path.dirname(self.config)

        if not os.path.isdir(self.address + '\\' + self.name):
            os.mkdir(self.address + '\\' + self.name)

            num_frame_annotated = int(self.annotation.get_Text(3))
            num_frame_automatic = int(self.annotation.get_Text(4))

            my_list = list(range(start,
                                 end))  # list of integers from 1 to 99                # adjust this boundaries to fit your needs
            random.shuffle(my_list)
            if num_frame_annotated != 0 or self.import_from_deeplabcut_flag == 1:
                if num_frame_annotated < (end - start):
                    if self.import_from_deeplabcut_flag:
                        frames = self.annotation.Get_annotation_for_deeplabcut_compat()
                    else:
                        frames = my_list[0:num_frame_annotated]
                    frames_annotated = my_list[num_frame_annotated:num_frame_automatic + num_frame_annotated]
                address = os.path.dirname(self.annotation.config)
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
                wx.YES_NO | wx.NO_DEFAULT, self)
            if permission == 2:
                shutil.rmtree(self.address + '\\' + self.name)
                os.mkdir(self.address + '\\' + self.name)



                num_frame_annotated = int(self.annotation.get_Text(3))
                num_frame_automatic = int(self.annotation.get_Text(4))

                my_list = list(range(start,
                                     end))  # list of integers from 1 to 99                # adjust this boundaries to fit your needs
                random.shuffle(my_list)
                if num_frame_annotated != 0 or self.import_from_deeplabcut_flag==1:
                        if num_frame_annotated < (end -start):
                            if self.import_from_deeplabcut_flag:
                                frames = self.annotation.Get_annotation_for_deeplabcut_compat()
                            else:
                                frames = my_list[0:num_frame_annotated]
                            frames_annotated = my_list[num_frame_annotated:num_frame_automatic + num_frame_annotated]
                        address = os.path.dirname(self.annotation.config)
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
                
        self.frames_id =np.asarray(range(start,end,1)).astype(int)
        progress = wx.ProgressDialog("extraction in progress", "please wait", maximum=100, parent=self,
                                     style=wx.PD_SMOOTH | wx.PD_AUTO_HIDE)

        self.processSents()
        while success and count<start:
            count += 1
            success, image = self.cap.read()
        while success and count >=start and count < end:
            success, image = self.cap.read()
            progress.Update(int(count/len(self.frames_id))*100)
            cv2.imwrite(self.address + '\\' + self.name + '\\' + 'frame_' + ("{:04d}".format(count)) + '.png', image)
            count += 1

        progress.Destroy()

        self.cap.release()


    def processSents(self):
        wx.Yield()
