import wx
import os
from VisionTool.annotation import *


class nb_panel_features(wx.Panel):

    def __init__(self,parent,name,gui_size,menu,menuName,address):
        self.flag_was_already_open=0
        self.flag = 0
        wx.Panel.__init__(self,parent=parent)
        self.parent = parent
        self.address = address
        # wx.StaticBitmap(self, bitmap=wx.Bitmap(logo), pos=(100, 200))
        self.Menu = menu
        self.size = gui_size
        self.MenuName = menuName
        load_videos = wx.MenuItem(self.Menu, wx.ID_ANY, '&Select Videos\tCtrl+L')
        self.Menu.Append(load_videos)
        self.Menu.Bind(wx.EVT_MENU, self.Load_Videos, load_videos)
        self.Setsizer2()

    def Setsizer2(self):

        #img = wx.StaticBitmap(self, bitmap=wx.Bitmap(logo), pos=(100, 200))
        label = wx.StaticText(self, label="Features_Extraction")

    def Setsizer3(self):
        self.SetBackgroundColour('white')
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        label = wx.StaticText(self, label="Features_Extraction")
        self.sizer.Add(label, 0, wx.EXPAND)
        self.sizer.Add(self.frame, 1, wx.EXPAND,0)
        self.SetSizerAndFit(self.sizer)
        self.Layout()
        self.parent.Refresh()
        self.Refresh()




    def Load_Videos(self,e):

        if self.address == "":

            error = wx.MessageBox("you need to create a new project or open an existing one, first!")

        else:
            try:
                self.config_file = open(self.address + "\\file_configuration.txt", "r")
            except:
                    wx.MessageBox('Error in reading the configuration file, file not accessible \n '
                                  'Configuration_file_error'
                                  , 'Error!', wx.OK | wx.ICON_ERROR)
                    return
            count = 1
            a = self.config_file.readlines()
            self.config_file.close()
            if len(a) > 1:
                if wx.MessageBox(
                        "The project already contains videos.\n"
                        "Do you want to add new videos?", "Confirm",
                        wx.YES_NO | wx.NO_DEFAULT, self) == wx.NO:
                    self.frame = Label_frames(self, self.size, self.address + "\\file_configuration.txt")
                    self.Setsizer3()
                else:
                    self.flag_was_already_open = 1
                    self.upload_new_Videos()
            else:
                self.upload_new_Videos()

    def upload_new_Videos(self):

                with wx.FileDialog(self, "Select videos",wildcard='select video files (*.mp4;*.avi;*.mpeg4)|*.mp4;*.avi;*.mpeg4|''mp4 files (*.mp4)|*.mp4|'
                                                                 'avi files (*.avi)|*.avi|' 'mpeg4 (*.mpeg4)|*.mpeg4',
                                       style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST | wx.FD_MULTIPLE) as fileDialog:

                    if fileDialog.ShowModal() == wx.ID_CANCEL:
                        return  # the user changed their mind

                    if self.flag_was_already_open:
                        self.frame.Destroy()

                    pathname = fileDialog.GetPaths()

                # videos = os.listdir(dlg.GetPath())
                # if len(videos)==0:
                #     error = wx.MessageBox("Warning","Attention, no valid video has been selected")
                    try:
                        self.config_file = open(self.address + "\\file_configuration.txt", "r")
                    except:
                        wx.MessageBox('Error in reading the configuration file, file not accessible \n '
                                      'Configuration_file_error'
                                      , 'Error!', wx.OK | wx.ICON_ERROR)
                        return
                    count = 1
                    a = self.config_file.readlines()
                    self.config_file.close()

                    try:
                        self.config_file = open(self.address + "\\file_configuration.txt", "a")
                    except:
                        wx.MessageBox('Error in reading the configuration file, file not accessible \n '
                                      'Configuration_file_error'
                                      , 'Error!', wx.OK | wx.ICON_ERROR)
                        return
                    if len(a)>1:
                        for i in range(0, len(pathname)):
                            self.config_file.writelines(pathname[i] + '\n')
                    else:
                        self.config_file.writelines("\nVideo\n")

                        for i in range(0,len(pathname)):
                            self.config_file.writelines( pathname[i] + '\n')

                self.config_file.close()
                self.frame = Label_frames(self,self.size,self.address + "\\file_configuration.txt")
                self.Setsizer3()
    # open annotation interface

    def update_address(self,address):
        self.address = address

class nb_panel_pose(wx.Panel):
    def __init__(self,parent,name,menu,menuName,address):
        wx.Panel.__init__(self,parent=parent)
        self.address = address
        # media_path = 'C:\\Users\\vitop\\Desktop\\2020_WORK\\features selector\\'
        # logo = os.path.join(media_path, 'malga.png')
        # wx.StaticBitmap(self, bitmap=wx.Bitmap(logo), pos=(10, 250))
        self.Menu = menu
        self.MenuName = menuName
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        label = wx.StaticText(self,label="Pose estimation tool")
        sizer.Add(label)


    def update_address(self,address):
        self.address = address
