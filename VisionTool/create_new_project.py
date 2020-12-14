import wx
import os
from datetime import date
from New_project_features import *



class routine():

    def __init__(self, *args, **kwargs):

        super(routine, self).__init__(*args, **kwargs)

    def new_project(self):

        dlg = wx.DirDialog(None,"Choose directory where saving project", "",
                           wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
        if dlg.ShowModal() == wx.ID_OK:
            fdir = dlg.GetPath() + "/"
            dlg.SetPath(fdir)

        self.address = dlg.GetPath()+  "\\New_project_" + str(date.today())
        try:
            os.mkdir(dlg.GetPath() + "\\New_project_" + str(date.today()))
        except:
            error = wx.MessageBox("Folder already exists")
        if not os.path.isfile(self.address + "\\file_configuration.txt"):
            self.config_file = open(self.address + "\\file_configuration.txt", "a")
            self.config_file.write("New_project_" + str(date.today()))
            dlg.Destroy()
            self.config_file.close()
        else:
            error = wx.MessageBox("configuration file already exists")

        return self.address

    def open_existing_project(self):

        dlg = wx.DirDialog(None,"Choose project directory", "",
                           wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
        if dlg.ShowModal() == wx.ID_OK:
            fdir = dlg.GetPath() + "/"
            dlg.SetPath(fdir)

        self.address = dlg.GetPath()

        if not os.path.isfile(self.address + "\\file_configuration.txt"):
            error = wx.MessageBox("Project folder is not valid: Configuration file not found!")
            return ""

        return self.address