"""
    Source file name: create_new_project.py  
    
    Description: this file allows VisionTool to create a new project, creating the correspondent configuration file   
    
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
from datetime import date
from VisionTool.New_project_features import *

class routine():

    def __init__(self, *args, **kwargs):

        super(routine, self).__init__(*args, **kwargs)

    
    
    def ask(self,parent=None, message='insert name for project'):
        default_value = "New_project_" + str(date.today())
        dlg = wx.TextEntryDialog(parent, message, value=default_value)
        dlg.ShowModal()
        result = dlg.GetValue()
        dlg.Destroy()
        return result
    
    
    def new_project(self):

        dlg = wx.DirDialog(None,"Choose directory where saving project", "",
                           wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
        if dlg.ShowModal() == wx.ID_OK:
            fdir = dlg.GetPath() + "/"
            dlg.SetPath(fdir)
        filename = self.ask()
        self.address = os.path.join(dlg.GetPath(),filename)
        try:
            os.mkdir(os.path.join(dlg.GetPath(),filename))
        except:
            error = wx.MessageBox("Folder already exists")
        if not os.path.isfile(os.path.join(self.address,"file_configuration.txt")):
            self.config_file = open(os.path.join(self.address,"file_configuration.txt"), "a")
            self.config_file.write(filename)
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

        if not os.path.isfile(os.path.join(self.address,"file_configuration.txt")):
            error = wx.MessageBox("Project folder is not valid: Configuration file not found!")
            return ""

        return self.address
