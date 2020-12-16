"""
    Source file name: pose_estimation.py  
    
    Description: this file contains the code to handle the GUI for the pose estimation section of VisionTool
    
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


import wx
import os
from VisionTool.nb_panel import *



class pose_estimation(nb_panel_pose):
    # ----------------------------------------------------------------------
    def __init__(self, parent,address):
        """"""
        Menu = wx.Menu()
        Menu.Append(wx.ID_ANY, "Pose estimation", "TBD")
        self.panel = nb_panel_pose.__init__(self,
                               parent=parent,
                               name="Pose",
                               menu=Menu,
                               menuName="&Pose",address= address)
        self.SetBackgroundColour("Blue")



    def update_address(self, address):
        nb_panel_pose.update_address(self,address)
