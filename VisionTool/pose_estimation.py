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
