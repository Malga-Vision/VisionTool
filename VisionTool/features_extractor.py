import wx
import os
from nb_panel import *

class features_Extractor(nb_panel_features):
    # ----------------------------------------------------------------------
    def __init__(self, parent,size,address):
        """"""

        Menu = wx.Menu()
        self.nb = nb_panel_features.__init__(self,
                               parent=parent,
                               name="Features",
                               gui_size = size,
                               menu=Menu,
                               menuName="&Features",address= address)


    def update_Address(self,address):
        self.nb2 = nb_panel_features.update_address(address)
