"""
    Source file name: New_project_features.py  
    
    Description: this file contains the code to save the preferences for annotation (e.g., name of interesting points and annotator name)
    
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

class Frame_selection(wx.Frame):
    """Contains the main GUI and button boxes"""

    def __init__(self,parent,config):

        self.config = config
        screenWidth = 800
        screenHeight = 800
        self.gui_size = (800,800)

        super().__init__(parent, size = self.gui_size, title='Insert_preferences')


        label1 = wx.StaticText(self, label="Bodyparts (separated by ;)", pos=(70, 80))
        self.text = wx.TextCtrl(self, id=wx.ID_ANY, pos=(70, 100))
        label2 = wx.StaticText(self, label="Name_user", pos=(70, 150))
        self.text2 = wx.TextCtrl(self, id=wx.ID_ANY, pos=(70, 170))
        self.text2.SetValue('user_1')
        label3 = wx.StaticText(self, label="MarkerSize", pos=(250, 80))
        self.text3 = wx.TextCtrl(self, id=wx.ID_ANY, pos=(250, 100))
        self.text3.SetValue(str(13))
        label4 = wx.StaticText(self, label="Alpha", pos=(250, 150))
        self.text4 = wx.TextCtrl(self, id=wx.ID_ANY, pos=(250, 170))
        self.text4.SetValue(str(0.7))
        label5 = wx.StaticText(self, label="Colormap", pos=(70, 200))
        self.text5 = wx.TextCtrl(self, id=wx.ID_ANY, pos=(70, 220))
        self.text5.SetValue('inferno')
        self.Button1 = wx.Button(self, id=wx.ID_ANY, pos=(150,300),label="Finish")
        self.Button1.Bind(wx.EVT_BUTTON,self.Button)
        self.Bind(wx.EVT_CLOSE, self.OnClose)
        self.ShowModal()

        # (Explicit call to MakeModal)

    def OnClose(self, event):
        """
        To handle closing the windows because you need to exit the eventLoop
        of the modal window.
        """
        del self._disabler
        self.eventLoop.Exit()
        self.Destroy()

    def ShowModal(self):
        """
        This function is the one giving the wx.FileDialog behavior
        """
        self._disabler = wx.WindowDisabler(self)
        self.Show()
        self.eventLoop = wx.GUIEventLoop()
        self.eventLoop.Run()


    def Button(self,event):
        file = open(self.config, 'w')
        file.write('Name_user\n')
        file.write(self.text2.GetValue())
        file.write('\nMarker Size\n')
        file.write(self.text3.GetValue())
        file.write('\nAlpha\n')
        file.write(self.text4.GetValue())
        file.write('\nColor map\n')
        file.write(self.text5.GetValue())
        file.write('\nBodyparts\n')
        Bodyparts = self.text.GetValue()
        body_parts = Bodyparts.split(';')
        for item in body_parts:
            file.write(item +  '\n')
        file.close()
        self.Close()



    if __name__ == '__main__':
        main()
