"""
    Source file name: interface_net.py  
    
    Description: this file contains the code to handle the GUI for setting the neural network preferences for svideo segmentation
    
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
import sys
import os

class Open_interface(wx.Frame):

    def __init__(self, parent, gui_size,address):
	
        wx.Frame.__init__(self, parent=parent)
        self.address = address
        # variable initilization
        self.method = "automatic"
        self.config = os.path.join(self.address , 'Architecture_Preferences.txt')
        
        # design the panel
        self.sizer = wx.GridBagSizer(5, 5)

        sb = wx.StaticBox(self, label="Optional Attributes")
        boxsizer = wx.StaticBoxSizer(sb, wx.VERTICAL)
    
        self.hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        self.hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        self.hbox3 = wx.BoxSizer(wx.HORIZONTAL)
    
        net_text = wx.StaticBox(self, label="Select the network")
        netboxsizer = wx.StaticBoxSizer(net_text, wx.VERTICAL)
        self.net_choice = wx.ComboBox(self, style=wx.CB_READONLY)
        options = ['unet','PSPnet','Linknet','FPN']
        self.net_choice.Set(options)
        self.net_choice.SetValue('Linknet')
        netboxsizer.Add(self.net_choice, 20, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)
    
        aug_text = wx.StaticBox(self, label="Select the backbone for your network")
        augboxsizer = wx.StaticBoxSizer(aug_text, wx.VERTICAL)
        self.aug_choice = wx.ComboBox(self, style=wx.CB_READONLY)
        options = ['vgg16' ,'vgg19', 'tensorpack', 'resnet18', 'resnet34', 'resnet50' ,'resnet101' ,'resnet152',
                   'seresnet18', 'seresnet34' ,'seresnet50' ,'seresnet101', 'seresnet152',
                   'resnext50' ,'resnext101','seresnext50' ,'seresnext101''senet154',
                   'densenet121' ,'densenet169' ,'densenet201',
                   'inceptionv3','inceptionresnetv2','mobilenet' ,'mobilenetv2',
                   'efficientnetb0', 'efficientnetb1', 'efficientnetb2' ,'efficientnetb3',
                   'efficientnetb4', 'efficientnetb5' ,'efficientnetb6' ,'efficientnetb7']
        self.aug_choice.Set(options)
        self.aug_choice.SetValue('resnet50')

        augboxsizer.Add(self.aug_choice, 20, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)
        # trainingindex_box = wx.StaticBox(self, label="Specify the trainingset index")
        # trainingindex_boxsizer = wx.StaticBoxSizer(trainingindex_box, wx.VERTICAL)
        # self.trainingindex = wx.SpinCtrl(self, value='0', min=0, max=100)
        # trainingindex_boxsizer.Add(self.trainingindex, 0, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)
    
        self.userfeedback = wx.RadioBox(self, label='Image net fine tuning?', choices=['Yes', 'No'], majorDimension=1,
                                        style=wx.RA_SPECIFY_COLS)
        self.userfeedback.SetSelection(0)

        image_text = wx.StaticBox(self, label="Saving Options")

        self.single_image = wx.StaticBoxSizer(image_text, wx.HORIZONTAL)

        self.userfeedback_images = wx.RadioBox(self, label='Single label image savings?', choices=['Yes', 'No'], majorDimension=1,
                                        style=wx.RA_SPECIFY_COLS)
        #
        # self.userfeedback_images2 = wx.RadioBox(self, label='Single label image savings2?', choices=['Yes', 'No'],
        #                                        majorDimension=1,
        #                                        style=wx.RA_SPECIFY_COLS)

        self.userfeedback_images.SetSelection(0)


        self.single_image.Add(self.userfeedback_images, 20, wx.EXPAND, 10)
        self.single_image.Add( self.userfeedback, 20, wx.EXPAND, 10)
        # self.single_image.Add(self.userfeedback_images2, 20, wx.EXPAND, 10)

        train_text = wx.StaticBox(self, label="training parameters")

        self.train_para = wx.StaticBoxSizer(train_text, wx.VERTICAL)

        self.train_label = wx.StaticText(self, label="Learning rate")

        self.text =  wx.TextCtrl(self, id=wx.ID_ANY)
        self.text.SetValue("0.001")

        self.Batch_text = wx.StaticText(self, label="Learning rate")

        self.Batch =  wx.TextCtrl(self, id=wx.ID_ANY)
        self.Batch.SetValue("1")
        self.train_para.Add(self.train_label)
        self.train_para.Add(self.text)
        self.train_para.Add( self.Batch_text)
        self.train_para.Add(self.Batch)
        self.loss = wx.RadioBox(self, label='Loss Function', choices=['Weighted Categorical_cross_entropy', 'Weighted Dice Loss'],
                                               majorDimension=1,
                                               style=wx.RA_SPECIFY_COLS)
        self.train_para.Add(self.loss)
        self.hbox1.Add(netboxsizer, 10, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)
        self.hbox1.Add(augboxsizer, 10, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)

    
        # self.model_comparison_choice = wx.RadioBox(self, label='Want to compare models?', choices=['Yes', 'No'],
        #                                            majorDimension=1, style=wx.RA_SPECIFY_COLS)
        # self.model_comparison_choice.SetSelection(1)
        #
        # self.model_comparison_choice.Bind(wx.EVT_RADIOBOX, self.chooseOption)




        networks = ['unet','PSPnet','Linknet','FPN']

        self.network_box = wx.StaticBox(self, label="Select the networks")
        self.network_boxsizer = wx.StaticBoxSizer(self.network_box, wx.VERTICAL)
        self.networks_to_compare = wx.CheckListBox(self, choices=networks, style=0, name="Select the model to compare")
        self.networks_to_compare.Bind(wx.EVT_CHECKLISTBOX, self.get_network_names)
        self.network_boxsizer.Add(self.networks_to_compare, 1, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)

        self.network_box.Hide()
        self.networks_to_compare.Hide()

        # self.hbox3.Add(self.model_comparison_choice, 10, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)
        self.hbox3.Add(self.network_boxsizer, 10, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)

        boxsizer.Add(self.train_para)
        boxsizer.Add(self.single_image)
        boxsizer.Add(self.hbox1, 0, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)
        boxsizer.Add(self.hbox2, 0, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)
        boxsizer.Add(self.hbox3, 0, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)
    
        self.sizer.Add(boxsizer, pos=(0, 0), span=(1, 5), flag=wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT, border=10)

        self.ok = wx.Button(self, label="Ok")
        self.sizer.Add(self.ok, pos=(1, 1),border=10)
        self.ok.Bind(wx.EVT_BUTTON, self.create_training_dataset)

        self.reset = wx.Button(self, label="Reset")
        self.sizer.Add(self.reset, pos=(1, 3), span=(1, 1), flag=wx.BOTTOM | wx.RIGHT, border=10)
        self.reset.Bind(wx.EVT_BUTTON, self.reset_create_training_dataset)
    
        self.sizer.AddGrowableCol(2)
    
        self.SetSizer(self.sizer)
        self.sizer.Fit(self)

        self.Layout()
        self.Show()
     

    def on_focus(self,event):
        pass



    def select_config(self,event):
        """
        """
        self.config = self.sel_config.GetPath()

    # def chooseOption(self,event):
    #     if self.model_comparison_choice.GetStringSelection() == 'Yes':
    #         self.network_box.Show()
    #         self.networks_to_compare.Show()
    #         self.net_choice.Enable(False)
    #         self.aug_choice.Enable(False)
    #
    #         self.SetSizer(self.sizer)
    #         self.sizer.Fit(self)
    #         self.get_network_names(event)
    #     else:
    #         self.net_choice.Enable(True)
    #         self.aug_choice.Enable(True)
    #         self.network_box.Hide()
    #         self.networks_to_compare.Hide()
    #
    #         self.SetSizer(self.sizer)
    #         self.sizer.Fit(self)

    def get_network_names(self,event):
        self.net_type = list(self.networks_to_compare.GetCheckedStrings())

    def create_training_dataset(self,event):
        userfeedback_option = self.userfeedback.GetStringSelection()
        try:
            file = open(self.config,'w')
            file.write('NN\n')
            file.write(self.net_choice.GetStringSelection() + '\n')
            file.write('Backbone\n')
            file.write(self.aug_choice.GetStringSelection() + '\n')
            file.write('Image net pre-trained weights\n')
            file.write(userfeedback_option + '\n')
            file.write('Single labels\n')
            file.write(str(self.userfeedback_images.GetStringSelection()) + '\n')
            file.write('Learning rate\n')
            file.write(str(self.text.GetValue())+ '\n')
            file.write('Loss function\n')
            file.write(str(self.loss.GetStringSelection()) + '\n')
            file.close()
        except:
            wx.MessageBox('Error in writing the configuration file, please check permissions \n '
                          'Configuration_file_error'
                          , 'Error!', wx.OK | wx.ICON_ERROR)

    def show(self,parent,gui_size,config):
        app = wx.App()
        frame = Open_interface(None, gui_size, config).Show()
        app.MainLoop()

    def reset_create_training_dataset(self,event):
        """
        Reset to default
        """
        try:
            os.remove(self.config)
        except:
            pass

        self.model_comparison_choice.SetSelection(1)
        self.userfeedback.SetSelection(0)
        self.network_box.Hide()
        self.networks_to_compare.Hide()
        self.net_choice.Enable(True)
        self.aug_choice.Enable(True)
        self.userfeedback_images.SetSelection(0)
        self.net_choice.SetValue('Linknet')
        self.aug_choice.SetValue('resnet50')
        self.SetSizer(self.sizer)
        self.sizer.Fit(self)
        self.Layout()
