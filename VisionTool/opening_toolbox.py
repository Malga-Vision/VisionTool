"""
    Source file name: opening_toolbox.py  
    
    Description: this file contains the code to handle the GUI for actually performing the annotation of frames into VisionTool. 
    
    Code adapted and modified from: 
    Alexander Mathis, Pranav Mamidanna, Kevin M Cury, Taiga Abe, Venkatesh N Murthy,Mackenzie Weygandt Mathis, and Matthias Bethge. Deeplabcut: markerless pose estimation
    of user-defined body parts with deep learning. Nature neuroscience, 21(9):1281{1289, 2018.
    
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



import os
import wx
import cv2
import os
import os.path
import glob
import cv2
from shutil import copyfile
import wx
import wx.lib.scrolledpanel as SP
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
import argparse
from VisionTool.auxfun_drag_label import*
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar
from VisionTool.annotation import *
#automatic annotation
from VisionTool.architectures_segmentation import unet
from VisionTool.training import *



class ImagePanel(wx.Panel):
    def __init__(self, parent,config, gui_size, **kwargs):
        h = gui_size[0] / 2
        w = gui_size[1] / 3
        wx.Panel.__init__(self, parent, -1, style=wx.SUNKEN_BORDER, size=(h, w))

        self.figure = matplotlib.figure.Figure()
        self.axes = self.figure.add_subplot(1, 1, 1)
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.orig_xlim = None
        self.orig_ylim = None
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        self.SetSizer(self.sizer)
        self.Fit()

    def getfigure(self):
        return (self.figure)

    def drawplot(self, img, img_name, itr, index, bodyparts, cmap, keep_view=False):
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        self.axes.clear()
        img = cv2.imread(img)
        # convert the image to RGB as you are showing the image with matplotlib
        im = img[..., ::-1]
        ax = self.axes.imshow(im, cmap=cmap)
        self.orig_xlim = self.axes.get_xlim()
        self.orig_ylim = self.axes.get_ylim()
        divider = make_axes_locatable(self.axes)
        colorIndex = np.linspace(np.min(im), np.max(im), len(bodyparts))

        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # cbar = self.figure.colorbar(ax, cax=cax, spacing='proportional', ticks=colorIndex)
        # cbar.set_ticklabels(bodyparts[::-1])
        colorIndex = colorIndex.astype(int)
        color = cmap(colorIndex)


        patchList = []
        for key in range(0, len(bodyparts)):
            data_key = Line2D([0],[0], marker='o',color='black', markerfacecolor=color[len(bodyparts)-key-1],label=bodyparts[key],markersize=8)
            patchList.append(data_key)
        self.axes.legend(handles=patchList,bbox_to_anchor=(1.1, 1.1))
        self.axes.set_title(str(str(itr) + "/" + str(len(index) - 1) + " " + img_name))
        if keep_view:
            self.axes.set_xlim(xlim)
            self.axes.set_ylim(ylim)
        self.toolbar = NavigationToolbar(self.canvas)
        return (self.figure, self.axes, self.canvas, self.toolbar)

    def resetView(self):
        self.axes.set_xlim(self.orig_xlim)
        self.axes.set_ylim(self.orig_ylim)

    def getColorIndices(self, img, bodyparts):
        """
        Returns the colormaps ticks and . The order of ticks labels is reversed.
        """
        im = cv2.imread(img)
        norm = mcolors.Normalize(vmin=0, vmax=np.max(im))
        ticks = np.linspace(0, np.max(im), len(bodyparts))[::-1]

        return norm, ticks


class WidgetPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, -1, style=wx.SUNKEN_BORDER)


class ScrollPanel(SP.ScrolledPanel):
    def __init__(self, parent):
        SP.ScrolledPanel.__init__(self, parent, -1, style=wx.SUNKEN_BORDER)
        self.SetupScrolling(scroll_x=True, scroll_y=True, scrollToTop=False)
        self.Layout()

    def on_focus(self, event):
        pass

    def addRadioButtons(self, bodyparts, fileIndex, markersize):
        """
        Adds radio buttons for each bodypart on the right panel
        """
        self.choiceBox = wx.BoxSizer(wx.VERTICAL)
        choices = [l for l in bodyparts]
        self.fieldradiobox = wx.RadioBox(self, label='Select a bodypart to label',
                                         style=wx.RA_SPECIFY_ROWS, choices=choices)
        self.slider = wx.Slider(self, -1, markersize, 1, markersize * 3, size=(250, -1),
                                style=wx.SL_HORIZONTAL | wx.SL_AUTOTICKS | wx.SL_LABELS)
        self.slider.Enable(False)
        self.checkBox = wx.CheckBox(self, id=wx.ID_ANY, label='Adjust marker size.')
        self.choiceBox.Add(self.slider, 0, wx.ALL, 5)
        self.choiceBox.Add(self.checkBox, 0, wx.ALL, 5)
        self.choiceBox.Add(self.fieldradiobox, 0, wx.EXPAND | wx.ALL, 10)
        self.SetSizerAndFit(self.choiceBox)
        self.Layout()
        return (self.choiceBox, self.fieldradiobox, self.slider, self.checkBox)

    def clearBoxer(self):
        self.choiceBox.Clear(True)


class MainFrame(wx.Frame):
    """Contains the main GUI and button boxes"""

    def __init__(self,parent, video_list_with_address,index_video,Label_frame, extract_frame_option, config, imtypes):
        # Settting the GUI size and panels design
        self.config = config
        self.counter_frames = 0
        preferences_file =  os.path.dirname(self.config) + '//annotation_options.txt'
        self.Label_frame = Label_frame
        if os.path.isfile(os.path.join(os.path.dirname(self.config), '_index_annotation.txt')):
            self.pref_ann = open(os.path.join(os.path.dirname(self.config), '_index_annotation.txt'), 'r')
            temporary = self.pref_ann.readlines()
            for i in range(0, len(temporary)):
                temporary[i] = temporary[i][:-1]
            temporary = np.asarray(temporary)
            self.frame_selected_for_annotation= temporary.astype(int)
            self.frame_selected_for_annotation=np.sort(self.frame_selected_for_annotation)
        if os.path.isfile(os.path.join(os.path.dirname(self.config), '_index_annotation_auto.txt')):
            self.pref_ann = open(os.path.join(os.path.dirname(self.config), '_index_annotation_auto.txt'), 'r')
            temporary = self.pref_ann.readlines()
            for i in range(0, len(temporary)):
                temporary[i] = temporary[i][:-1]
            temporary = np.asarray(temporary)
            self.frame_selected_for_annotation_auto= temporary.astype(int)
            self.frame_selected_for_annotation_auto=np.sort(self.frame_selected_for_annotation_auto)
        # Gets the size of each display
        displays = (wx.Display(i) for i in range(wx.Display.GetCount()))  # Gets the number of displays
        screenSizes = [display.GetGeometry().GetSize() for display in displays]  # Gets the size of each display
        index = 0  # For display 1.
        screenWidth = screenSizes[index][0]
        screenHeight = screenSizes[index][1]
        self.gui_size = (screenWidth * 0.85, screenHeight * 0.85)
        wx.Frame.__init__(self, parent, id=wx.ID_ANY, title='Annotation interface',
                          size=wx.Size(self.gui_size), pos=wx.DefaultPosition,
                          style=wx.RESIZE_BORDER | wx.DEFAULT_FRAME_STYLE | wx.TAB_TRAVERSAL)
        try:
            file = open(preferences_file)
            self.pref = file.readlines()
        except:
            wx.MessageBox('Error in reading the network configuration file, please check the existence or choose'
                          'preferences again \n '
                          'Preferences_file_error'
                          , 'Error!', wx.OK | wx.ICON_ERROR)
            self.Destroy()

            return

        for i in range(0, len(self.pref)):
            self.pref[i] = self.pref[i][:-1]
        self.options_frame = extract_frame_option
        try:
            config2 = open(config, 'r')
        except:

            wx.MessageBox('Error in reading the configuration file \n '
                      'Configuration file missing'
                      , 'Error!', wx.OK | wx.ICON_ERROR)
            self.Destroy()
            return

        self.address_video = config2.readlines()[1:]
        self.video_list_with_address = video_list_with_address
        self.index_video = index_video
        self.address = os.path.dirname(self.config)
        index = self.address_video[0].find(':')
        self.address_video = self.address_video[0][index+1:]
        self.scorer = self.pref[1]
        self.videos = self.video_list_with_address[self.index_video][:-1]
        self.markerSize = int(self.pref[3])
        self.alpha = float(self.pref[5])
        self.map = self.pref[7]
        try:
            self.colormap = plt.get_cmap(self.map)
        except:
            wx.MessageBox(self.map + 'is not recognized as a valid map \n Settings color map to Pastel2'
              , 'Error!', wx.OK | wx.ICON_ERROR)
            self.colormap = plt.get_cmap('inferno')

        self.colormap = self.colormap.reversed()
        config2.close()
        self.title_video = self.video_list_with_address[self.index_video][self.find(self.video_list_with_address[self.index_video], os.sep)[-1] + 1:-1]
        self.name = 'Extracted_frames_' + self.title_video
        self.filename = self.address + "//Annotation_" + self.title_video + '_' + self.scorer

        self.imtypes = imtypes  # imagetypes to look for in folder e.g. *.png

        self.statusbar = self.CreateStatusBar()
        self.statusbar.SetStatusText("Looking for a folder to start labeling. Click 'Load frames' to begin.")
        self.Bind(wx.EVT_CHAR_HOOK, self.OnKeyPressed)

        self.SetSizeHints(wx.Size(self.gui_size))  # This sets the minimum size of the GUI. It can scale now!
        ###################################################################################################################################################

        # Spliting the frame into top and bottom panels. Bottom panels contains the widgets. The top panel is for showing images and plotting!

        topSplitter = wx.SplitterWindow(self)
        vSplitter = wx.SplitterWindow(topSplitter)

        self.image_panel = ImagePanel(vSplitter,config, self.gui_size)
        self.choice_panel = ScrollPanel(vSplitter)
        vSplitter.SplitVertically(self.image_panel, self.choice_panel, sashPosition=self.gui_size[0] * 0.8)
        vSplitter.SetSashGravity(1)
        self.widget_panel = WidgetPanel(topSplitter)
        topSplitter.SplitHorizontally(vSplitter, self.widget_panel, sashPosition=self.gui_size[1] * 0.83)  # 0.9
        topSplitter.SetSashGravity(1)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(topSplitter, 1, wx.EXPAND)
        self.SetSizer(sizer)

        ###################################################################################################################################################
        # Add Buttons to the WidgetPanel and bind them to their respective functions.

        widgetsizer = wx.WrapSizer(orient=wx.HORIZONTAL)
        self.load = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Load frames")
        widgetsizer.Add(self.load, 1, wx.ALL, 15)
        self.load.Bind(wx.EVT_BUTTON, self.browseDir)

        self.prev = wx.Button(self.widget_panel, id=wx.ID_ANY, label="<<Previous")
        widgetsizer.Add(self.prev, 1, wx.ALL, 15)
        self.prev.Bind(wx.EVT_BUTTON, self.prevImage)
        self.prev.Enable(False)

        self.next = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Next>>")
        widgetsizer.Add(self.next, 1, wx.ALL, 15)
        self.next.Bind(wx.EVT_BUTTON, self.nextImage)
        self.next.Enable(False)

        # self.next_labeled = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Next_annotate>>")
        # widgetsizer.Add(self.next_labeled, 1, wx.ALL, 15)
        # self.next_labeled.Bind(wx.EVT_BUTTON, self.nextImage_ann)
        # self.next_labeled.Enable(False)
        self.next_labeled = wx.CheckBox(self.widget_panel, label='annot. only')
        self.next_labeled.Enable(False)
        self.next_labeled_annotated = wx.CheckBox(self.widget_panel, label='annot. auto only')
        self.next_labeled_annotated.Enable(False)

        widgetsizer.Add(self.next_labeled, 1, wx.ALL, 15)
        self.next_labeled.Bind(wx.EVT_CHECKBOX, self.next_labeled_check)

        widgetsizer.Add(self.next_labeled_annotated, 1, wx.ALL, 15)
        self.next_labeled_annotated.Bind(wx.EVT_CHECKBOX, self.next_labeled_annotated_check)

        self.help = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Help")
        widgetsizer.Add(self.help, 1, wx.ALL, 15)
        self.help.Bind(wx.EVT_BUTTON, self.helpButton)
        self.help.Enable(True)
        #
        self.zoom = wx.ToggleButton(self.widget_panel, label="Zoom")
        widgetsizer.Add(self.zoom, 1, wx.ALL, 15)
        self.zoom.Bind(wx.EVT_TOGGLEBUTTON, self.zoomButton)
        self.widget_panel.SetSizer(widgetsizer)
        self.zoom.Enable(False)

        self.home = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Home")
        widgetsizer.Add(self.home, 1, wx.ALL, 15)
        self.home.Bind(wx.EVT_BUTTON, self.homeButton)
        self.widget_panel.SetSizer(widgetsizer)
        self.home.Enable(False)

        self.pan = wx.ToggleButton(self.widget_panel, id=wx.ID_ANY, label="Pan")
        widgetsizer.Add(self.pan, 1, wx.ALL, 15)
        self.pan.Bind(wx.EVT_TOGGLEBUTTON, self.panButton)
        self.widget_panel.SetSizer(widgetsizer)
        self.pan.Enable(False)

        self.lock = wx.CheckBox(self.widget_panel, id=wx.ID_ANY, label="Lock View")
        widgetsizer.Add(self.lock, 1, wx.ALL, 15)
        self.lock.Bind(wx.EVT_CHECKBOX, self.lockChecked)
        self.widget_panel.SetSizer(widgetsizer)
        self.lock.Enable(False)

        self.save = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Save")
        widgetsizer.Add(self.save, 1, wx.ALL, 15)
        self.save.Bind(wx.EVT_BUTTON, self.saveDataSet)
        self.save.Enable(False)

        #widgetsizer.AddStretchSpacer(5)
        self.quit = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Quit")
        widgetsizer.Add(self.quit, 1, wx.ALL, 15)
        self.quit.Bind(wx.EVT_BUTTON, self.quitButton)

        self.annotation = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Help in annotation")
        widgetsizer.Add(self.annotation, 1, wx.ALL, 15)
        self.annotation.Bind(wx.EVT_BUTTON, self.annotation_help)
        self.cancel_annotation = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Cancel")
        widgetsizer.Add(self.cancel_annotation, 1, wx.ALL, 15)
        self.cancel_annotation.Bind(wx.EVT_BUTTON, self.annotation_reset)
        self.cancel_annotation.Enable(False)
        self.annotation.Enable(False)
        self.next_labeled_annotated.Enable(False)
        self.widget_panel.SetSizer(widgetsizer)
        self.widget_panel.SetSizerAndFit(widgetsizer)
        self.widget_panel.Layout()

        ###############################################################################################################################
        # Variables initialization

        self.currentDirectory = os.getcwd()
        self.iter = []
        self.file = 0
        self.updatedCoords = []
        self.dataFrame = None
        self.config_file = config
        self.new_labels = False
        self.buttonCounter = []
        self.bodyparts2plot = []
        self.drs = []
        self.num = []
        self.view_locked = False
        # Workaround for MAC - xlim and ylim changed events seem to be triggered too often so need to make sure that the
        # xlim and ylim have actually changed before turning zoom off
        self.prezoom_xlim = []
        self.prezoom_ylim = []

    ###############################################################################################################################
    # BUTTONS FUNCTIONS FOR HOTKEYS
    def OnKeyPressed(self, event=None):
        if event.GetKeyCode() == wx.WXK_RIGHT:
            self.nextImage(event=None)
        elif event.GetKeyCode() == wx.WXK_LEFT:
            self.prevImage(event=None)
        elif event.GetKeyCode() == wx.WXK_DOWN:
            self.nextLabel(event=None)
        elif event.GetKeyCode() == wx.WXK_UP:
            self.previousLabel(event=None)

    def activateSlider(self, event):
        """
        Activates the slider to increase the markersize
        """
        self.checkSlider = event.GetEventObject()
        if self.checkSlider.GetValue() == True:
            self.activate_slider = True
            self.slider.Enable(True)
            MainFrame.updateZoomPan(self)
        else:
            self.slider.Enable(False)

    def OnSliderScroll(self, event):
        """
        Adjust marker size for plotting the annotations
        """
        MainFrame.saveEachImage(self)
        MainFrame.updateZoomPan(self)
        self.buttonCounter = []
        self.markerSize = self.slider.GetValue()
        img_name = Path(self.index[self.iter]).name
        self.figure, self.axes, self.canvas, self.toolbar = self.image_panel.drawplot(self.address + os.sep + self.name + os.sep+self.img, img_name, self.iter,
                                                                                      self.index, self.bodyparts,
                                                                                      self.colormap, keep_view=True)

        self.axes.callbacks.connect('xlim_changed', self.onZoom)
        self.axes.callbacks.connect('ylim_changed', self.onZoom)
        self.buttonCounter = MainFrame.plot(self, self.img)

    def quitButton(self, event):
        """
        Asks user for its inputs and then quits the GUI
        """
        self.statusbar.SetStatusText("Quitting now!")

        nextFilemsg = wx.MessageBox('Are you sure you want to quit?', 'Quit?',
                                    wx.YES_NO | wx.ICON_INFORMATION)
        if nextFilemsg == 2:
            self.Destroy()
            print(
                "You can now proceed with the deep learning-based analysis for the labels")


        else:
            self.save.Enable(True)

    def highlight_max(self,s):
        '''
        highlight the maximum in a Series yellow.
        '''
        annotated = np.where(np.isnan(self.dataFrame.iloc[:, 0].values) == False)[0]
        color = 'black' if val in annotated  else 'red'
        return 'color: %s' % color

    def prediction_to_annotation(self):
        # load the predictions
        self.dataFrame.to_pickle(self.filename+'_MANUAL')  # where to save it, usually as a .pkl

        predicted_Address = (self.address + '\\prediction')
        imlist = os.listdir(self.address + os.sep + self.name)
        annotated = np.where(np.isnan(self.dataFrame.iloc[:, 0].values) == False)[0]

        for j in range(0, len(imlist)):
            if j in annotated:
                continue
            for i in range(1, len(self.bodyparts)+1):
                a = cv2.imread(os.path.join(predicted_Address + os.sep,
                                            ("{:02d}".format(i)) + self.relativeimagenames[j][7:]))
                a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
                thresh, a = cv2.threshold(a, 0.99, 1, cv2.THRESH_BINARY)
                contour, hierarchy = cv2.findContours(a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if contour != []:
                    max_area = 0
                    i_max = []
                    for cc in contour:
                        mom = cv2.moments(a)
                        area = mom['m00']
                        if area > max_area:
                            max_area = area
                            i_max = cc

                    center = cv2.moments(i_max)
                    xc = center['m10'] / center['m00']
                    yc = center['m01'] / center['m00']
                    self.dataFrame[self.dataFrame.columns[(i-1)*2]].values[j] = -xc
                    self.dataFrame[self.dataFrame.columns[(i-1)*2+1]].values[j] = yc
                else:
                    #maybe one joint is missing, but the other were correctly identified
                    self.dataFrame[self.dataFrame.columns[(i - 1) * 2]].values[j] = -1
                    self.dataFrame[self.dataFrame.columns[(i - 1) * 2 + 1]].values[j] = -1


        self.statusbar.SetStatusText("File saved")
        MainFrame.updateZoomPan(self)

        copyfile(os.path.join(self.filename + ".csv"), os.path.join(self.filename + "_MANUAL.csv"))
        self.dataFrame.to_csv(os.path.join(self.filename + ".csv"))
        copyfile(os.path.join(self.filename + ".csv"), os.path.join(self.filename + "_MANUAL.csv"))
        self.dataFrame.to_pickle(self.filename)  # where to save it, usually as a .pkl
        wx.PyCommandEvent(wx.EVT_BUTTON.typeId, self.load.GetId())

    def annotation_reset(self,event):

        try:

            self.dataFrame = pd.read_pickle(self.filename+'_MANUAL')
            os.remove(self.filename)
            self.dataFrame.to_pickle(self.filename)  # where to save it, usually as a .pkl

            os.remove(os.path.join(self.filename + ".csv"))
            os.rename(os.path.join(self.filename + "_MANUAL.csv"),os.path.join(self.filename + ".csv"))
            self.cancel_annotation.Enable(False)
            self.next_labeled_annotated.Enable(False)

        except:
            pass

    def find(self,s, ch):
            return [i for i, ltr in enumerate(s) if ltr == ch]

    def deep_lab_cut_conversion_dataframe(self):

        imlist = os.listdir(self.address + os.sep + self.name)
        for i in range(0, len(imlist)):
            imlist[i] = 'labeled' + imlist[i]
        a = np.empty((len(self.index), 2,))
        self.dataFrame3 = None
        a[:] = np.nan
        for bodypart in self.bodyparts:
            index = pd.MultiIndex.from_product([[self.scorer], [bodypart], ['x', 'y']],
                                               names=['scorer', 'bodyparts', 'coords'])
            frame = pd.DataFrame(a, columns=index, index=imlist)
            self.dataFrame3 = pd.concat([self.dataFrame3, frame], axis=1)
        num_columns = len(self.dataFrame3.columns)
        for j in range(0, num_columns):
            for i in range(3, len(self.dataFrame[self.dataFrame.columns[0]].values)):
                temp = self.dataFrame[self.dataFrame.columns[0]].values[i]
                index = self.find(temp, os.sep)
                index = index[len(index) - 1]
                index3 = self.find(temp, '.')
                index3 = index3[len(index3) - 1]
                index2 = temp.find('img')
                a = int(temp[index2 + 3:index3])
                self.dataFrame3[self.dataFrame3.columns[j]].values[a] = \
                self.dataFrame[self.dataFrame.columns[j+1]].values[i]




        # for i in range(3, len(self.dataFrame[self.dataFrame.columns[0]].values)):
        #     temp = self.dataFrame[self.dataFrame.columns[0]].values[i]
        #     index = self.find(temp, os.sep)
        #     index = index[len(index) - 1]
        #     index3 = self.find(temp, '.')
        #     index3 = index3[len(index3) - 1]
        #     index2 = temp.find('img')
        #     a = int(temp[index2 + 3:index3])
        #     self.dataFrame3[self.dataFrame3.columns[0]].values[a] = self.dataFrame[self.dataFrame.columns[0]].values[i]
        #

    def annotation_help(self, event):
        """
        Asks user for its inputs and then quits the GUI
        """

        self.cancel_annotation.Enable(True)
        self.next_labeled_annotated.Enable(True)

        self.statusbar.SetStatusText("Running segmentation network-based support")


        nextFilemsg = wx.MessageBox('Are you sure you want annotation-assisted?', 'Confirm?',
                                    wx.YES_NO | wx.ICON_INFORMATION)
        try:
            self.dataFrame = pd.read_pickle(self.filename)
        except:
            wx.MessageBox('Are you sure you saved your annotation masks? \n '
                          'Annotation not found'
                          , 'Error!', wx.OK | wx.ICON_ERROR)
            return
        if nextFilemsg == 2:
            print(
                "Proceeding with assistance")

            if os.path.isfile(self.address + '//Unet.h5'):
                dlg = wx.MessageDialog(None,
                                       "Do you want to use the already trained network?",
                                       "Found existing trained architecture", wx.YES_NO | wx.ICON_WARNING)
                result = dlg.ShowModal()
                if result == wx.ID_NO:
                    self.flag_train =1
                else:
                    self.flag_train = 0
            else:
                self.flag_train = 1

            a = training(address=self.address, file_annotation=self.filename,
                     image_folder=self.address + os.sep + self.name + os.sep,
                     annotation_folder=os.path.join(self.address, self.name + 'annotation'), bodyparts=self.bodyparts,
                     train_flag = self.flag_train,annotation_assistance=1)
            if a.error !=-1:
                self.dataFrame = pd.read_pickle(self.filename)
                pass
        else:
            return



    def helpButton(self, event):
        """
        Opens Instructions
        """
        MainFrame.updateZoomPan(self)
        wx.MessageBox(
            '1. Select one of the body parts from the radio buttons to add a label (if necessary change config.yaml first to edit the label names). \n\n2. Right clicking on the image will add the selected label and the next available label will be selected from the radio button. \n The label will be marked as circle filled with a unique color.\n\n3. To change the marker size, mark the checkbox and move the slider. \n\n4. Hover your mouse over this newly added label to see its name. \n\n5. Use left click and drag to move the label position.  \n\n6. Once you are happy with the position, right click to add the next available label. You can always reposition the old labels, if required. You can delete a label with the middle button mouse click. \n\n7. Click Next/Previous to move to the next/previous image.\n User can also add a missing label by going to a previous/next image and using the left click to add the selected label.\n NOTE: the user cannot add a label if the label is already present. \n\n8. When finished labeling all the images, click \'Save\' to save all the labels as a .h5 file. \n\n9. Click OK to continue using the labeling GUI.',
            'User instructions', wx.OK | wx.ICON_INFORMATION)
        self.statusbar.SetStatusText("Help")

    def homeButton(self, event):
        self.image_panel.resetView()
        self.figure.canvas.draw()
        MainFrame.updateZoomPan(self)
        self.zoom.SetValue(False)
        self.pan.SetValue(False)
        self.statusbar.SetStatusText("")

    def panButton(self, event):
        if self.pan.GetValue() == True:
            self.toolbar.pan()
            self.statusbar.SetStatusText("Pan On")
            self.zoom.SetValue(False)
        else:
            self.toolbar.pan()
            self.statusbar.SetStatusText("Pan Off")

    def next_labeled_check(self,event):
        if self.next_labeled.GetValue():
            self.next_labeled_annotated.SetValue(0)
            self.counter_frames = 0
        else:
            self.next_labeled.SetValue(0)


    def next_labeled_annotated_check(self, event):

        if self.next_labeled_annotated.GetValue():

            self.next_labeled.SetValue(0)
            self.counter_frames = 0
        else:
            self.next_labeled_annotated.SetValue(0)


    def zoomButton(self, event):
        if self.zoom.GetValue() == True:
            # Save pre-zoom xlim and ylim values
            self.prezoom_xlim = self.axes.get_xlim()
            self.prezoom_ylim = self.axes.get_ylim()
            self.toolbar.zoom()
            self.statusbar.SetStatusText("Zoom On")
            self.pan.SetValue(False)
        else:
            self.toolbar.zoom()
            self.statusbar.SetStatusText("Zoom Off")

    def onZoom(self, ax):
        # See if axis limits have actually changed
        curr_xlim = self.axes.get_xlim()
        curr_ylim = self.axes.get_ylim()
        if self.zoom.GetValue() and not (
                self.prezoom_xlim[0] == curr_xlim[0] and self.prezoom_xlim[1] == curr_xlim[1] and self.prezoom_ylim[
            0] == curr_ylim[0] and self.prezoom_ylim[1] == curr_ylim[1]):
            self.updateZoomPan()
            self.statusbar.SetStatusText("Zoom Off")

    def onButtonRelease(self, event):
        if self.pan.GetValue():
            self.updateZoomPan()
            self.statusbar.SetStatusText("Pan Off")

    def lockChecked(self, event):
        self.cb = event.GetEventObject()
        self.view_locked = self.cb.GetValue()

    def onClick(self, event):
        """
        This function adds labels and auto advances to the next label.
        """
        x1 = event.xdata
        y1 = event.ydata

        if event.button == 3:
            if self.rdb.GetSelection() in self.buttonCounter:
                wx.MessageBox('%s is already annotated. \n Select another body part to annotate.' % (
                    str(self.bodyparts[self.rdb.GetSelection()])), 'Error!', wx.OK | wx.ICON_ERROR)
            else:
                color = self.colormap(self.norm(self.colorIndex[self.rdb.GetSelection()]))
                circle = [patches.Circle((x1, y1), radius=self.markerSize, fc=color, alpha=self.alpha)]
                self.num.append(circle)
                self.axes.add_patch(circle[0])
                self.dr = DraggablePoint(circle[0], self.bodyparts[self.rdb.GetSelection()])
                self.dr.connect()
                self.buttonCounter.append(self.rdb.GetSelection())
                self.dr.coords = [[x1, y1, self.bodyparts[self.rdb.GetSelection()], self.rdb.GetSelection()]]
                self.drs.append(self.dr)
                self.updatedCoords.append(self.dr.coords)
                if self.rdb.GetSelection() < len(self.bodyparts) - 1:
                    self.rdb.SetSelection(self.rdb.GetSelection() + 1)
                self.figure.canvas.draw()

        self.canvas.mpl_disconnect(self.onClick)
        self.canvas.mpl_disconnect(self.onButtonRelease)

    def nextLabel(self, event):
        """
        This function is to create a hotkey to skip down on the radio button panel.
        """
        if self.rdb.GetSelection() < len(self.bodyparts) - 1:
            self.rdb.SetSelection(self.rdb.GetSelection() + 1)

    def previousLabel(self, event):
        """
        This function is to create a hotkey to skip up on the radio button panel.
        """
        if self.rdb.GetSelection() < len(self.bodyparts) - 1:
            self.rdb.SetSelection(self.rdb.GetSelection() - 1)


    def browseDir(self, event):



        self.load.Enable(False)
        self.next.Enable(True)
        self.next_labeled.Enable(True)

        self.save.Enable(True)
        self.annotation.Enable(True)
        self.next_labeled_annotated.Enable(True)

        # Enabling the zoom, pan and home buttons
        self.zoom.Enable(True)
        self.home.Enable(True)
        self.pan.Enable(True)
        self.lock.Enable(True)
        self.bodyparts  = self.pref[9:]
        # Reading config file and its variables
        #self.cfg = auxiliaryfunctions.read_config(self.config_file)
        self.project_path = self.address

        imlist = []
        #for imtype in self.imtypes:
            #imlist.extend([fn for fn in glob.glob(os.path.join(self.address, 'Extracted_frames')) if ('labeled.png' not in fn)])
        imlist = os.listdir(self.address + os.sep + self.name)
        if len(imlist) == 0:
            print("No images found!!")

        self.index = np.sort(imlist)
        self.statusbar.SetStatusText('Working on folder: TEMP')
        self.relativeimagenames = ['labeled' + n.split('labeled')[0] for n in
                                   self.index]  # [n.split(self.project_path+'/')[1] for n in self.index]

        # Reading the existing dataset,if already present
        try:
            # self.dataFrame = pd.read_hdf(os.path.join(self.dir, 'CollectedData_' + self.scorer + '.h5'),
            #                              'df_with_missing')
            self.dataFrame = pd.read_pickle(self.filename)
            self.dataFrame.sort_index(inplace=True)
            self.prev.Enable(True)

            # Finds the first empty row in the dataframe and sets the iteration to that index
            for idx, j in enumerate(self.dataFrame.index):
                values = self.dataFrame.loc[j, :].values
                if np.prod(np.isnan(values)) == 1:
                    self.iter = idx
                    break
                else:
                    self.iter = 0

        except:
            try:
                #FOR DEEPLABCUT COMPATIBILITY
                dat = pd.read_csv(self.filename + '.csv', header=None)
                os.rename(os.path.join(self.filename), os.path.join(self.filename + "old"))
                dat.to_pickle(self.filename)
                self.dataFrame = pd.read_pickle(self.filename)
                self.dataFrame.sort_index(inplace=True)
                self.prev.Enable(True)
                self.deep_lab_cut_conversion_dataframe()
                self.dataFrame = self.dataFrame3
                self.dataFrame.to_pickle(self.filename)

                # where to save it, usually as a .pkl

                
            except:
                a = np.empty((len(self.index), 2,))
                a[:] = np.nan
                for bodypart in self.bodyparts:
                    index = pd.MultiIndex.from_product([[self.scorer], [bodypart], ['x', 'y']],
                                                       names=['scorer', 'bodyparts', 'coords'])
                    frame = pd.DataFrame(a, columns=index, index=self.relativeimagenames)
                    self.dataFrame = pd.concat([self.dataFrame, frame], axis=1)
            self.iter = 0

        # Reading the image name
        self.img = self.index[self.iter]
        img_name = Path(self.index[self.iter]).name
        self.norm, self.colorIndex = self.image_panel.getColorIndices(self.address + os.sep + self.name + os.sep + self.img, self.bodyparts)

        # Checking for new frames and adding them to the existing dataframe
        old_imgs = np.sort(list(self.dataFrame.index))
        self.newimages = list(set(self.relativeimagenames) - set(old_imgs))
        if self.newimages == []:
            pass
        else:
            print("Found new frames..")
            # Create an empty dataframe with all the new images and then merge this to the existing dataframe.
            self.df = None
            a = np.empty((len(self.newimages), 2,))
            a[:] = np.nan
            for bodypart in self.bodyparts:
                index = pd.MultiIndex.from_product([[self.scorer], [bodypart], ['x', 'y']],
                                                   names=['scorer', 'bodyparts', 'coords'])
                frame = pd.DataFrame(a, columns=index, index=self.newimages)
                self.df = pd.concat([self.df, frame], axis=1)
            self.dataFrame = pd.concat([self.dataFrame, self.df], axis=0)
            # Sort it by the index values
            self.dataFrame.sort_index(inplace=True)

        # checks for unique bodyparts
        if len(self.bodyparts) != len(set(self.bodyparts)):
            print(
                "Error - bodyparts must have unique labels! Please choose unique bodyparts in config.yaml file and try again. Quitting for now!")
            self.Close(True)

        # Extracting the list of new labels
        oldBodyParts = self.dataFrame.columns.get_level_values(1)
        _, idx = np.unique(oldBodyParts, return_index=True)
        oldbodyparts2plot = list(oldBodyParts[np.sort(idx)])
        self.new_bodyparts = [x for x in self.bodyparts if x not in oldbodyparts2plot]
        # Checking if user added a new label
        if self.new_bodyparts == []:  # i.e. no new label
            self.figure, self.axes, self.canvas, self.toolbar = self.image_panel.drawplot(self.address + os.sep + self.name + os.sep + self.img, img_name, self.iter,
                                                                                          self.index, self.bodyparts,
                                                                                          self.colormap)
            self.axes.callbacks.connect('xlim_changed', self.onZoom)
            self.axes.callbacks.connect('ylim_changed', self.onZoom)

            self.choiceBox, self.rdb, self.slider, self.checkBox = self.choice_panel.addRadioButtons(self.bodyparts,
                                                                                                     self.file,
                                                                                                     self.markerSize)
            self.buttonCounter = MainFrame.plot(self, self.img)
            self.cidClick = self.canvas.mpl_connect('button_press_event', self.onClick)
            self.canvas.mpl_connect('button_release_event', self.onButtonRelease)
        else:
            dlg = wx.MessageDialog(None, "New label found in the config file. Do you want to see all the other labels?",
                                   "New label found", wx.YES_NO | wx.ICON_WARNING)
            result = dlg.ShowModal()
            if result == wx.ID_NO:
                self.bodyparts = self.new_bodyparts
                self.norm, self.colorIndex = self.image_panel.getColorIndices(self.img, self.bodyparts)
            a = np.empty((len(self.index), 2,))
            a[:] = np.nan
            for bodypart in self.new_bodyparts:
                index = pd.MultiIndex.from_product([[self.scorer], [bodypart], ['x', 'y']],
                                                   names=['scorer', 'bodyparts', 'coords'])
                frame = pd.DataFrame(a, columns=index, index=self.relativeimagenames)
                self.dataFrame = pd.concat([self.dataFrame, frame], axis=1)

            self.figure, self.axes, self.canvas, self.toolbar = self.image_panel.drawplot(self.address + os.sep + self.name + os.sep + self.img, img_name, self.iter,
                                                                                          self.index, self.bodyparts,
                                                                                          self.colormap)
            self.axes.callbacks.connect('xlim_changed', self.onZoom)
            self.axes.callbacks.connect('ylim_changed', self.onZoom)

            self.choiceBox, self.rdb, self.slider, self.checkBox = self.choice_panel.addRadioButtons(self.bodyparts,
                                                                                                     self.file,
                                                                                                     self.markerSize)
            self.cidClick = self.canvas.mpl_connect('button_press_event', self.onClick)
            self.canvas.mpl_connect('button_release_event', self.onButtonRelease)
            self.buttonCounter = MainFrame.plot(self, self.img)

        self.checkBox.Bind(wx.EVT_CHECKBOX, self.activateSlider)
        self.slider.Bind(wx.EVT_SLIDER, self.OnSliderScroll)


    def nextImage(self, event):
        """
        Moves to next image
        """
        #  Checks for the last image and disables the Next button
        if len(self.index) - self.iter == 1:
            self.next.Enable(False)
            self.next_labeled.Enable(False)
            self.next_labeled_annotated.Enable(False)

            return
        self.prev.Enable(True)

        if self.next_labeled.GetValue():

            if self.counter_frames == len(self.frame_selected_for_annotation):
                self.next.Enable(False)
                self.next_labeled.Enable(False)
                return
        elif self.next_labeled_annotated.GetValue():

            if self.counter_frames == len(self.frame_selected_for_annotation_auto):
                self.next.Enable(False)
                self.next_labeled.Enable(False)
                self.next_labeled_annotated.Enable(False)
                return
        # Checks if zoom/pan button is ON
        MainFrame.updateZoomPan(self)

        self.statusbar.SetStatusText('Working on folder: {}')
        self.rdb.SetSelection(0)
        self.file = 1
        # Refreshing the button counter
        self.buttonCounter = []
        MainFrame.saveEachImage(self)
        if  self.next_labeled.GetValue():
            self.iter = self.frame_selected_for_annotation[self.counter_frames]
            self.counter_frames += 1
        elif self.next_labeled_annotated.GetValue():
            self.iter = self.frame_selected_for_annotation_auto[self.counter_frames]
            self.counter_frames += 1
        else:
            self.iter = self.iter + 1

        if len(self.index) >= self.iter:
            #self.updatedCoords = MainFrame.getLabels(self, self.iter)
            self.img = self.index[self.iter]
            img_name = 'temp' + str(self.iter)
            #self.figure.delaxes(self.figure.axes[1])  # Removes the axes corresponding to the colorbar
            self.figure, self.axes, self.canvas, self.toolbar = self.image_panel.drawplot(self.address + os.sep + self.name + os.sep+ self.img, img_name, self.iter,
                                                                                          self.index, self.bodyparts,
                                                                                          self.colormap,
                                                                                          keep_view=self.view_locked)
            self.axes.callbacks.connect('xlim_changed', self.onZoom)
            self.axes.callbacks.connect('ylim_changed', self.onZoom)

            self.buttonCounter = MainFrame.plot(self, self.img)
            self.cidClick = self.canvas.mpl_connect('button_press_event', self.onClick)
            self.canvas.mpl_connect('button_release_event', self.onButtonRelease)

    def find(self, s, ch):
        return [i for i, ltr in enumerate(s) if ltr == ch]
    def prevImage(self, event):
        """
        Checks the previous Image and enables user to move the annotations.
        """
        # Checks for the first image and disables the Previous button
        if self.iter == 0:
            self.prev.Enable(False)
            return
        else:
            self.next.Enable(True)
            self.next_labeled.Enable(True)
        # Checks if zoom/pan button is ON
        MainFrame.updateZoomPan(self)
        MainFrame.saveEachImage(self)

        self.buttonCounter = []
        if  self.next_labeled.GetValue():
            if self.counter_frames == 0:
                self.prev.Enable(False)
                return
            self.counter_frames -= 1
            self.iter = self.frame_selected_for_annotation[self.counter_frames]
        if self.next_labeled_annotated.GetValue():
            if self.counter_frames == 0:
                self.prev.Enable(False)
                return
            self.counter_frames -= 1
            self.iter = self.frame_selected_for_annotation_auto[self.counter_frames]
        else:
            self.iter = self.iter - 1

        self.rdb.SetSelection(0)
        self.img = self.index[self.iter]
        img_name = Path(self.index[self.iter]).name
        self.figure, self.axes, self.canvas, self.toolbar = self.image_panel.drawplot(self.address + os.sep + self.name + os.sep+ self.img, img_name, self.iter,
                                                                                      self.index, self.bodyparts,
                                                                                      self.colormap,
                                                                                      keep_view=self.view_locked)
        self.axes.callbacks.connect('xlim_changed', self.onZoom)
        self.axes.callbacks.connect('ylim_changed', self.onZoom)

        self.buttonCounter = MainFrame.plot(self, self.img)
        self.cidClick = self.canvas.mpl_connect('button_press_event', self.onClick)
        self.canvas.mpl_connect('button_release_event', self.onButtonRelease)
        MainFrame.saveEachImage(self)

    def getLabels(self, img_index):
        """
        Returns a list of x and y labels of the corresponding image index
        """
        self.previous_image_points = []
        for bpindex, bp in enumerate(self.bodyparts):
            image_points = [[self.dataFrame[self.scorer][bp]['x'].values[self.iter],
                             self.dataFrame[self.scorer][bp]['y'].values[self.iter], bp, bpindex]]
            self.previous_image_points.append(image_points)
        return (self.previous_image_points)


    def plot(self, img):
        """
        Plots and call auxfun_drag class for moving and removing points.
        """
        self.drs = []
        self.updatedCoords = []

        for bpindex, bp in enumerate(self.bodyparts):
            if self.dataFrame[self.scorer][bp]['x'].values[self.iter]>0:
                color = self.colormap(self.norm(self.colorIndex[bpindex]))
                self.points = [self.dataFrame[self.scorer][bp]['x'].values[self.iter],
                               self.dataFrame[self.scorer][bp]['y'].values[self.iter]]

            elif  self.dataFrame[self.scorer][bp]['y'].values[self.iter] != -1:

                if self.colormap != 'cool':
                    color = plt.get_cmap('cool')(self.norm(self.colorIndex[bpindex]))
                else:
                    color = plt.get_cmap('summer')(self.norm(self.colorIndex[bpindex]))
                self.points = [-self.dataFrame[self.scorer][bp]['x'].values[self.iter],
                               self.dataFrame[self.scorer][bp]['y'].values[self.iter]]


            else:
                continue

            circle = [
                patches.Circle((self.points[0], self.points[1]), radius=self.markerSize, fc=color, alpha=self.alpha)]
            self.axes.add_patch(circle[0])
            self.dr = DraggablePoint(circle[0], self.bodyparts[bpindex])
            self.dr.connect()
            self.dr.coords = MainFrame.getLabels(self, self.iter)[bpindex]
            self.drs.append(self.dr)
            self.updatedCoords.append(self.dr.coords)
            if np.isnan(self.points)[0] == False:
                self.buttonCounter.append(bpindex)





            patchList = []

            for key in range(0, len(self.bodyparts)):

                if self.dataFrame[self.scorer][bp]['x'].values[self.iter] > 0 or np.isnan(self.dataFrame[self.scorer][bp]['x'].values[self.iter]) :
                    color = plt.get_cmap(self.colormap)(self.norm(self.colorIndex[key]))
                else:

                    if self.colormap != 'cool':
                        color = plt.get_cmap('cool')(self.norm(self.colorIndex[key]))
                    else:
                        color = plt.get_cmap('summer')(self.norm(self.colorIndex[key]))


                data_key = Line2D([0], [0], marker='o', color='black', markerfacecolor=color,
                                  label=self.bodyparts[key], markersize=8)
                patchList.append(data_key)

            self.axes.legend(handles=patchList, bbox_to_anchor=(1.1, 1.1))




        self.figure.canvas.draw()

        return (self.buttonCounter)

    def saveEachImage(self):
        """
        Saves data for each image
        """
        for idx, bp in enumerate(self.updatedCoords):
            self.dataFrame.loc[self.relativeimagenames[self.iter]][self.scorer, bp[0][-2], 'x'] = bp[-1][0]
            self.dataFrame.loc[self.relativeimagenames[self.iter]][self.scorer, bp[0][-2], 'y'] = bp[-1][1]

    def saveDataSet(self, event):
        """
        Saves the final dataframe
        """
        self.statusbar.SetStatusText("File saved")
        MainFrame.saveEachImage(self)
        MainFrame.updateZoomPan(self)
        # Windows compatible
        self.dataFrame.sort_index(inplace=True)
        self.dataFrame.to_csv(os.path.join( self.filename + ".csv"))

        self.dataFrame.to_pickle(self.filename)  # where to save it, usually as a .pkl

        # if not os.path.isdir(os.path.join(self.address,self.name + 'annotation')):
        #     os.mkdir(os.path.join(self.address,self.name + 'annotation'))

        annotated = np.where(np.isnan(self.dataFrame.iloc[:, 0].values) == False)[0]

        print(len(annotated))
        # for j in range(0, len(annotated)):
        #
        #     imm = cv2.imread(os.path.join(self.address, self.name, self.relativeimagenames[annotated[j]][7:]))
        #     points = self.dataFrame.iloc[annotated[j]]
        #     for i in range(0, len(self.bodyparts)):
        #         ann = np.zeros_like(imm)
        #         if not np.isnan(points[i * 2] and points[i * 2 + 1]):
        #             cv2.circle(ann, (int(round((points[i * 2] * (2 ** 4)))), int(round(points[i * 2 + 1] * (2 ** 4)))),
        #                        int(round(self.markerSize * (2 ** 4))), (255, 255, 255), thickness=-1, shift=4)
        #         cv2.imwrite(os.path.join(self.address, self.name + 'annotation', ("{:02d}".format(i))+self.relativeimagenames[annotated[j]]),
        #                 ann)





    def onChecked(self, event):
        self.cb = event.GetEventObject()
        if self.cb.GetValue() == True:
            self.slider.Enable(True)
            self.cidClick = self.canvas.mpl_connect('button_press_event', self.onClick)
            self.canvas.mpl_connect('button_release_event', self.onButtonRelease)
        else:
            self.slider.Enable(False)

    def updateZoomPan(self):
        # Checks if zoom/pan button is ON
        if self.pan.GetValue() == True:
            self.toolbar.pan()
            self.pan.SetValue(False)
        if self.zoom.GetValue() == True:
            self.toolbar.zoom()
            self.zoom.SetValue(False)


def show(label_frames, video_list_with_address,index_video,config,index, imtypes=['*.png']):
    app = wx.App()
    frame = MainFrame(None, video_list_with_address,index_video,label_frames,index, config, imtypes).Show()
    app.MainLoop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    cli_args = parser.parse_args()

