import wx
from VisionTool.create_new_project  import *
import os
import random
from VisionTool.features_extractor import *
from VisionTool.pose_estimation import *


class Notebook(wx.Notebook):

    def __init__(self,parent,address,size):
        self.address = address
        wx.Notebook.__init__(self,parent=parent,size = size)
        self.page_features = features_Extractor(self,size,self.address)
        self.page_pose = pose_estimation(self,self.address)
        self.AddPage(self.page_features, "     Features_Extraction")
        self.AddPage(self.page_pose, "Pose_Estimation")
        # tabThree = action_classification(self, size)
        # notebook.AddPage(tabThree, "Action classification")

    def update_address(self,address):
        self.address = address
        self.page_pose.update_address(address)
        self.page_features.update_address(address)



class Frame_main(wx.Frame):

    def __init__(self, parent):
        size = (700, 700)
        wx.Frame.__init__(self,None,title="VisionTool", pos = (500,50), size = size)

        self.address = ""
        self.panel = wx.Panel(self)

        self.ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        self.CONFIG_PATH = os.path.dirname(os.path.join(self.ROOT_DIR, 'configuration.conf') ) # requires `import os`

        logo = os.path.join(self.CONFIG_PATH, 'malga2.png')

        self.ico=wx.Icon(wx.Bitmap(logo))#path to icon
        self.SetIcon(self.ico)

        self.notebook = Notebook(self, self.address,size=size)
        menubar = wx.MenuBar()
        fileMenu = wx.Menu()

        new_project = wx.MenuItem(fileMenu, wx.LANGUAGE_DEFAULT+1, '&New Project\tCtrl+N')

        self.Bind(wx.EVT_MENU, self.new_project_routine, new_project)

        fileMenu.Append(new_project)
        open_project = wx.MenuItem(fileMenu, wx.ID_ANY, '&Open Existing Project\tCtrl+O')

        fileMenu.Append(open_project)
        self.Bind(wx.EVT_MENU, self.open_project, open_project)

        # fileMenu.Append(wx.ID_SAVE, '&Save')
        fileMenu.AppendSeparator()

        imp = wx.Menu()
        imp.Append(wx.ID_ANY, 'Import new architecture')
        fileMenu.Append(wx.ID_ANY, 'I&mport', imp)

        qmi = wx.MenuItem(fileMenu, wx.ID_EXIT, '&Quit\tCtrl+W')
        fileMenu.Append(qmi)

        self.Bind(wx.EVT_MENU, self.OnQuit, qmi)
        menubar.Append(fileMenu, '&File')


        for p in range(self.notebook.GetPageCount()):
            name = self.notebook.GetPageText(p)

            fileMenu.Append(p+1,"%s\tCtrl+%d" % (name,p),"Go to the %s page" % (name))
            self.Bind(wx.EVT_MENU,self.GoToPage,id=p+1)


        fileMenu.AppendSeparator()


        m = self.notebook.GetCurrentPage().Menu
        n = self.notebook.GetCurrentPage().MenuName

        menubar.Append(m, n)

        self.SetMenuBar(menubar)
        sizer = wx.BoxSizer()
        sizer.Add(self.notebook,1,wx.EXPAND)
        self.SetSizerAndFit(sizer)
        self.Layout()

        ### The EVT_NOTEBOOX_PAGE_CHANGED is registered
        ###  at the frame level
        self.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.AdjustMenus)
        self.Bind(wx.EVT_MENU, self.OnMenu)
        ### This disables the currently selected menu item
        menubar.Enable(self.notebook.GetSelection()+1, False)

    def GoToPage(self, evt):
        self.notebook.ChangeSelection(evt.GetId())
        self.AdjustMenus(evt)

    def AdjustMenus(self, evt):
        ### Menubar position 1 is the swapping menu
        m = self.notebook.GetCurrentPage().Menu
        n = self.notebook.GetCurrentPage().MenuName
        mbar = self.GetMenuBar()
        mbar.Replace(1, m, n)
        ### Disable current page on tab menu
        ### because the tabmenu item id's match the index
        ### of the notebook page index, this works
        for page in range(self.notebook.GetPageCount()):
            if page == self.notebook.GetSelection()+1:
                ### The Menubar can enable any menu item by ID alone
                mbar.Enable(page, False)
            else:
                mbar.Enable(page, True)
        ### If you don't enable this, trying to
        ###  Change a notebook panel to the current panel
        ###  raises an error

    def OnMenu(self, evt):
        ### While not used in this demo, if the other menus
        ### actually did anything, this method would catch them
        evt.Skip()

    def OnQuit(self, e):
        self.Close()

    def new_project_routine(self, e):

        initiate = routine()
        self.address = initiate.new_project()
        self.notebook.update_address(self.address)
        pass

    def open_project(self, e):

        initiate = routine()
        self.address = initiate.open_existing_project()
        self.notebook.update_address(self.address)
        pass

def main():

    app = wx.App()
    ex = Frame_main(None)
    ex.Show()
    app.MainLoop()

#
if __name__ == '__main__':
     main()
