# VisionTool
VisionTool: a toolbox for semantic features extraction 
## Introduction
VisionTool is a python toolbox for pose estimation on videos containing actions. It is powered with a simple and intuititive Graphical User Interface (GUI) to allow a complete analysis from annotation to joint position estimation.

![Picture1](https://user-images.githubusercontent.com/51142446/103312491-87ee9900-4a1d-11eb-9781-595327da4d0a.png)

*Figure 1. VisionTool main GUI*

## VisionTool GUI
VisionTool’s main GUI shows two different interfaces. The first one “Features_Extraction” can be used to perform annotation, training and testing on an imported set of videos. The second one “Pose Estimation” includes the implementation of neural networks architectures to perform action recognition. In its first release, only features extraction is available. Future releases will be focus on the implementation of an actual action recognition submodule including the latest architectures used in the scientific community for the solution of recognition problems.

## Preliminary Operations 

A description of the preliminary operations necessary to create a new project, import a set of videos and set the analysis parameters follows in the next paragraph. 

### VisionTool Menu 

VisionTool menu consists of two different submenus. The File Menu allows to perform a set of preliminary operations (see Figure 2). A set of keyboard shortcuts is implemented and reported in the menu for faster selections. 

### Create a new project

To create a new project, select the submenu File and click on the menu tab New Project, or press Ctrl+N. A browse directory win
ow will pop-up asking where do you want to save the new project. 

![Picture2](https://user-images.githubusercontent.com/51142446/103312970-ecf6be80-4a1e-11eb-8cf8-a93491a66165.png)

*Figure 2. VisionTool’s File menu*

### Open an existing project

