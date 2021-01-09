"""
    Source file name: test.py

    Description: this file contains the code to perform tests operations

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

from VisionTool.Test_lib import test

test = test()

'''initialize the test object from the source file 'Test_lib'. The only purpose of 'Test_lib' is to provide a fast example of working flow for VisionTool out of the 
regular operation which would be performed totally using the provided GUI (in source file 'main.py'). 
Besides the tests, for regular usage, please use the module 'main' as described in the tutorial and file readme. '''


test.new_project_routine()

# create a new project

test.open_project()

# or open an existing one  (choose one or the other, you can comment one of the two lines of code)

test.Load_Videos()

# add videos to the project, or load the ones already added, for test, you can use the file 'sample_video.avi' downloaded with the repository 

#test.Frame_selection()

# set preferences for annotation (not necessary if you use the line of code load_testing_annotation, uncomment if you want to make your annotation again). 


test.load_testing_annotation(test.config_file_text)

# if you are using the sample_video uploaded in the repository, use the next line of code to download existing annotation for training and testing

test.preferences_annotation()
# set the preferences for the neural network prediction

test.annotate(0, test.config_file_text)

# needed for proceeding with manual or automatic annotation, code to handle the rest of operations.
# you can view the annotation, and more important, this line of code extracts the frames. 

# perform training
test.check_and_train()

# use 
#test.check_and_test() to perform testing if a neural network has already been trained.

test.view_annotation()

# view results of prediction!
