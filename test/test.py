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

# test.open_project()

# or open an existing one  (choose one or the other, you can comment one of the two lines of code)

test.Load_Videos()

# adds sample video included within the repository to the project, loads the sample annotations and labels,
# performs frame extraction and reads information about the video (you have to call it before 'view_annotation' or 'check_and_train' modules)

test.preferences_annotation()
# set the preferences for the neural network prediction, the suggested parameters for sample video analysis are 
#LinkNet with backbone an EfficientNet architecture and a minimum batch size equal to 5.


# perform training
test.check_and_train()

# use 
#test.check_and_test() to perform testing if a neural network has already been trained.

test.view_annotation()

# view results of prediction!
