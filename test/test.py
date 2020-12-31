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

# initialize the test object

test = test()

# create a new project

test.new_project_routine()

# or open an existing one  (choose one or the other)

test.open_project()

# add videos to the project, or load the ones already added

test.Load_Videos()

# set preferences for annotation

Frame_selection(None, os.path.dirname(test.config_file_text) + os.sep + 'annotation_options.txt')

# if you are using the sample_video uploaded in the repository, use the next line of code to download existing annotation for testing
test.load_testing_annotation(test.config_file_text)

# needed for proceeding with manual or automatic annotation, code to handle the rest of operations

test.annotate(0,test.config_file_text)

# set the preferences for the neural network prediction

test.preferences_annotation()

# perform training
test.check_and_train()

# view results
test.view_annotation()

