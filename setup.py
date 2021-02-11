"""
    Source file name: setup.py
    
    Description: empty file for installation requirements
    
    Copyright (C) <2020>  <Vito Paolo Pastore, Simone Bianco>
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
setuptools.setup(
    name="Filo_Analyzer",
    version="0.0.1",
    author="Vito Paolo Pastore, Simone Bianco",
    author_email="vitopaolopastore@gmail.com",
    description="Toolbox for image processing- and deep-learning - based cell filopodia detection and analysis",
    long_description="Toolbox for image processing- and deep-learning - based cell filopodia detection and analysis",
    url="https://github.com/VitoPaoloPastore/Filo_Analyzer",
    packages=setuptools.find_packages(),
    install_requires = ['absl-py==0.10','Keras-Applications==1.0.8',
'Keras-Preprocessing==1.1.2','opencv-python==4.4.0.46','pandas==1.0.4',
'Pillow==7.1.2','scikit-image==0.17.2',
'scikit-learn==0.23.1',
'scipy==1.4.1',
'segmentation-models==1.0.1','sklearn==0.0',
'tensorboard==2.2.0',
'tensorflow==2.2.0'],
extra_requires= ['tensorflow-gpu==2.2.0']
)