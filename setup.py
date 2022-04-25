import setuptools


setuptools.setup(
    name="VisionTool",
    version="0.0.1",
    author="Vito Paolo Pastore, Matteo Moro, Francesca Odone",
    author_email="vitopaolopastore@gmail.com",
    description="Toolbox for semantic features extraction ",
    long_description="Toolbox for semantic features extraction ",
    url="https://github.com/Malga-Vision/VisionTool.git",
    packages=setuptools.find_packages(),
    install_requires = ['absl-py==0.10',
'astor==0.8.1',
'astunparse==1.6.3',
'cachetools==4.1.0',
'chardet==3.0.4',
'cycler==0.10.0',
'decorator==4.4.2',
'efficientnet==1.0.0',
'gast==0.3.3',
'google-auth==1.16.1',
'google-auth-oauthlib==0.4.1',
'google-pasta==0.2.0',
'grpcio==1.32.0',
'h5py==2.10.0',
'scikit-video==1.1.11',
'idna==2.9',
'image-classifiers==1.0.0',
'importlib-metadata==1.6.1',
'joblib==0.16.0',
'Keras-Applications==1.0.8',
'Keras-Preprocessing==1.1.2',
'kiwisolver==1.2.0',
'Markdown==3.2.2',
'matplotlib==3.3.0',
'mock==4.0.2',
'networkx==2.4',
'numpy==1.19.2',
'oauthlib==3.1.0',
'opencv-python==4.4.0.46',
'opt-einsum==3.3.0',
'pandas==1.0.4',
'Pillow==7.1.2',
'protobuf==3.12.2',
'pyasn1==0.4.8',
'pyasn1-modules==0.2.8',
'pyparsing==2.4.7',
'python-dateutil==2.8.1',
'pytz==2020.1',
'PyWavelets==1.1.1',
'PyYAML==5.3.1',
'requests==2.23.0',
'requests-oauthlib==1.3.0',
'rsa==4.0',
'scikit-image==0.17.2',
'scikit-learn==0.23.1',
'scipy==1.4.1',
'segmentation-models>=1.0.1',
'six==1.15.0',
'sklearn==0.0',
'tensorboard==2.2.0',
'tensorflow>=2.2.0',
'termcolor==1.1.0',
'threadpoolctl==2.1.0',
'tifffile==2020.6.3',
'tqdm==4.46.1',
'urllib3==1.25.9',
'wxPython==4.1.0',
'Werkzeug==1.0.1',
'wincertstore==0.2',
'wrapt==1.12.1',
'zipp==3.1.0'        ],
extra_requires= ['tensorflow-gpu>=2.2.0']
    
)
