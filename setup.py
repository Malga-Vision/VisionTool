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
    install_reqs = parse_requirements('requirements.txt', session='hack')
                    
    classifiers=[
        "Programming Language :: Python :: 3",
        "License ::GNU License",
        "Operating System :: OS Independent",
    ],
)
