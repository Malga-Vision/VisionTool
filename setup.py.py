import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="VisionTool",
    version="0.0.1",
    author="Vito Paolo Pastore, Matteo Moro, Francesca Odone",
    author_email="vitopaolopastore@gmail.com",
    description="Toolbox for semantic features extraction ",
    long_description="Toolbox for semantic features extraction ",
    url="https://github.com/Malga-Vision/VisionTool",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License ::GNU License",
        "Operating System :: OS Independent",
    ],
)