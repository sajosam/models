# mACHINE LEARNING SETUP FILE
from setuptools import setup, find_packages
import codecs
import os

VERSION = '0.0.1'
DESCRIPTION = 'Machine Learning models'
LONG_DESCRIPTION = 'A package that allows to build machine learning models for the purpose of prediction of the future. ' 

# Setting up
setup(
    name="ml-ss-models",
    version=VERSION,
    author="sajo sam",
    author_email="<saajosaam@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['pandas'],
    keywords=['python', 'classification','regression','machine learning models','hyper parameter turning'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)

