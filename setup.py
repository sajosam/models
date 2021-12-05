# mACHINE LEARNING SETUP FILE
from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '1.0.2.2'
DESCRIPTION = 'Machine Learning models make simple'
LONG_DESCRIPTION = '''
                                        Machine Learning models
                                    (single-file, easy-to-use, and modular)'''

# Setting up
setup(
    name="ml-pkg-models",
    version=VERSION,
    author="sajo sam",
    author_email="<sajosamambalakara@gmail.com>",
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

