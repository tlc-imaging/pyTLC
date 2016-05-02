from setuptools import setup, find_packages

from pyImagingMSpec import __version__

setup(name='pyTLC',
      version=__version__,
      description='Python library for processing thin layer chromatography imaging datasets',
      url='https://github.com/alexandrovteam/pyTLC',
      author='Alexandrov Team, EMBL',
      packages=find_packages())
