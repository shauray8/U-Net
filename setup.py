import sys
from setuptools import setup, find_packages

sys.path[0:0] = ["U-Net"]
#from version import __version__

setup(
  name = 'U-Net',
  packages = find_packages(),
  #version = __version__,
  license='MIT',
  description = 'U Net and other bio stuff',
  author = 'Shauray Singh',
  author_email = 'shauray9@gmail.com',
  url = 'https://github.com/shauray8/U-Net',
  keywords = ['generative adversarial networks',"segmentation","covid-19", 'machine learning'],
  install_requires=[
      'numpy',
      'tqdm',
      'torch',
      'torchvision',
      'pillow',
      "matplotlib",
      "tensorboard"
  ],
)
