# coding: utf8
"""
Setup script for peaknet2020
"""

from glob import glob
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


def main():
    setup(name='peaknet2020',
          version='0.1.0',
          author='Po-Nan Li',
          author_email="liponan@stanford.edu",
          description="neural network for peak detection",
          packages=["peaknet", "unet"],
          package_dir={"peaknet": "peaknet", "unet:": "unet"},
          install_requires=["numpy>=1.14", "torch>=1.0", "h5py>=2.8.0"],
          scripts=[s for s in glob('scripts/*') if not s.endswith('__.py')]
          )


if __name__ == "__main__":
    main()