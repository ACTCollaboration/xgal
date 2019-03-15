from distutils.core import setup, Extension
import os



setup(name='xgal',
      version='0.1',
      description='Cosmology Analysis',
      license='BSD-2-Clause',
      packages=['xgal'],
      package_dir={'xgal':'xgal'},
      zip_safe=False)
