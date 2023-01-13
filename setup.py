import setuptools
from numpy.distutils.core import setup, Extension

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md')) as f:
    long_description = f.read()

exts = []
exts.append(Extension(name='cactus.src.matrix', sources=['cactus/src/matrix.f90']))
exts.append(Extension(name='cactus.src.pixel_util', sources=['cactus/src/pixel_util.f90']))
exts.append(Extension(name='cactus.src.pixel_1dto2d', sources=['cactus/src/pixel_1dto2d.f90']))
exts.append(Extension(name='cactus.src.pixel_1dto3d', sources=['cactus/src/pixel_1dto3d.f90']))
exts.append(Extension(name='cactus.src.union_finder', sources=['cactus/src/union_finder.f90']))

setup(name = 'cactus',
      version = '0.0.0',
      description       = "Cosmic web Classification Toolkit",
      long_description  = long_description,
      long_description_content_type = 'text/markdown',
      url               = 'https://github.com/knaidoo29/cactus',
      author            = "Krishna Naidoo",
      author_email      = "krishna.naidoo.11@ucl.ac.uk",
      license='MIT',
      packages=setuptools.find_packages(),
      install_requires=['numpy', 'matplotlib', 'healpy'],
      ext_modules = exts,
      python_requires = '>=3',
      classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Mathematics',
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      )
