import setuptools
from numpy.distutils.core import setup, Extension

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md')) as f:
    long_description = f.read()

exts = []
exts.append(Extension(name='cactus.src.fortran_src.matrix', sources=['cactus/src/fortran_src/matrix.f90']))
exts.append(Extension(name='cactus.src.fortran_src.pixel_util', sources=['cactus/src/fortran_src/pixel_util.f90']))
exts.append(Extension(name='cactus.src.fortran_src.pixel_1dto2d', sources=['cactus/src/fortran_src/pixel_1dto2d.f90']))
exts.append(Extension(name='cactus.src.fortran_src.pixel_1dto3d', sources=['cactus/src/fortran_src/pixel_1dto3d.f90']))
exts.append(Extension(name='cactus.src.fortran_src.union_finder', sources=['cactus/src/fortran_src/union_finder.f90']))

exts.append(Extension(name='cactus.ext.fiesta.src.matrix', sources=['cactus/ext/fiesta/src/matrix.f90']))
exts.append(Extension(name='cactus.ext.fiesta.src.polygon', sources=['cactus/ext/fiesta/src/polygon.f90']))
exts.append(Extension(name='cactus.ext.fiesta.src.delaunay2d', sources=['cactus/ext/fiesta/src/delaunay2d.f90']))
exts.append(Extension(name='cactus.ext.fiesta.src.polyhedron', sources=['cactus/ext/fiesta/src/polyhedron.f90']))
exts.append(Extension(name='cactus.ext.fiesta.src.delaunay3d', sources=['cactus/ext/fiesta/src/delaunay3d.f90']))
exts.append(Extension(name='cactus.ext.fiesta.src.differentiate', sources=['cactus/ext/fiesta/src/differentiate.f90']))
exts.append(Extension(name='cactus.ext.fiesta.src.grid', sources=['cactus/ext/fiesta/src/grid.f90']))
exts.append(Extension(name='cactus.ext.fiesta.src.part2grid_wei', sources=['cactus/ext/fiesta/src/part2grid_wei.f90']))
exts.append(Extension(name='cactus.ext.fiesta.src.part2grid_pix', sources=['cactus/ext/fiesta/src/part2grid_pix.f90']))
exts.append(Extension(name='cactus.ext.fiesta.src.part2grid_2d', sources=['cactus/ext/fiesta/src/part2grid_2d.f90']))
exts.append(Extension(name='cactus.ext.fiesta.src.part2grid_3d', sources=['cactus/ext/fiesta/src/part2grid_3d.f90']))

exts.append(Extension(name='cactus.ext.magpie.src.pixel_utils', sources=['cactus/ext/magpie/src/pixel_utils.f90']))
exts.append(Extension(name='cactus.ext.magpie.src.pixel_1dto2d', sources=['cactus/ext/magpie/src/pixel_1dto2d.f90']))
exts.append(Extension(name='cactus.ext.magpie.src.pixel_1dto3d', sources=['cactus/ext/magpie/src/pixel_1dto3d.f90']))
exts.append(Extension(name='cactus.ext.magpie.src.pixel_binbyindex', sources=['cactus/ext/magpie/src/pixel_binbyindex.f90']))

setup(name = 'cactus',
      version = "0.3.0",
      description       = "Cosmic web Classification Toolkit",
      long_description  = long_description,
      long_description_content_type = 'text/markdown',
      url               = 'https://github.com/knaidoo29/cactus',
      author            = "Krishna Naidoo",
      author_email      = "krishna.naidoo.11@ucl.ac.uk",
      license='MIT',
      packages=setuptools.find_packages(),
      install_requires=['numpy', 'scipy'],
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
