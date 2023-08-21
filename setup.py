from setuptools import Extension, setup
from Cython.Build import cythonize

import numpy

extensions = [
    Extension(name="*",
              sources=["*.pyx"],
              include_dirs=[numpy.get_include()],
              extra_compile_args=["-std=c++11"],
             )
]

setup(
    ext_modules=cythonize(extensions),
    zip_safe=False,
)