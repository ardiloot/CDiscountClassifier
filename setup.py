import numpy
from distutils.core import setup
from Cython.Build import cythonize

include_dirs  = [numpy.get_include()]
setup(
    name = "CDiscountClassifier",
    ext_modules = cythonize("CDiscountClassifier/_HelperFunctions.pyx", include_dirs = include_dirs),
    include_dirs=[numpy.get_include()],
    packages = ["CDiscountClassifier"],
)
