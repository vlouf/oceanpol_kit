# setup.py – only for extensions (pyproject.toml handles the rest)
from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize(
        Extension(
            "oceanpol_kit.echo_steiner",
            ["oceanpol_kit/echo_steiner.pyx"],
            include_dirs=[np.get_include()],
        ),
        language_level="3str",
    )
)