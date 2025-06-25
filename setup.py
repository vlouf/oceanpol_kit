from setuptools import setup, find_packages
from setuptools import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "oceanpol_kit.echo_steiner", 
        ["oceanpol_kit/echo_steiner.pyx"], 
        include_dirs=[numpy.get_include()],        
    )
]

setup(
    name='oceanpol_kit',
    version='0.1.0',
    packages=find_packages(),
    ext_modules=cythonize(extensions, language_level="3", build_dir="build"),
    install_requires=[
        'pyodim',
        "arm_pyart",
        'unravel',
        'pandas',
        'numpy',
        'scipy',
        'cftime',
        'xarray',        
        'phido',
        'numba',
        'csu_radartools',
        "Cython"
    ],
    package_dir={"oceanpol_kit": "oceanpol_kit"},
    author='Valentin Louf',
    author_email='valentin.louf@bom.gov.au',
    description='A processing suite for OceanPOL radar data',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/vlouf/oceanpol_kit',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)

