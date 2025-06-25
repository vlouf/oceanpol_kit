import os
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

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "requirements.txt"), "r") as fid:
    REQUIREMENTS = [line for line in fid.read().splitlines() if not line.startswith("#")]

with open(os.path.join(here, "README.md")) as readme_file:
    README = readme_file.read()


setup(
    name='oceanpol_kit',
    version='0.1.0',
    packages=find_packages(),
    ext_modules=cythonize(extensions, language_level="3", build_dir="build"),
    install_requires=REQUIREMENTS,
    package_dir={"oceanpol_kit": "oceanpol_kit"},
    author='Valentin Louf',
    author_email='valentin.louf@bom.gov.au',
    description='A processing suite for OceanPOL radar data',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/vlouf/oceanpol_kit',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)
