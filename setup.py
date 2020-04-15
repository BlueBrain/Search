from setuptools import setup, find_packages

VERSION = '0.1'

description = "Blue Brain Search"

install_requires = [
    'numpy',
    'pandas',
    'scikit-learn',
    'scipy',
    'torch',
]
setup_requires = ['pytest-runner']
tests_require = [
    'flake8',
    'pytest',
    'pytest-cov',
]

setup(
    name="BBSearch",
    description=description,
    author='Blue Brain Project',
    version=VERSION,
    package_dir={'': 'src'},
    packages=find_packages("./src"),
    python_requires='>=3.6',
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
)
