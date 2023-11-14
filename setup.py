from setuptools import find_packages, setup
from codecs import open
from os import path

#twine upload dist/*

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='edvinspythonlib',
    packages=find_packages(include=['edvinspythonlib']),
    version='0.1.2',
    description='My first Python library',
    author='Me',
    install_requires=['numpy','pandas','matplotlib'],
    tests_require=['pytest == 4.4.1'],
    setup_requires=['pytest-runner == 4.4'],
    test_suite='tests',
    long_description=long_description,
    long_description_content_type="text/markdown"
)