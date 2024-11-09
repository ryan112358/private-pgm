from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='private-pgm',
    version='0.0.2',
    description='Graphical-model based estimation and inference for differential privacy.',
    url='git@github.com:ryan112358/private-pgm.git',
    author='Ryan McKenna',
    author_email='rmckenna21@gmail.com',
    license='Apache License 2.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=requirements,
)
