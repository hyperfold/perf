from setuptools import setup, find_packages

setup(
    name='perf',
    version='dev'
    author='hyperfold',
    packages=find_packages(include=['perf']),
    install_requires=[
        'torch==1.13.1+cu116'
    ]
)
