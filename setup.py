"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

from setuptools import setup, find_packages, Extension
import sys

if sys.version_info.major != 3:
    raise RuntimeError('rrl requires Python 3')

setup(
    name='rrl',
    description='An implementation of RRL (Rule-based Representation Learner). See paper `Scalable Rule-Based Representation Learning for Interpretable Classiﬁcation` for details',
    long_description=open('README.md').read(),
    version='1.0.4',
    author='Yu Huang',
    author_email='yuhuang-cst@foxmail.com',
    packages=find_packages(),
    zip_safe=False,
    url='https://github.com/yuhuang-cst/rrl',
    license='LICENSE',
    install_requires=open('requirements.txt').read(),
)

