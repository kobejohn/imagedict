#from distutils.core import setup
from setuptools import setup, find_packages #seems required to make install_requires work

setup(
    name = 'imagedict',
    version = '0.1.0',
    url = 'http://github.com/kobejohn/imagedict',
    py_modules = ['imagedict', '_opencv_fallback'],

    #todo: this is probably totally wrong but first way I found to get cv2.pyd included
    packages = find_packages('.'),
    package_data = {
        '': ['_opencv_fallback/cv2.pyd']
        },
    include_package_data = True,

    install_requires = ['numpy'],
    author_email = 'niericentral@gmail.com',
    description = 'ImageDict emulates the basic behavior of a dict but with images as keys.',
    license = 'MIT',
    author = 'KobeJohn',
    classifiers = [
        'Development Status :: 4 - Beta',
        'Environment :: Other Environment',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2'
    ]
)