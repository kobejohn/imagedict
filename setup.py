from distutils.core import setup

setup(
    name = 'imagedict',
    version = '0.1.0',
    url = 'http://github.com/kobejohn/imagedict',
    py_modules = ['imagedict'],
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