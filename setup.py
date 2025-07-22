from setuptools import setup, find_packages
__version__='0.0.1'
setup(
    name='dart',
    author= 'Seyyed Reza Mashhadi',
    packages=find_packages('src'),
    package_dir={'':'src'},
    install_requires=['numpy', 'matplotlib', 'scipy', 'pandas'] ,#
    version=__version__,
    license='MIT?',
    description='borehole nmr utilities',
    python_requires=">=3.8",
)
