from setuptools import setup, find_packages

setup(
    name='pennypulse',
    version='1.2.0',
    description='Custom modifications to Pennylane functions',
    url='https://github.com/nacedob/Pennypulse.git',
    author='Nacho Acedo',
    license='BSD 2-clause',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'PennyLane==0.38.0',
        'PennyLane_Lightning==0.38.0',
    ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
)
