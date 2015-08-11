from setuptools import setup

setup(
    name='pyklip',
    version='1.1',
    description='pyKLIP: PSF Subtraction for Exoplanets and Disks',
    url='https://bitbucket.org/pyKLIP/pyklip',
    author='pyKLIP Developers',
    author_email='jwang@astro.berkeley.edu',
    license='BSD',
    packages=['pyklip'],
    zip_safe=False,
    classifiers=[
        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: BSD License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        ],
    keywords='KLIP PSF Subtraction',
    install_requires=['numpy', 'scipy', 'astropy']
    )
