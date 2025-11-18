from setuptools import setup, find_packages

setup(
    name='mangrove-carbon-pipeline',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A pipeline for estimating carbon in mangroves using satellite images and deep learning segmentation.',
    url='https://github.com/ayoqill/mangrove-carbon-estimation',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'numpy>=1.24.0',
        'rasterio>=1.3.0',
        'geopandas>=0.12.0',
        'matplotlib>=3.7.0',
        'scikit-learn>=1.2.0',
        'pandas>=2.0.0',
        'pyyaml>=6.0',
        'opencv-python>=4.7.0',
        'opencv-contrib-python>=4.7.0',
        'segmentation-models-pytorch>=0.3.0',
        'transformers>=4.30.0',  # For SAM-2 model loading
        'pillow>=9.0.0',  # Image handling
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'jupyter>=1.0.0',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    keywords='mangrove carbon satellite remote-sensing deep-learning segmentation SAM-2',
)