from setuptools import setup, find_packages
setup(
    name='vamsl-lib',
    version='1.0',
    description='VaMSL: Variational Mixture Structure Learning',
    author='',
    author_email='',
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    install_requires=[
        'jax>=0.3.17',
        'jaxlib>=0.3.14',
        'numpy',
        'igraph',
        'imageio',
        'jupyter',
        'tqdm',
        'matplotlib',
        'scikit-learn',
    ]
)