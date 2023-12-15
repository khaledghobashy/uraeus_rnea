import setuptools

short_description = "Modeling and Simulation of tree-like multibody systems using Recusrive Newton-Euler Algorithms"

setuptools.setup(
    name="uraeus.rnea",
    version="0.0.1.dev3",
    author="Khaled Ghobashy",
    author_email="khaled.ghobashy@live.com",
    description=short_description,
    # long_description = long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/khaledghobashy/uraeus_rnea",
    packages=setuptools.find_packages(exclude=("tests",)),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Development Status :: 1 - Planning",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "jax",
        "jaxlib",
        "matplotlib",
        "networkx",
        "pydantic",
    ],
)
