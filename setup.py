#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

dist = setup(
    name="agginc",
    version="1.0.0",
    description="Aggregated Incomplete Kernel Tests",
    author="Antonin Schrab",
    author_email="a.lastname@ucl.ac.uk",
    license="MIT License",
    packages=["agginc", ],
    install_requires=["numpy", "scipy", "jax", "jaxlib", "psutil", "gputil"],
    python_requires=">=3.9",
)
