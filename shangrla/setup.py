"""
SHANGRLA: risk-limiting audits
"""

import os


DISTNAME = "shangrla"
DESCRIPTION = "Sets of Half-Average Nulls Generate Risk-Limiting Audits"
AUTHOR = "K. Jarrod Millman, Kellie Ottoboni, and Philip B. Stark"
AUTHOR_EMAIL = "pbstark@berkeley.edu"
URL = ""
LICENSE = "BSD License"
DOWNLOAD_URL = "http://github.com/pbstark/shangrla"


def parse_requirements_file(filename):
    with open(filename, encoding="utf-8") as fid:
        requires = [l.strip() for l in fid.readlines() if l]

    return requires


INSTALL_REQUIRES = parse_requirements_file("requirements.txt")
TESTS_REQUIRE = parse_requirements_file("requirements.txt")

with open("shangrla/__init__.py") as fid:
    for line in fid:
        if line.startswith("__version__"):
            VERSION = line.strip().split()[-1][1:-1]
            break

with open("README.md") as fh:
    LONG_DESCRIPTION = fh.read()


if __name__ == "__main__":

    from setuptools import setup

    setup(
        name=DISTNAME,
        version=VERSION,
        license=LICENSE,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        url=URL,
        download_url=DOWNLOAD_URL,
        classifiers=[
            "Development Status :: 0.3 - Alpha",
            "Environment :: Console",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3.10",
            "Topic :: Elections",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX",
            "Operating System :: Unix",
            "Operating System :: MacOS",
        ],
        install_requires=INSTALL_REQUIRES,
        tests_require=TESTS_REQUIRE,
        python_requires=">=3.10.4",
        packages=["shangrla", "shangrla.tests"],
        package_data={"shangrla.Examples.Data": ["*.csv", "*/*.csv", "*/*/*.csv", "*.json", "*.xlsx"]},
    )