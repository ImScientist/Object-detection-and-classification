import os

from setuptools import find_packages, setup

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# PACKAGE_DIR = os.path.join(BASE_DIR, "superset", "static", "assets")

# with io.open("README.md", encoding="utf-8") as f:
#     long_description = f.read()

# with open('requirements.txt') as f:
#     requirements = f.readlines()

setup(
    name="clouds",
    description="Cloud detection project",
    # long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    # scripts=["superset/bin/superset"],
    # install_requires=requirements,
    author="Anton Ivanov",
    email="a.ivanov@kigroup.de",
    classifiers=["Programming Language :: Python :: 3.7"],
)
