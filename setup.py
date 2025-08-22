from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="ML PROJECT",
    version= "0.1",
    author= "Upendra",
    packages= find_packages(),
    install_requires = requirements,
)