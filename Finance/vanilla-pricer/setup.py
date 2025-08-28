from setuptools import setup, find_packages

setup(
    name="vanilla-pricer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["scipy", "numpy"],  # Necessary dependencies
    author="Alexis PERDEREAU",
    description="Vanilla options pricing library",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=["Programming Language :: Python :: 3"],
)
