import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tecoradors-elunico",
    version="4.1.0",
    author="Thomas Povinelli",
    author_email="tompov227@gmail.com",
    description="A small collection of decorators I like to use often",
    long_description=long_description,
    long_description_content_type="text/markdown",
include_package_data=True,
    url="https://github.com/elunico/tecoradors",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
)
