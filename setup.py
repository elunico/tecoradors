import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tecoradors-elunico",
    version="2.0.0",
    author="Thomas Povinelli",
    author_email="author@example.com",
    description="A small collection of decorators I like to use often",
    long_description=long_description,
    long_description_content_type="text/markdown",
include_package_data=True,
    url="https://gist.github.com/elunico/bde1125c1c31fae18f64a6437f2fbe03",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
)