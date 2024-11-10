from setuptools import setup, find_packages

setup(
    name="mleda",
    version="0.1.0",
    author="MohammadJavad Yousefi",
    author_email="mj.yousefi.dev@gmail.com",
    description="Automated EDA tool for machine learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mj-yousefi/mleda",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "matplotlib",
        "seaborn",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)