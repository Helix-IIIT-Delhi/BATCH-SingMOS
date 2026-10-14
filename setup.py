from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="batch-singmos",
    version="1.0.0",
    author="Orchid Chetia Phukan, Girish, Mohd Mujtaba Akhtar, Swarup Ranjan Behera, Pailla Balakrishna Reddy, Arun Balaji Buduru, Rajesh Sharma",
    description="BATCH: Bhattacharyya Distance Fusion for Singing Voice Quality Assessment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Helix-IIIT-Delhi/BATCH-SingMOS",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
