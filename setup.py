from setuptools import setup, find_packages

with open("requirements.txt") as f:
    install_requires = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="dpamplify",
    version="0.1.0",
    author="[Author]",
    description=(
        "Byzantine attacks exploiting the Gaussian DP mechanism "
        "in federated learning"
    ),
    url="https://github.com/[YOUR_USERNAME]/dpamplify",
    python_requires=">=3.11",
    packages=find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
