import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="TFeatureExtractor",
    version="0.1.2",
    author="Beato Bongco",
    author_email="beatobongco@gmail.com",
    description="Vectorize strings in 2 lines of code with the latest Transformer models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/beatobongco/TFeatureVectorizer",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy', 'torch', 'transformers']
)
