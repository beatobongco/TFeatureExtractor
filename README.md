# TFeatureExtractor

Vectorize strings in 2 lines of code with the latest Transformer models c/o huggingface!

## Installation

This is meant to be used in noteboooks. If using for a script feel free to just copy the entire class.

For colab
```
!git clone https://github.com/beatobongco/TFeatureExtractor.git
!pip install /content/TFeatureExtractor/dist/TFeatureExtractor-0.0.1-py3-none-any.whl
```

For jupyter notebooks
```
!git clone https://github.com/beatobongco/TFeatureExtractor.git
!pip install TFeatureExtractor/dist/TFeatureExtractor-0.0.1-py3-none-any.whl
```

## Usage

```py
from tfeatureextractor import TFeatureExtractor
# Instantiate a feature extractor for BERT
# this can be "roberta", "distilbert", and many more
# please refer to the main code
tfe = TFeatureExtractor("bert")
# Encode a list of strings
encs = tfe.encode(["I'm sentence #1", "I'm sentence #2"])
# Get a np.array in return
assert encs.shape == (2, 768)
```