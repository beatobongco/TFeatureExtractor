# TFeatureExtractor

Vectorize strings in 2 lines of code with the latest Transformer models c/o https://github.com/huggingface/transformers!

## Installation

This is meant to be used in noteboooks too!

```
!pip install git+https://github.com/beatobongco/TFeatureExtractor.git
```

## Usage

```py
from tfeatureextractor import TFeatureExtractor
# Instantiate a feature extractor for BERT
# this can be "roberta", "distilbert", and many more!
tfe = TFeatureExtractor("bert")
# Encode a list of strings
encs = tfe.encode(["I'm sentence #1", "I'm sentence #2"])
# Get a np.array in return
assert encs.shape == (2, 768)
```