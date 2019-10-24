# TFeatureExtractor

Vectorize strings in 2 lines of code with the latest Transformer models c/o https://github.com/huggingface/transformers!

## Installation

This is meant to be used in noteboooks too!

```
!pip install git+https://github.com/beatobongco/TFeatureExtractor.git
```

## Usage (GPU)

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

## Usage (CPU)

`TFeatureExtractor` should automatically detect if you're on CPU. 

However, please set `batch_size` to a lower number, depending on how powerful your CPU is. `16` is a great number to start with.

```py
from tfeatureextractor import TFeatureExtractor
# Instantiate a feature extractor for BERT
# this can be "roberta", "distilbert", and many more!
tfe = TFeatureExtractor("bert")
# Encode a list of strings
encs = tfe.encode(["I'm sentence #1", "I'm sentence #2"], batch_size=16)
# Get a np.array in return
assert encs.shape == (2, 768)
```