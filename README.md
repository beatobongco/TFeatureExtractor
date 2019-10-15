# TFeatureExtractor

Vectorize strings in 2 lines of code with the latest Transformer models

## Usage


```py
# Instantiate a feature extractor for BERT
tfe = TFeatureExtractor("bert")
# Encode a list of strings
encs = tfe.encode(["I'm sentence #1", "I'm sentence #2"])
# Get a np.array in return
assert encs.shape == (2, 768)
```