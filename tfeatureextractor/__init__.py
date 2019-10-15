import torch
import numpy as np
from transformers import (
    BertModel,
    BertTokenizer,
    OpenAIGPTModel,
    OpenAIGPTTokenizer,
    GPT2Model,
    GPT2Tokenizer,
    CTRLModel,
    CTRLTokenizer,
    TransfoXLModel,
    TransfoXLTokenizer,
    XLNetModel,
    XLNetTokenizer,
    XLMModel,
    XLMTokenizer,
    DistilBertModel,
    DistilBertTokenizer,
    RobertaModel,
    RobertaTokenizer,
)

class TFeatureExtractor:
    """Utility class that uses Transformer models to vectorize batches of strings"""

    def __init__(self, model_type, use_cpu=False):
        """Initializes the feature extractor with the kind of Transformer model you want to use"""
        MODELS = {
            "bert": (BertModel, BertTokenizer, "bert-base-uncased"),
            "gpt": (OpenAIGPTModel, OpenAIGPTTokenizer, "openai-gpt"),
            "gpt2": (GPT2Model, GPT2Tokenizer, "gpt2"),
            "ctrl": (CTRLModel, CTRLTokenizer, "ctrl"),
            "transformer-xl": (TransfoXLModel, TransfoXLTokenizer, "transfo-xl-wt103"),
            "xl-net": (XLNetModel, XLNetTokenizer, "xlnet-base-cased"),
            "xlm": (XLMModel, XLMTokenizer, "xlm-mlm-enfr-1024"),
            "distilbert": (
                DistilBertModel,
                DistilBertTokenizer,
                "distilbert-base-uncased",
            ),
            "roberta": (RobertaModel, RobertaTokenizer, "roberta-base"),
        }
        m = MODELS[model_type]
        self.model = m[0].from_pretrained(m[2])
        self.tokenizer = m[1].from_pretrained(m[2])
        if use_cpu:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # Turns off dropout and batchnorm
        self.model.eval()

    def encode(self, input_strings, pooling_layer=-2, max_length=512):
        """Encode a list of strings with selected transformer.
        
        Returns np.array of embeddings.
        
        TODO: add pooling_strategy = ("mean", "max")
        """
        embeddings = []
        for s in input_strings:
            # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
            input_ids = torch.tensor(
                [
                    self.tokenizer.encode(
                        s, add_special_tokens=True, max_length=max_length
                    )
                ]
            ).to(self.device)
            with torch.no_grad():
                last_hidden_states = self.model(input_ids)[
                    pooling_layer
                ]  # Models outputs are now tuples
            mean_pooled = torch.mean(last_hidden_states[0], 0)
            embeddings.append(mean_pooled.cpu().numpy())
            print(
                f"\rEncoding strings ({len(embeddings)} / {len(input_strings)}): {round(len(embeddings) / len(input_strings) * 100, 4)}%",
                end="",
                flush=True,
            )
        print("\n")
        return np.array(embeddings)
