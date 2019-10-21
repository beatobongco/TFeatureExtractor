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

from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm

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

    def encode(
        self,
        input_strings,
        pooling_layer=-2,
        max_length=512,
        batch_size=10,
        pad_token=0,
        pad_on_left=False,
        verbose=True,
    ):
        """Encode a list of strings with selected transformer.
        
        Returns np.array of embeddings.
        
        TODO: add pooling_strategy = ("mean", "max")
        """
        input_tensor = self.encode_strings(
            input_strings, max_length, pad_token, pad_on_left
        )
        input_dataset = TensorDataset(input_tensor)
        input_sampler = SequentialSampler(input_dataset)
        input_dataloader = DataLoader(
            input_dataset, sampler=input_sampler, batch_size=batch_size
        )
        embeddings = torch.Tensor().to(self.device)
        input_dataloader = tqdm(input_dataloader) if verbose else input_dataloader
        for step, batch in enumerate(input_dataloader):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {"input_ids": batch[0]}
                last_hidden_states = self.model(**inputs)[pooling_layer]  # bs x sl x hr
            mean_pooled = torch.mean(last_hidden_states, 1)  # bs x hr
            embeddings = torch.cat((embeddings, mean_pooled), dim=0)
        return embeddings.cpu().numpy()

    def pad_ids(self, input_ids, max_length, pad_token=0, pad_on_left=False):
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
        return input_ids

    def encode_strings(self, input_strings, max_length, pad_token, pad_on_left):
        input_list = []
        for s in input_strings:
            input_ids = self.tokenizer.encode(
                s, max_length=max_length, add_special_tokens=True
            )
            input_ids_padded = self.pad_ids(
                input_ids, max_length, pad_token, pad_on_left
            )
            input_list.append(input_ids_padded)
        input_tensor = torch.Tensor(input_list).long()
        return input_tensor
