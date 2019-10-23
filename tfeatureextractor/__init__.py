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
from keras.preprocessing import sequence


class PadTruncCollator(object):
    """
    A pytorch Collator that sets the max_seq_len argument at the batch level.
    """
    def __init__(self, percentile=100.0, padding="post", pad_token=0):
        """
        percentile float - indicates the percentile of the sequence lengths in a batch that will be used as basis for the max_seq_len for that batch.
        padding str - indicates whether padding should be added at the end or beginning of the string (post or pre). This is the approach taken during fine-tuning of bert-base-uncased)

        """
        self.percentile = percentile
        self.padding = padding
        self.pad_token = pad_token

    def __call__(self, batch):

        texts, lens = zip(*batch)
        lens = np.array(lens)
        max_len = int(np.percentile(lens, self.percentile))
        texts = torch.tensor(
            sequence.pad_sequences(
                texts, maxlen=max_len, padding=self.padding, value=self.pad_token
            ),
            dtype=torch.long,
        )

        return (texts,)
    
    
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
        batch_size=128,
        pad_token=0,
        verbose=True,
        max_len_percentile=100,
        padding="post",
    ):
        """Encode a list of strings with selected transformer.
        
        Returns np.array of embeddings.
        
        TODO: add pooling_strategy = ("mean", "max")
        """
        input_tensor, length_sorted_idx, length_tensor = self.encode_strings(
            input_strings, max_length, pad_token, padding
        )
        input_dataset = TensorDataset(input_tensor, length_tensor)
        input_sampler = SequentialSampler(input_dataset)
        pad_trunc_collate = PadTruncCollator(max_len_percentile, padding, pad_token)
        input_dataloader = DataLoader(
            input_dataset,
            sampler=input_sampler,
            batch_size=batch_size,
            collate_fn=pad_trunc_collate,
        )
        embeddings = torch.Tensor().to(self.device)
        for batch in input_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {"input_ids": batch[0]}
                last_hidden_states = self.model(**inputs)[pooling_layer]  # bs x sl x hr
            mean_pooled = torch.mean(last_hidden_states, 1)  # bs x hr
            embeddings = torch.cat((embeddings, mean_pooled), dim=0)
            print(
                f"\rEncoding strings ({len(embeddings)} / {len(input_strings)}): {round(len(embeddings) / len(input_strings) * 100, 4)}%",
                end="",
                flush=True,
            )
        return embeddings.cpu().numpy()

    def encode_strings(self, input_strings, max_length, pad_token, padding):
        """
        Returns encoded strings (sorted based on length), their lengths, and the sorting indices used
        """
        input_list = []
        length_list = []
        for s in input_strings:
            input_ids = self.tokenizer.encode(
                s, max_length=max_length, add_special_tokens=True
            )
            length = len(input_ids)
            input_list.append(input_ids)
            length_list.append(length)

        input_tensor = torch.Tensor(
            sequence.pad_sequences(
                input_list, maxlen=max_length, padding=padding, value=pad_token
            )
            
        ).long()
        length_tensor = torch.Tensor(length_list).long()
        length_sorted_idx = torch.argsort(length_tensor, descending=True)
        return (
            input_tensor[length_sorted_idx],
            length_sorted_idx,
            length_tensor[length_sorted_idx],
        )

