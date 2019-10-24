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

def print_sparingly(message, measured, total, every=1000):
    """Prints a progress string sparingly so as not to take up too much computation"""
    if measured % every == 0 or (total - measured) <= 1: 
        print(f"\r{message} ({measured} / {total}): {round(measured / total * 100, 2)}%", end="", flush=True)

class PadTruncCollator(object):
    """
    A pytorch Collator that sets the max_seq_len argument at the batch level.
    """

    def __init__(self, percentile=100.0):
        """
        percentile float - indicates the percentile of the sequence lengths in a batch that will be used as basis for the max_seq_len for that batch.
        """
        self.percentile = percentile

    def __call__(self, batch):

        texts, lens = zip(*batch)
        lens = np.array(lens)
        max_len = int(np.percentile(lens, self.percentile))
        texts = torch.tensor(
            sequence.pad_sequences(
                texts, maxlen=max_len, padding="post", truncating="post"
            ),
            dtype=torch.long,
        )

        return (texts,)


class TFeatureExtractor:
    """Utility class that uses Transformer models to vectorize batches of strings"""

    def __init__(
        self, model_type, use_cpu=False, saved_model_directory=None, from_tf=False
    ):
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
        model_dir = saved_model_directory or m[2]
        self.model = m[0].from_pretrained(model_dir, from_tf=from_tf)
        self.tokenizer = m[1].from_pretrained(model_dir, from_tf=from_tf)
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
        batch_size=128
    ):
        """Encode a list of strings with selected transformer.
        
        Returns np.array of embeddings.
        
        TODO: add pooling_strategy = ("mean", "max")
        """
        input_tensor, length_sorted_idx_reversed, length_tensor = self.encode_strings(
            input_strings, max_length
        )
        input_dataset = TensorDataset(input_tensor, length_tensor)
        input_sampler = SequentialSampler(input_dataset)
        pad_trunc_collate = PadTruncCollator()
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
            print_sparingly(
                "Encoding strings",
                measured=len(embeddings),
                total=len(input_strings),
                every=batch_size * 4
            )
        print("")
        return embeddings[length_sorted_idx_reversed].cpu().numpy()

    def encode_strings(self, input_strings, max_length):
        """
        Returns encoded strings (sorted based on length), their lengths, and the sorting indices used
        """
        input_list = []
        length_list = []
        for s in input_strings:
            input_ids = self.tokenizer.encode(
                s, max_length=max_length, add_special_tokens=True
            )
            input_list.append(input_ids)
            length_list.append(len(input_ids))
            print_sparingly(
                "Tokenizing strings",
                measured=len(input_list),
                total=len(input_strings)
            )
        print("")

        input_tensor = torch.Tensor(
            sequence.pad_sequences(
                input_list, maxlen=max_length, padding="post", truncating="post"
            )
        ).long()
        length_tensor = torch.Tensor(length_list).long()
        sorted_indices = torch.argsort(length_tensor, descending=True)
        length_tensor_sorted = length_tensor[sorted_indices]
        sorted_indices_reversed = torch.zeros(length_tensor_sorted.size()[0]).long()
        sorted_indices_reversed[sorted_indices] = torch.arange(
            length_tensor_sorted.size()[0]
        )
        return (input_tensor[sorted_indices], sorted_indices_reversed, length_tensor_sorted)
