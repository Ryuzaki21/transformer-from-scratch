import os
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


def causal_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0


def train_and_save_tokenizer(sentences, path, vocab_size=32000):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[SOS]", "[EOS]", "[UNK]"]
    )

    tokenizer.train_from_iterator(sentences, trainer)
    tokenizer.save(path)
    return tokenizer


def load_or_train_tokenizers(dataset, vocab_size=32000):
    src_path = "tokenizer_src.json"
    tgt_path = "tokenizer_tgt.json"

    if os.path.exists(src_path) and os.path.exists(tgt_path):
        tokenizer_src = Tokenizer.from_file(src_path)
        tokenizer_tgt = Tokenizer.from_file(tgt_path)
        print("Loaded existing tokenizers")
    else:
        print("Training new tokenizers")
        tokenizer_src = train_and_save_tokenizer(
            (x["translation"]["en"] for x in dataset),
            src_path,
            vocab_size
        )
        tokenizer_tgt = train_and_save_tokenizer(
            (x["translation"]["hi"] for x in dataset),
            tgt_path,
            vocab_size
        )

    return tokenizer_src, tokenizer_tgt


def load_opus_en_hi(seq_len=50, vocab_size=32000):
    dataset = load_dataset("Helsinki-NLP/opus-100", "en-hi", split="train")
    dataset = dataset.shuffle(seed=42).select(range(100000))

    tokenizer_src, tokenizer_tgt = load_or_train_tokenizers(
        dataset, vocab_size
    )

    return dataset, tokenizer_src, tokenizer_tgt


class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, seq_len=50):
        self.seq_len = seq_len
        self.samples = []

        self.src_sos = tokenizer_src.token_to_id("[SOS]")
        self.src_eos = tokenizer_src.token_to_id("[EOS]")
        self.src_pad = tokenizer_src.token_to_id("[PAD]")

        self.tgt_sos = tokenizer_tgt.token_to_id("[SOS]")
        self.tgt_eos = tokenizer_tgt.token_to_id("[EOS]")
        self.tgt_pad = tokenizer_tgt.token_to_id("[PAD]")

        for x in ds:
            src = tokenizer_src.encode(x["translation"]["en"]).ids
            tgt = tokenizer_tgt.encode(x["translation"]["hi"]).ids
            if len(src) <= seq_len - 2 and len(tgt) <= seq_len - 1:
                self.samples.append((src, tgt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src, tgt = self.samples[idx]

        encoder_input = [self.src_sos] + src + [self.src_eos]
        encoder_input += [self.src_pad] * (self.seq_len - len(encoder_input))

        decoder_input = [self.tgt_sos] + tgt
        decoder_input += [self.tgt_pad] * (self.seq_len - len(decoder_input))

        label = tgt + [self.tgt_eos]
        label += [self.tgt_pad] * (self.seq_len - len(label))

        encoder_input = torch.tensor(encoder_input)
        decoder_input = torch.tensor(decoder_input)
        label = torch.tensor(label)

        encoder_mask = (encoder_input != self.src_pad).unsqueeze(0).unsqueeze(0)
        padding_mask = (decoder_input != self.tgt_pad).unsqueeze(0).unsqueeze(1)
        decoder_mask = padding_mask & causal_mask(self.seq_len).unsqueeze(0)

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask,
            "label": label
        }
