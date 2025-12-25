%%writefile /kaggle/working/MyTransformer/eval.py
import torch
import glob
from dataset import load_opus_en_hi, causal_mask
from model import Transformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 50

# load dataset + SAME tokenizers used in training
ds, tokenizer_src, tokenizer_tgt = load_opus_en_hi(SEQ_LEN)

# model
model = Transformer(
    tokenizer_src.get_vocab_size(),
    tokenizer_tgt.get_vocab_size(),
    SEQ_LEN,
    512, 6, 8, 2048, 0.1
).to(DEVICE)

# ===== load latest checkpoint automatically =====
checkpoint_files = glob.glob("/kaggle/working/transformer_epoch_*.pt")

if len(checkpoint_files) == 0:
    raise FileNotFoundError("No model checkpoint found.")

checkpoint_files.sort(
    key=lambda x: int(x.split("_")[-1].split(".")[0])
)
latest_checkpoint = checkpoint_files[-1]
print("Loading checkpoint:", latest_checkpoint)

checkpoint = torch.load(latest_checkpoint, map_location=DEVICE)

# support old + new checkpoint formats
if isinstance(checkpoint, dict) and "model_state" in checkpoint:
    model.load_state_dict(checkpoint["model_state"])
else:
    model.load_state_dict(checkpoint)

model.eval()

# special tokens
SRC_PAD = tokenizer_src.token_to_id("[PAD]")
SRC_SOS = tokenizer_src.token_to_id("[SOS]")
SRC_EOS = tokenizer_src.token_to_id("[EOS]")

TGT_PAD = tokenizer_tgt.token_to_id("[PAD]")
TGT_SOS = tokenizer_tgt.token_to_id("[SOS]")
TGT_EOS = tokenizer_tgt.token_to_id("[EOS]")


def translate(sentence):
    encoder_input = (
        [SRC_SOS]
        + tokenizer_src.encode(sentence).ids
        + [SRC_EOS]
    )
    encoder_input += [SRC_PAD] * (SEQ_LEN - len(encoder_input))
    encoder_input = torch.tensor(encoder_input).unsqueeze(0).to(DEVICE)

    encoder_mask = (encoder_input != SRC_PAD).unsqueeze(1).unsqueeze(1)

    with torch.no_grad():
        encoder_output = model.encode(encoder_input, encoder_mask)

    decoder_input = [TGT_SOS]

    for _ in range(SEQ_LEN):
        tgt = torch.tensor(decoder_input).unsqueeze(0).to(DEVICE)

        padding_mask = (tgt != TGT_PAD).unsqueeze(1).unsqueeze(1)
        causal = causal_mask(tgt.size(1)).unsqueeze(0).to(DEVICE)
        decoder_mask = padding_mask & causal

        with torch.no_grad():
            decoder_output = model.decode(
                tgt, encoder_output, decoder_mask, encoder_mask
            )
            logits = model.proj(decoder_output)
            next_token = logits[0, -1].argmax().item()

        decoder_input.append(next_token)

        if next_token == TGT_EOS:
            break

    return tokenizer_tgt.decode(
        decoder_input[1:], skip_special_tokens=True
    )


print(translate("I love to play football"))
print(translate("She is my friend"))
print(translate("This is my house"))


