import torch
import torch.nn as nn
import glob
from torch.utils.data import DataLoader
from dataset import load_opus_en_hi, BilingualDataset
from model import Transformer
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEQ_LEN = 50
EPOCHS = 10
BATCH = 32

# load dataset + tokenizers (tokenizers are saved & reused internally)
ds, tokenizer_src, tokenizer_tgt = load_opus_en_hi(SEQ_LEN)

train_ds = BilingualDataset(ds, tokenizer_src, tokenizer_tgt, SEQ_LEN)
loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)

# model
model = Transformer(
    tokenizer_src.get_vocab_size(),
    tokenizer_tgt.get_vocab_size(),
    SEQ_LEN,
    512, 6, 8, 2048, 0.1
).to(DEVICE)

loss_fn = nn.CrossEntropyLoss(
    ignore_index=tokenizer_tgt.token_to_id("[PAD]")
)
opt = torch.optim.Adam(model.parameters(), lr=3e-4)

# (backward compatible) 
start_epoch = 0
checkpoint_files = glob.glob("/kaggle/working/transformer_epoch_*.pt")

if len(checkpoint_files) > 0:
    checkpoint_files.sort(
        key=lambda x: int(x.split("_")[-1].split(".")[0])
    )
    latest_checkpoint = checkpoint_files[-1]
    print("Resuming from:", latest_checkpoint)

    checkpoint = torch.load(latest_checkpoint, map_location=DEVICE)

    # new checkpoint format
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
        opt.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"]

    # old checkpoint format (model only)
    else:
        model.load_state_dict(checkpoint)
        start_epoch = int(latest_checkpoint.split("_")[-1].split(".")[0])

# ===== training loop =====
for e in range(start_epoch, EPOCHS):
    model.train()
    total_loss = 0

    for batch in tqdm(loader, desc=f"Epoch {e+1}"):
        encoder_input = batch["encoder_input"].to(DEVICE)
        decoder_input = batch["decoder_input"].to(DEVICE)
        encoder_mask = batch["encoder_mask"].to(DEVICE)
        decoder_mask = batch["decoder_mask"].to(DEVICE)
        label = batch["label"].to(DEVICE)

        opt.zero_grad()

        output = model(
            encoder_input,
            decoder_input,
            encoder_mask,
            decoder_mask
        )

        loss = loss_fn(
            output.reshape(-1, output.size(-1)),
            label.reshape(-1)
        )

        loss.backward()
        opt.step()

        total_loss += loss.item()

    # save checkpoint (new format)
    torch.save(
        {
            "epoch": e + 1,
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict()
        },
        f"/kaggle/working/transformer_epoch_{e+1}.pt"
    )

    print("Avg loss:", total_loss / len(loader))
