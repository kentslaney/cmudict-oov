import torch
import json
from train import Seq2SeqTransformer, char_to_idx, phone_to_idx, idx_to_char, idx_to_phone, EMBED_SIZE, NUM_HEADS, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DIM_FEEDFORWARD, DROPOUT

device = torch.device('cpu')

# 1. Save vocabularies
vocab_data = {
    "char_to_idx": char_to_idx,
    "phone_to_idx": phone_to_idx,
    "idx_to_char": idx_to_char,
    "idx_to_phone": idx_to_phone
}
with open("vocab.json", "w") as f:
    json.dump(vocab_data, f)

print("Vocabularies saved to vocab.json")

# 2. Load model
model = Seq2SeqTransformer(
    len(char_to_idx), len(phone_to_idx),
    EMBED_SIZE, NUM_HEADS, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DIM_FEEDFORWARD, DROPOUT
).to(device)

model.load_state_dict(torch.load("cmudict_transformer.pth", map_location=device))
model.eval()

# Bypass mask creation during export to avoid Dynamo guard errors
model.create_mask = lambda src, tgt: (None, None, None, None)

# 3. Export to ONNX
src_len = 50
tgt_len = 50
batch_size = 5

src = torch.randint(0, len(char_to_idx), (src_len, batch_size), dtype=torch.long)
tgt = torch.randint(0, len(phone_to_idx), (tgt_len, batch_size), dtype=torch.long)

torch.onnx.export(
    model,
    (src, tgt),
    "cmudict_transformer.onnx",
    export_params=True,
    external_data=False,
    do_constant_folding=True,
    input_names=['src', 'tgt'],
    output_names=['output']
)

print("Model exported to cmudict_transformer.onnx")
