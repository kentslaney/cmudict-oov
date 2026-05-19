import json
import numpy as np
import onnxruntime as ort

def load_vocab(path="vocab.json"):
    with open(path, "r") as f:
        vocab = json.load(f)
    return vocab

def predict_onnx(word, session, vocab, max_len=50):
    char_to_idx = vocab["char_to_idx"]
    idx_to_phone = vocab["idx_to_phone"]

    SOS_CHAR = char_to_idx.get("<SOS>")
    EOS_CHAR = char_to_idx.get("<EOS>")
    PAD_CHAR = char_to_idx.get("<PAD>")

    SOS_PHONE = vocab["phone_to_idx"].get("<SOS>")
    EOS_PHONE = vocab["phone_to_idx"].get("<EOS>")
    PAD_PHONE = vocab["phone_to_idx"].get("<PAD>")

    # Prepare src sequence (exact length, batch 5 to match ONNX graph batch dimension if necessary)
    word = word.lower()
    src_tokens = [SOS_CHAR] + [char_to_idx.get(c, PAD_CHAR) for c in word] + [EOS_CHAR]

    src_seq_len = len(src_tokens)
    src_data = np.full((src_seq_len, 5), PAD_CHAR, dtype=np.int64)
    for i, token in enumerate(src_tokens):
        for b in range(5):
            src_data[i, b] = token

    tgt_tokens = [SOS_PHONE]

    for step in range(max_len):
        tgt_seq_len = len(tgt_tokens)
        tgt_data = np.full((tgt_seq_len, 5), PAD_PHONE, dtype=np.int64)
        for i, token in enumerate(tgt_tokens):
            for b in range(5):
                tgt_data[i, b] = token

        inputs = {
            'src': src_data,
            'tgt': tgt_data
        }

        output = session.run(['output'], inputs)[0]
        # Output shape: [tgt_seq_len, 5, vocab_size]

        # The next token is predicted at index tgt_seq_len - 1
        logits = output[tgt_seq_len - 1, 0, :]
        next_token = int(np.argmax(logits))

        tgt_tokens.append(next_token)
        if next_token == EOS_PHONE:
            break

    phones = [idx_to_phone.get(str(idx), str(idx)) for idx in tgt_tokens[1:-1]]
    return " ".join(phones)

if __name__ == "__main__":
    vocab = load_vocab()
    print("Loading ONNX model...")
    session = ort.InferenceSession("cmudict_transformer.onnx")

    test_words = ["antigravity", "hello", "world", "xylophone"]
    print("\nONNX Inference Results:")
    for w in test_words:
        print(f"{w} -> {predict_onnx(w, session, vocab)}")
