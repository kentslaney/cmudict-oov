import json
import numpy as np
import onnxruntime as ort

def load_vocab(path="vocab.json"):
    with open(path, "r") as f:
        vocab = json.load(f)
    return vocab

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def predict_onnx(word, session, vocab, beam_size=5, max_len=50):
    char_to_idx = vocab["char_to_idx"]
    idx_to_phone = vocab["idx_to_phone"]

    SOS_CHAR = char_to_idx.get("<SOS>")
    EOS_CHAR = char_to_idx.get("<EOS>")
    PAD_CHAR = char_to_idx.get("<PAD>")

    SOS_PHONE = vocab["phone_to_idx"].get("<SOS>")
    EOS_PHONE = vocab["phone_to_idx"].get("<EOS>")
    PAD_PHONE = vocab["phone_to_idx"].get("<PAD>")

    vocab_size = len(idx_to_phone)

    word = word.lower()
    src_tokens = [SOS_CHAR] + [char_to_idx.get(c, PAD_CHAR) for c in word] + [EOS_CHAR]
    src_seq_len = len(src_tokens)

    completed_beams = []
    active_beams = [{"tokens": [SOS_PHONE], "score": 0.0}]

    for step in range(max_len):
        if not active_beams:
            break

        batch_size = len(active_beams)
        tgt_seq_len = len(active_beams[0]["tokens"])

        src_data = np.full((src_seq_len, batch_size), PAD_CHAR, dtype=np.int64)
        for i, token in enumerate(src_tokens):
            for b in range(batch_size):
                src_data[i, b] = token

        tgt_data = np.full((tgt_seq_len, batch_size), PAD_PHONE, dtype=np.int64)
        for i in range(tgt_seq_len):
            for b in range(batch_size):
                tgt_data[i, b] = active_beams[b]["tokens"][i]

        inputs = {
            'src': src_data,
            'tgt': tgt_data
        }

        output = session.run(['output'], inputs)[0]

        all_candidates = []
        for b in range(batch_size):
            logits = output[tgt_seq_len - 1, b, :]
            probs = softmax(logits)
            log_probs = np.log(np.maximum(probs, 1e-10))

            for v in range(vocab_size):
                all_candidates.append({
                    "tokens": active_beams[b]["tokens"] + [v],
                    "score": active_beams[b]["score"] + log_probs[v]
                })

        all_candidates.sort(key=lambda x: x["score"], reverse=True)

        active_beams = []
        for cand in all_candidates:
            if cand["tokens"][-1] == EOS_PHONE:
                completed_beams.append(cand)
            else:
                active_beams.append(cand)

            if len(active_beams) + len(completed_beams) >= beam_size:
                break

        if len(completed_beams) >= beam_size:
            break

    completed_beams.extend(active_beams)
    completed_beams.sort(key=lambda x: x["score"], reverse=True)

    best_beam = completed_beams[0]
    tokens = best_beam["tokens"][1:]
    if tokens and tokens[-1] == EOS_PHONE:
        tokens = tokens[:-1]

    phones = [idx_to_phone.get(str(idx), str(idx)) for idx in tokens]
    return " ".join(phones)

if __name__ == "__main__":
    vocab = load_vocab()
    print("Loading ONNX model...")
    session = ort.InferenceSession("cmudict_transformer.onnx")

    test_words = ["antigravity", "hello", "world", "xylophone"]
    print("\nONNX Inference Results:")
    for w in test_words:
        print(f"{w} -> {predict_onnx(w, session, vocab)}")
