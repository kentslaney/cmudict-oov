let session;
let vocab;
const PAD_IDX = 0n;
const SOS_IDX = 1n;
const EOS_IDX = 2n;

const statusEl = document.getElementById('status');
const inputEl = document.getElementById('wordInput');
const btnEl = document.getElementById('predictBtn');
const resultsEl = document.getElementById('resultsContainer');

async function init() {
    try {
        // Load vocab
        const vocabRes = await fetch('vocab.json');
        vocab = await vocabRes.json();
        
        // Load ONNX model
        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
        session = await ort.InferenceSession.create('cmudict_transformer.onnx');
        
        statusEl.textContent = 'Model loaded successfully. Enter a word!';
        statusEl.style.color = '#27ae60';
        inputEl.disabled = false;
        btnEl.disabled = false;
    } catch (e) {
        statusEl.textContent = `Error loading: ${e.message}`;
        statusEl.style.color = '#e74c3c';
        console.error(e);
    }
}

btnEl.addEventListener('click', async () => {
    const word = inputEl.value.trim().toLowerCase();
    if (!word) return;
    
    btnEl.disabled = true;
    inputEl.disabled = true;
    resultsEl.innerHTML = '<div style="text-align:center;">Predicting...</div>';
    
    try {
        const topBeams = await predict(word, 5);
        displayResults(topBeams);
    } catch (e) {
        resultsEl.innerHTML = `<div style="color:red;">Prediction error: ${e.message}</div>`;
        console.error(e);
    }
    
    btnEl.disabled = false;
    inputEl.disabled = false;
    inputEl.focus();
});

inputEl.addEventListener('keyup', (e) => {
    if (e.key === 'Enter') btnEl.click();
});

function softmax(arr) {
    const max = Math.max(...arr);
    const exps = arr.map(x => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(x => x / sum);
}

async function predict(word, beamSize = 5) {
    // 1. Prepare src
    let srcTokens = [SOS_IDX];
    for (let i = 0; i < word.length; i++) {
        const char = word[i];
        if (vocab.char_to_idx[char] !== undefined) {
            srcTokens.push(BigInt(vocab.char_to_idx[char]));
        } else {
            srcTokens.push(PAD_IDX); // unknown char
        }
    }
    srcTokens.push(EOS_IDX);
    
    let completedBeams = [];
    let activeBeams = [{ tokens: [SOS_IDX], score: 0.0 }];
    const MAX_LEN = 50;
    const VOCAB_SIZE = Object.keys(vocab.idx_to_phone).length;
    
    for (let step = 0; step < MAX_LEN; step++) {
        if (activeBeams.length === 0) break;
        
        const tgtSeqLen = activeBeams[0].tokens.length;
        
        // Prepare src tensor [50, 5]
        const srcData = new BigInt64Array(50 * 5);
        for (let i = 0; i < 50; i++) {
            for (let b = 0; b < 5; b++) {
                if (i < srcTokens.length) {
                    srcData[i * 5 + b] = srcTokens[i];
                } else {
                    srcData[i * 5 + b] = PAD_IDX;
                }
            }
        }
        
        // Prepare tgt tensor [50, 5]
        const tgtData = new BigInt64Array(50 * 5);
        for (let i = 0; i < 50; i++) {
            for (let b = 0; b < 5; b++) {
                // If this beam exists, pad its tokens, otherwise copy beam 0
                const beamIdx = b < activeBeams.length ? b : 0;
                if (i < activeBeams[beamIdx].tokens.length) {
                    tgtData[i * 5 + b] = activeBeams[beamIdx].tokens[i];
                } else {
                    tgtData[i * 5 + b] = PAD_IDX;
                }
            }
        }
        
        const srcTensor = new ort.Tensor('int64', srcData, [50, 5]);
        const tgtTensor = new ort.Tensor('int64', tgtData, [50, 5]);
        
        const feeds = { src: srcTensor, tgt: tgtTensor };
        const results = await session.run(feeds);
        const output = results.output.data; // Float32Array [tgtSeqLen, batchSize, vocabSize]
        
        let allCandidates = [];
        // The output shape is [50, 5, VOCAB_SIZE]. We want the step corresponding to tgtSeqLen - 1
        // We know we padded tgt to 50, but the "last" valid token was at index (tgtSeqLen - 1).
        const lastStepOffset = (tgtSeqLen - 1) * 5 * VOCAB_SIZE;
        
        for (let b = 0; b < activeBeams.length; b++) {
            // Get logits for this beam
            const logits = new Float32Array(VOCAB_SIZE);
            for (let v = 0; v < VOCAB_SIZE; v++) {
                logits[v] = output[lastStepOffset + b * VOCAB_SIZE + v];
            }
            
            // Log softmax
            const probs = softmax(Array.from(logits));
            const logProbs = probs.map(p => Math.log(Math.max(p, 1e-10)));
            
            for (let v = 0; v < VOCAB_SIZE; v++) {
                allCandidates.push({
                    tokens: [...activeBeams[b].tokens, BigInt(v)],
                    score: activeBeams[b].score + logProbs[v]
                });
            }
        }
        
        // Sort all candidates by score descending
        allCandidates.sort((a, b) => b.score - a.score);
        
        activeBeams = [];
        for (const cand of allCandidates) {
            if (cand.tokens[cand.tokens.length - 1] === EOS_IDX) {
                completedBeams.push(cand);
            } else {
                activeBeams.push(cand);
            }
            
            if (activeBeams.length + completedBeams.length >= beamSize) {
                break;
            }
        }
        
        if (completedBeams.length >= beamSize) {
            break;
        }
    }
    
    // Fallback if not enough completed beams
    completedBeams.push(...activeBeams);
    completedBeams.sort((a, b) => b.score - a.score);
    
    return completedBeams.slice(0, beamSize).map(b => {
        // Remove SOS and EOS
        const t = b.tokens.slice(1);
        if (t.length > 0 && t[t.length - 1] === EOS_IDX) t.pop();
        
        const phoneStrs = t.map(v => vocab.idx_to_phone[v.toString()]);
        
        // Convert log likelihood back to probability approx (very small, so just show exp score)
        const prob = Math.exp(b.score);
        return { phones: phoneStrs.join(" "), prob: prob, logScore: b.score };
    });
}

function displayResults(beams) {
    resultsEl.innerHTML = '';
    beams.forEach((b, i) => {
        const p = document.createElement('div');
        p.className = 'result-item';
        
        const phonesSpan = document.createElement('span');
        phonesSpan.className = 'phones';
        phonesSpan.textContent = b.phones;
        
        const probSpan = document.createElement('span');
        probSpan.className = 'prob';
        probSpan.textContent = `P ≈ ${(b.prob * 100).toFixed(4)}%`;
        
        p.appendChild(phonesSpan);
        p.appendChild(probSpan);
        resultsEl.appendChild(p);
    });
}

// Start
init();
