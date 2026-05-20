class G2p {
    static PAD = 0n
    static SOS = 1n
    static EOS = 2n

    static MAX_LEN = 64

    static model_url = 'g2p/v2.onnx'
    static vocab_url = 'g2p/vocab.json'

    #queue = Promise.resolve()

    constructor() {
        this.loading = Promise.all([
            fetch(G2p.vocab_url).then(x => x.json()),
            ort.InferenceSession.create(G2p.model_url)
        ]).then((([vocab, session]) => {
            this.vocab = vocab
            this.session = session
            this.vocab_size = Object.keys(vocab.idx_to_phone).length
        }).bind(this))
    }

    encode(word) {
        const srcTokens = [G2p.SOS]
        for (const char of word.toLowerCase()) {
            if (this.vocab.char_to_idx[char] !== undefined) {
                srcTokens.push(BigInt(this.vocab.char_to_idx[char]))
            } else {
                srcTokens.push(G2p.PAD) // unknown char
            }
        }
        srcTokens.push(G2p.EOS)
        return srcTokens
    }

    decode(tokens) {
        const trim = tokens.slice(1)
        if (trim.length > 0 && trim[trim.length - 1] === G2p.EOS) trim.pop()
        const phones = trim.map(x => this.vocab.idx_to_phone[x.toString()])
        return phones.join(" ")
    }

    softmax(arr) {
        const max = Math.max(...arr);
        const exps = arr.map(x => Math.exp(x - max));
        const sum = exps.reduce((a, b) => a + b, 0);
        return exps.map(x => x / sum);
    }

    enqueue(word, beams=5) {
        return this.#queue = this.#queue.then(() => this.predict(word, beams))
    }

    async predict(word, beams=5) {
        await this.loading
        const srcTokens = this.encode(word)

        let completedBeams = []
        let activeBeams = [{ tokens: [G2p.SOS], score: 0 }]

        for (let step = 0; step < G2p.MAX_LEN; step++) {
            if (activeBeams.length === 0) break

            const batchSize = activeBeams.length
            const tgtSeqLen = activeBeams[0].tokens.length
            const srcSeqLen = srcTokens.length

            // Prepare src tensor [srcSeqLen, batchSize]
            const srcData = new BigInt64Array(srcSeqLen * batchSize)
            for (let i = 0; i < srcSeqLen; i++) {
                for (let b = 0; b < batchSize; b++) {
                    srcData[i * batchSize + b] = srcTokens[i]
                }
            }

            // Prepare tgt tensor [tgtSeqLen, batchSize]
            const tgtData = new BigInt64Array(tgtSeqLen * batchSize)
            for (let i = 0; i < tgtSeqLen; i++) {
                for (let b = 0; b < batchSize; b++) {
                    tgtData[i * batchSize + b] = activeBeams[b].tokens[i]
                }
            }

            const srcTensor = new ort.Tensor(
                'int64', srcData, [srcSeqLen, batchSize])
            const tgtTensor = new ort.Tensor(
                'int64', tgtData, [tgtSeqLen, batchSize])

            const feeds = { src: srcTensor, tgt: tgtTensor }
            const results = await this.session.run(feeds)
            // Float32Array [tgtSeqLen, batchSize, vocabSize]
            const output = results.output.data

            let allCandidates = []
            const lastStepOffset = (tgtSeqLen - 1) * batchSize * this.vocab_size

            for (let b = 0; b < batchSize; b++) {
                // Get logits for this beam
                const logits = new Float32Array(this.vocab_size)
                for (let v = 0; v < this.vocab_size; v++) {
                    logits[v] = output[lastStepOffset + b * this.vocab_size + v]
                }

                // Log softmax
                const probs = this.softmax(Array.from(logits));
                const logProbs = probs.map(p => Math.log(Math.max(p, 1e-10)));

                for (let v = 0; v < this.vocab_size; v++) {
                    allCandidates.push({
                        tokens: [...activeBeams[b].tokens, BigInt(v)],
                        score: activeBeams[b].score + logProbs[v]
                    })
                }
            }

            // Sort all candidates by score descending
            allCandidates.sort((a, b) => b.score - a.score)

            activeBeams = []
            for (const cand of allCandidates) {
                if (cand.tokens[cand.tokens.length - 1] === G2p.EOS) {
                    completedBeams.push(cand)
                } else {
                    activeBeams.push(cand)
                }

                if (activeBeams.length + completedBeams.length >= beams) {
                    break
                }
            }

            if (completedBeams.length >= beams) {
                break
            }
        }

        // Fallback if not enough completed beams
        completedBeams.push(...activeBeams)
        completedBeams.sort((a, b) => b.score - a.score)
        console.log(completedBeams)
        return completedBeams.slice(0, beams)
            .map(({ tokens, score }) => this.decode(tokens))
    }
}

