import { useMemo, useState } from 'react'

interface DocumentParagraph {
  id: number
  title: string
  text: string
}

const DOCUMENT: DocumentParagraph[] = [
  {
    id: 1,
    title: 'Supervised Learning & Loss',
    text: 'In supervised machine learning, models minimise empirical risk across labelled examples by computing a scalar loss function that penalises discrepancy with ground truth targets.',
  },
  {
    id: 2,
    title: 'Chain Rule & Backpropagation',
    text: 'Backpropagation computes the exact gradient of the loss with respect to every parameter by recursively applying the chain rule backwards from the output layer to the inputs.',
  },
  {
    id: 3,
    title: 'Stochastic Optimisation',
    text: 'Stochastic gradient descent updates network weights in the opposite direction of the gradient vector, scaling step sizes with a tunable learning rate schedule.',
  },
  {
    id: 4,
    title: 'Generalisation & Overfitting',
    text: 'Regularisation techniques such as weight decay and dropout constrain model capacity, preventing overparameterised networks from memorising idiosyncratic training noise.',
  },
]

interface NegativeCandidate {
  type: 'random' | 'bm25' | 'dense' | 'false_neg'
  label: string
  score: number // cosine similarity with query
  title: string
  snippet: string
  explanation: string
}

const CANDIDATES: NegativeCandidate[] = [
  {
    type: 'random',
    label: 'In-batch Random Negative',
    score: 0.08,
    title: 'Deep Ocean Circulation',
    snippet:
      'Thermohaline circulation is driven by global density gradients created by surface heat and freshwater fluxes.',
    explanation:
      'Completely unrelated topic. Easy for bi-encoders to distinguish; yields near-zero gradient after initial epochs.',
  },
  {
    type: 'bm25',
    label: 'BM25 Lexical Hard Negative',
    score: 0.62,
    title: 'Artistic Colour Gradients',
    snippet: 'Applying multiple translucent layers of paint produces a smooth gradient across oil canvases.',
    explanation: 'Shares lexical tokens ("gradient", "layer") with the query but completely lacks semantic relevance.',
  },
  {
    type: 'dense',
    label: 'Dense Retriever Hard Negative',
    score: 0.82,
    title: 'Forward Pass Matrix Algebra',
    snippet:
      'During the forward inference pass, input activations are multiplied by weight matrices and passed through non-linearities.',
    explanation:
      'Shares deep semantic embedding space with deep learning, but does not actually answer the backpropagation question.',
  },
  {
    type: 'false_neg',
    label: 'False Negative (Unlabelled Relevant)',
    score: 0.89,
    title: 'Reverse-Mode Automatic Differentiation',
    snippet:
      'Reverse-mode automatic differentiation tracks intermediate Jacobians in reverse order, which is identical to backprop.',
    explanation:
      'Actually answers the query! Incomplete ground-truth pooling falsely labels this as negative, penalising the model.',
  },
]

export function RetrievalChunkingAndNegatives() {
  const [activeTab, setActiveTab] = useState<'chunking' | 'negatives'>('chunking')
  const [chunkWords, setChunkWords] = useState<number>(30)
  const [overlapPct, setOverlapPct] = useState<number>(20)
  const [tau, setTau] = useState<number>(0.05)
  const [includedNegatives, setIncludedNegatives] = useState<Record<string, boolean>>({
    random: true,
    bm25: true,
    dense: true,
    false_neg: false,
  })

  // Chunking computation
  const allWords = useMemo(() => {
    return DOCUMENT.flatMap((d) => d.text.split(' '))
  }, [])

  const chunks = useMemo(() => {
    const list: { id: number; words: string[]; startWord: number; endWord: number }[] = []
    const step = Math.max(5, Math.floor(chunkWords * (1 - overlapPct / 100)))

    for (let i = 0; i < allWords.length; i += step) {
      const slice = allWords.slice(i, i + chunkWords)
      if (slice.length > 0) {
        list.push({
          id: list.length + 1,
          words: slice,
          startWord: i,
          endWord: Math.min(allWords.length, i + chunkWords),
        })
      }
      if (i + chunkWords >= allWords.length) break
    }
    return list
  }, [allWords, chunkWords, overlapPct])

  // Contrastive InfoNCE Loss calculation
  const posScore = 0.94
  const lossMetrics = useMemo(() => {
    const activeCandidates = CANDIDATES.filter((c) => includedNegatives[c.type])
    const posExp = Math.exp(posScore / tau)
    const negExps = activeCandidates.map((c) => Math.exp(c.score / tau))
    const sumNeg = negExps.reduce((a, b) => a + b, 0)
    const prob = posExp / (posExp + sumNeg)
    const loss = -Math.log(Math.max(1e-12, prob))

    return {
      activeCandidates,
      prob,
      loss,
    }
  }, [posScore, tau, includedNegatives])

  return (
    <div className="my-6 rounded-lg border border-border bg-card p-4 text-card-foreground shadow-sm">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3 border-b border-border pb-3">
        <div>
          <h4 className="text-text font-semibold">Retrieval Corpus Explorer: Chunking & Hard Negatives</h4>
          <p className="text-text-muted text-xs">
            Inspect passage boundary chunking windows and contrastive InfoNCE negative mining dynamics.
          </p>
        </div>
        <div className="flex gap-2 text-xs">
          <button
            onClick={() => setActiveTab('chunking')}
            className={`rounded px-3 py-1 font-medium transition-colors ${
              activeTab === 'chunking'
                ? 'bg-primary text-primary-foreground'
                : 'bg-surface hover:bg-surface-elevated text-text'
            }`}
          >
            Passage Chunking
          </button>
          <button
            onClick={() => setActiveTab('negatives')}
            className={`rounded px-3 py-1 font-medium transition-colors ${
              activeTab === 'negatives'
                ? 'bg-primary text-primary-foreground'
                : 'bg-surface hover:bg-surface-elevated text-text'
            }`}
          >
            Hard Negative Mining
          </button>
        </div>
      </div>

      {activeTab === 'chunking' && (
        <div className="space-y-4">
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <div className="bg-surface/50 rounded border border-border p-2.5">
              <div className="text-text mb-1 flex items-center justify-between text-xs font-semibold">
                <span>Chunk Size:</span>
                <span className="font-mono text-primary">{chunkWords} words</span>
              </div>
              <input
                type="range"
                min={15}
                max={50}
                step={5}
                value={chunkWords}
                onChange={(e) => setChunkWords(Number(e.target.value))}
                className="w-full accent-primary"
              />
            </div>

            <div className="bg-surface/50 rounded border border-border p-2.5">
              <div className="text-text mb-1 flex items-center justify-between text-xs font-semibold">
                <span>Overlap:</span>
                <span className="font-mono text-primary">{overlapPct}%</span>
              </div>
              <input
                type="range"
                min={0}
                max={50}
                step={10}
                value={overlapPct}
                onChange={(e) => setOverlapPct(Number(e.target.value))}
                className="w-full accent-primary"
              />
            </div>
          </div>

          <div className="bg-surface rounded border border-border p-3">
            <div className="text-text-muted mb-2 flex items-center justify-between text-xs">
              <span className="font-semibold tracking-wider uppercase">
                Generated Passage Chunks ({chunks.length} total)
              </span>
              <span>Total document: {allWords.length} words</span>
            </div>
            <div className="space-y-2.5">
              {chunks.map((ch) => (
                <div key={ch.id} className="bg-surface-elevated rounded border border-border/80 p-2 text-xs">
                  <div className="text-text-muted mb-1 flex items-center justify-between font-mono text-[11px]">
                    <span className="font-bold text-primary">Passage #{ch.id}</span>
                    <span>
                      Words [{ch.startWord}:{ch.endWord}] ({ch.words.length} words)
                    </span>
                  </div>
                  <p className="text-text font-serif leading-relaxed">&ldquo;{ch.words.join(' ')}&rdquo;</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {activeTab === 'negatives' && (
        <div className="space-y-4">
          <div className="bg-surface rounded border border-border p-3">
            <div className="text-text-muted mb-1 text-xs font-semibold uppercase">Target Search Query</div>
            <p className="text-text text-sm font-medium">
              &ldquo;How does backpropagation compute gradients through neural layers?&rdquo;
            </p>
            <div className="mt-2 flex items-center gap-2 font-mono text-xs">
              <span className="rounded bg-emerald-500/20 px-2 py-0.5 font-bold text-emerald-800 dark:text-emerald-300">
                Ground Truth Positive ($d^+$): Sim = {posScore.toFixed(2)}
              </span>
              <span className="text-text-muted">Chain rule gradient backprop passage</span>
            </div>
          </div>

          {/* Negative selection checkboxes */}
          <div className="space-y-2">
            <span className="text-text-muted text-xs font-semibold uppercase">Negative Candidate Pool</span>
            {CANDIDATES.map((cand) => {
              const isChecked = includedNegatives[cand.type]
              return (
                <div
                  key={cand.type}
                  className={`rounded border p-2.5 transition-colors ${
                    isChecked ? 'bg-surface-elevated border-primary/50' : 'bg-surface/40 border-border opacity-70'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <label className="text-text flex cursor-pointer items-center gap-2 text-xs font-semibold">
                      <input
                        type="checkbox"
                        checked={isChecked}
                        onChange={(e) => setIncludedNegatives((prev) => ({ ...prev, [cand.type]: e.target.checked }))}
                        className="h-3.5 w-3.5 rounded accent-primary"
                      />
                      <span>{cand.label}</span>
                    </label>
                    <span className="text-text-muted font-mono text-xs font-bold">
                      Sim $s(q, d^-) = {cand.score.toFixed(2)}$
                    </span>
                  </div>
                  <p className="text-text mt-1 text-xs italic">&ldquo;{cand.snippet}&rdquo;</p>
                  <p className="text-text-muted mt-0.5 text-[11px]">{cand.explanation}</p>
                </div>
              )
            })}
          </div>

          {/* Loss Readouts */}
          <div className="bg-surface/60 rounded border border-border p-3">
            <div className="mb-2 flex items-center justify-between text-xs">
              <span className="text-text font-semibold">Contrastive InfoNCE Loss Dynamics</span>
              <div className="flex items-center gap-2">
                <span className="text-text-muted">Temperature $\tau$:</span>
                <input
                  type="range"
                  min={0.01}
                  max={0.2}
                  step={0.01}
                  value={tau}
                  onChange={(e) => setTau(Number(e.target.value))}
                  className="w-24 accent-primary"
                />
                <span className="font-mono font-bold text-primary">{tau.toFixed(2)}</span>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3 text-xs sm:grid-cols-3">
              <div className="bg-surface rounded border border-border p-2">
                <span className="text-text-muted block">Negatives Count</span>
                <span className="text-text font-mono text-base font-bold">{lossMetrics.activeCandidates.length}</span>
              </div>
              <div className="bg-surface rounded border border-border p-2">
                <span className="text-text-muted block">Positive Softmax Prob</span>
                <span className="font-mono text-base font-bold text-emerald-600 dark:text-emerald-400">
                  {(lossMetrics.prob * 100).toFixed(1)}%
                </span>
              </div>
              <div className="bg-surface rounded border border-border p-2">
                <span className="text-text-muted block">InfoNCE Loss</span>
                <span className="font-mono text-base font-bold text-primary">{lossMetrics.loss.toFixed(3)}</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
