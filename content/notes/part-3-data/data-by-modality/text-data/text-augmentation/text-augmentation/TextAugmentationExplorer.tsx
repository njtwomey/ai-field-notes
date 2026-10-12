import { useMemo } from 'react'
import { child, stream, uniform } from 'aifn-compute/foundation/random'
import { choice, Curve, Figure, Handle, int, Plot, Readout, slider, useAxis, useFigureState } from 'aifn-render'

const SENTENCES = {
  sentiment: {
    label: 'sentiment review',
    text: 'The documentary delivers an extraordinary narrative with compelling interviews and stunning cinematography.',
    synonyms: {
      extraordinary: 'remarkable',
      narrative: 'story',
      compelling: 'fascinating',
      stunning: 'breathtaking',
      interviews: 'discussions',
      cinematography: 'visuals',
    },
    backTrans:
      'The documentary provides a remarkable story featuring fascinating discussions and breathtaking visuals.',
  },
  news: {
    label: 'economic report',
    text: 'Central banks lowered target interest rates to stimulate business investment and reduce unemployment.',
    synonyms: {
      lowered: 'reduced',
      target: 'benchmark',
      stimulate: 'encourage',
      investment: 'spending',
      reduce: 'lower',
      unemployment: 'joblessness',
    },
    backTrans: 'Central banks reduced benchmark rates to encourage business spending and decrease joblessness.',
  },
  technical: {
    label: 'machine learning statement',
    text: 'Residual connections facilitate gradient propagation across exceptionally deep feedforward neural architectures.',
    synonyms: {
      facilitate: 'enable',
      propagation: 'flow',
      exceptionally: 'remarkably',
      deep: 'layered',
      architectures: 'networks',
    },
    backTrans: 'Skip connections enable gradient flow across remarkably deep feedforward network architectures.',
  },
} as const

type SentenceKey = keyof typeof SENTENCES

export function TextAugmentationExplorer() {
  const state = useFigureState({
    sentenceKey: choice<SentenceKey>(
      [
        { value: 'sentiment', label: 'film review' },
        { value: 'news', label: 'economic report' },
        { value: 'technical', label: 'technical ML statement' },
      ],
      'sentiment',
      { label: 'source text' },
    ),
    technique: choice<'synonym' | 'swap' | 'deletion' | 'backtrans'>(
      [
        { value: 'synonym', label: 'EDA: synonym replacement' },
        { value: 'swap', label: 'EDA: random word swap' },
        { value: 'deletion', label: 'EDA: random word deletion' },
        { value: 'backtrans', label: 'back-translation (pivot)' },
      ],
      'synonym',
      { label: 'augmentation method' },
    ),
    alpha: slider(0.05, 0.4, 0.15, { step: 0.05, label: 'mutation strength α' }),
    seed: int(1, { ge: 1, label: 'seed' }),
  })

  const { sentenceKey, technique, alpha, seed } = state
  const item = SENTENCES[sentenceKey]

  const transformed = useMemo(() => {
    const words = item.text.split(' ')
    if (technique === 'backtrans') {
      return item.backTrans
    }
    if (technique === 'synonym') {
      const synMap = item.synonyms as Record<string, string>
      return words
        .map((w, i) => {
          const clean = w.toLowerCase().replace(/[^\w]/g, '')
          if (synMap[clean] && uniform(child(stream(seed), 'synonym', i)) < alpha * 2.5) {
            return synMap[clean]
          }
          return w
        })
        .join(' ')
    }
    if (technique === 'swap') {
      const copy = [...words]
      const swaps = Math.max(1, Math.round(alpha * copy.length))
      for (let s = 0; s < swaps; s++) {
        const i = (s * 3) % copy.length
        const j = (s * 3 + 1) % copy.length
        const tmp = copy[i]
        copy[i] = copy[j]
        copy[j] = tmp
      }
      return copy.join(' ')
    }
    // Deletion
    return words.filter((_, idx) => idx % Math.round(1 / alpha) !== 0).join(' ')
  }, [item, technique, alpha, seed])

  // Curves across alpha from 0.05 to 0.4
  const tradeOff = useMemo(() => {
    const alphas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    // Semantic fidelity drops with alpha: 1 - 0.9 * alpha
    const fidelity = alphas.map((a) => Math.max(0.4, 1 - 1.2 * a))
    // Lexical novelty rises with alpha: 0.2 + 1.8 * a
    const novelty = alphas.map((a) => Math.min(1.0, 0.15 + 1.9 * a))
    return { alphas, fidelity, novelty }
  }, [])

  const currentFidelity = Math.max(0.4, 1 - 1.2 * alpha)
  const currentNovelty = Math.min(1.0, 0.15 + 1.9 * alpha)

  const xAxis = useAxis({ label: 'mutation strength α', range: [0.05, 0.4] })
  const yAxis = useAxis({ label: 'index score', range: [0, 1.05] })

  return (
    <Figure
      title="Text augmentation trade-off: lexical novelty vs semantic preservation"
      state={state}
      hoverReadout={false}
      caption="Trade-off between lexical diversity (novelty) and semantic label preservation (fidelity) as mutation rate α varies. Small perturbation (α ≈ 0.1 to 0.15) generates diverse linguistic variants without corrupting class labels. Excessive mutation rates (α > 0.3) disrupt grammatical syntax and invert semantic polarity. Drag the mutation handle to adjust the transformation strength."
      readouts={
        <>
          <Readout label="mutation strength α" value={alpha.toFixed(2)} />
          <Readout label="semantic fidelity" value={`${(currentFidelity * 100).toFixed(0)}%`} />
          <Readout label="lexical novelty" value={`${(currentNovelty * 100).toFixed(0)}%`} />
          <Readout label="augmented preview" value={transformed.slice(0, 28) + '...'} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} title="fidelity and diversity across mutation rate">
        <Curve name="semantic fidelity" x={tradeOff.alphas} y={tradeOff.fidelity} slot={0} />
        <Curve name="lexical novelty" x={tradeOff.alphas} y={tradeOff.novelty} slot={1} />
        <Curve name="recommended α=0.15" x={[0.15, 0.15]} y={[0, 1.05]} muted dashed />
        <Handle
          kind="x"
          at={alpha}
          label={`α=${alpha.toFixed(2)}`}
          onDrag={(val) => state.set('alpha', Math.max(0.05, Math.min(0.4, Math.round(val * 20) / 20)))}
        />
      </Plot>
    </Figure>
  )
}
