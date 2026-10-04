import { useMemo } from 'react'
import {
  Figure,
  formatNumber,
  Handle,
  int,
  MathText,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'

const N = 1000
const Y_RANGE: [number, number] = [-1, 1]

/** PMI and NPMI of a word pair from window counts; NPMI is −1 by convention when the pair never co-occurs. */
function scores(na: number, nb: number, nab: number) {
  if (nab <= 0) return { pmi: -Infinity, npmi: -1 }
  const pab = nab / N
  const pmi = Math.log(pab / ((na / N) * (nb / N)))
  return { pmi, npmi: pmi / -Math.log(pab) }
}

/** NPMI of a word pair as their co-occurrence count varies, for fixed individual counts. */
export function NpmiExplorer() {
  const state = useFigureState({
    na: int(100, { ge: 1, le: N, suggestions: [10, 50, 100, 500], label: 'windows containing a' }),
    nb: int(50, { ge: 1, le: N, suggestions: [10, 50, 100, 500], label: 'windows containing b' }),
    joint: int(20, { min: 0, max: N, step: 1, label: 'windows containing both (at most the smaller count)' }),
  })
  const { na, nb } = state
  const hi = Math.min(na, nb)
  const nab = Math.min(state.joint, hi)
  const independent = (na * nb) / N

  const curve = useMemo((): SeriesSpec[] => {
    const x: number[] = []
    const y: number[] = []
    for (let k = 1; k <= hi; k++) {
      x.push(k)
      y.push(scores(na, nb, k).npmi)
    }
    return [
      { name: 'NPMI', type: 'line', x, y, slot: 0 },
      { name: 'independence', type: 'line', x: [0, hi], y: [0, 0], muted: true, dashed: true },
    ]
  }, [na, nb, hi])

  const { pmi, npmi } = scores(na, nb, nab)
  const series: SeriesSpec[] = [
    ...curve,
    { name: 'current pair', type: 'scatter', x: [nab], y: [npmi], emphasis: true },
  ]

  const xAxis = useAxis({ label: 'windows containing both words, n_ab', range: [0, hi] })
  const yAxis = useAxis({ label: 'NPMI', range: Y_RANGE })
  return (
    <Figure
      title="Normalised pointwise mutual information"
      state={state}
      caption={
        <MathText
          text={`Two top words of a topic, $a$ and $b$, occur in $n_a$ and $n_b$ of $${N}$ sliding windows of a reference corpus and together in $n_{ab}$. PMI compares $n_{ab}$ with the $n_a n_b / ${N}$ co-occurrences expected under independence. NPMI divides PMI by $-\\log p(a, b)$, which pins it to $[-1, 1]$: $0$ at independence, $1$ when the words only ever occur together. Drag the vertical line to change $n_{ab}$.`}
        />
      }
      readouts={
        <>
          <Readout label="expected under independence" value={formatNumber(independent)} />
          <Readout label="PMI (nats)" value={Number.isFinite(pmi) ? formatNumber(pmi) : '−∞'} />
          <Readout label="NPMI" value={formatNumber(npmi)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={280}>
        {seriesLayers(series)}
        <Handle
          kind="x"
          at={nab}
          label="co-occurrences"
          onDrag={(x) => state.set('joint', Math.min(Math.round(x), hi))}
        />
      </Plot>
    </Figure>
  )
}
