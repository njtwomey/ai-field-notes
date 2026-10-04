import { useMemo } from 'react'
import {
  Figure,
  float,
  formatNumber,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'

const TF = Array.from({ length: 81 }, (_, i) => i / 4)
const LENGTHS = [
  { ratio: 0.5, name: 'half the average length' },
  { ratio: 1, name: 'average length' },
  { ratio: 2, name: 'twice the average length' },
]

/** BM25's term-frequency factor tf(k1 + 1) / (tf + k1(1 − b + b·L/avgL)) against tf, for three document lengths. */
export function Bm25Saturation() {
  const state = useFigureState({
    k1: float(1.2, { min: 0, max: 3, step: 0.05, label: 'saturation k₁' }),
    b: float(0.75, { min: 0, max: 1, step: 0.05, label: 'length normalisation b' }),
  })

  const series = useMemo<SeriesSpec[]>(() => {
    const out: SeriesSpec[] = [{ name: 'raw tf', type: 'line', x: TF, y: TF, muted: true, dashed: true }]
    LENGTHS.forEach(({ ratio, name }, slot) => {
      const K = state.k1 * (1 - state.b + state.b * ratio)
      out.push({ name, type: 'line', x: TF, y: TF.map((t) => (t === 0 ? 0 : (t * (state.k1 + 1)) / (t + K))), slot })
    })
    return out
  }, [state.k1, state.b])

  // At average length the normaliser 1 − b + b·1 is 1, so K = k1.
  const K = state.k1
  const xAxis = useAxis({ label: 'term frequency tf', range: [0, 20] })
  const yAxis = useAxis({ label: 'tf factor', range: [0, 4.2] })
  return (
    <Figure
      title="BM25 saturates term frequency"
      state={state}
      caption="The weight BM25 gives to a term's frequency tf in a document, before multiplying by the term's IDF. Raw tf (dashed) grows without limit; the BM25 factor rises steeply for the first few occurrences and levels off at k₁ + 1. k₁ sets how quickly it saturates; b sets how much a long document is penalised for its length. With b = 0 the three curves coincide."

      readouts={
        <>
          <Readout label="asymptote k₁ + 1" value={formatNumber(state.k1 + 1)} />
          <Readout label="factor at tf = 1, average length" value={formatNumber((state.k1 + 1) / (1 + K))} />
          <Readout label="factor at tf = 10, average length" value={formatNumber((10 * (state.k1 + 1)) / (10 + K))} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        {seriesLayers(series)}
      </Plot>
    </Figure>
  )
}
