import { useMemo, useState } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, type XYSeries } from 'aifn-render'

const TF = Array.from({ length: 81 }, (_, i) => i / 4)
const LENGTHS = [
  { ratio: 0.5, name: 'half the average length' },
  { ratio: 1, name: 'average length' },
  { ratio: 2, name: 'twice the average length' },
]

/** BM25's term-frequency factor tf(k1 + 1) / (tf + k1(1 − b + b·L/avgL)) against tf, for three document lengths. */
export function Bm25Saturation() {
  const [k1, setK1] = useState(1.2)
  const [b, setB] = useState(0.75)

  const series = useMemo<XYSeries[]>(() => {
    const out: XYSeries[] = [{ name: 'raw tf', type: 'line', x: TF, y: TF, muted: true, dashed: true }]
    LENGTHS.forEach(({ ratio, name }, slot) => {
      const K = k1 * (1 - b + b * ratio)
      out.push({ name, type: 'line', x: TF, y: TF.map((t) => (t === 0 ? 0 : (t * (k1 + 1)) / (t + K))), slot })
    })
    return out
  }, [k1, b])

  // At average length the normaliser 1 − b + b·1 is 1, so K = k1.
  const K = k1
  return (
    <Interactive
      title="BM25 saturates term frequency"
      caption="The weight BM25 gives to a term's frequency tf in a document, before multiplying by the term's IDF. Raw tf (dashed) grows without limit; the BM25 factor rises steeply for the first few occurrences and levels off at k₁ + 1. k₁ sets how quickly it saturates; b sets how much a long document is penalised for its length. With b = 0 the three curves coincide."
      controls={
        <>
          <ParamSlider label="saturation k₁" value={k1} onChange={setK1} min={0} max={3} step={0.05} />
          <ParamSlider label="length normalisation b" value={b} onChange={setB} min={0} max={1} step={0.05} />
        </>
      }
      readout={
        <>
          <Readout label="asymptote k₁ + 1" value={formatNumber(k1 + 1)} />
          <Readout label="factor at tf = 1, average length" value={formatNumber((k1 + 1) / (1 + K))} />
          <Readout label="factor at tf = 10, average length" value={formatNumber((10 * (k1 + 1)) / (10 + K))} />
        </>
      }
    >
      <XYChart
        series={series}
        xLabel="term frequency tf"
        yLabel="tf factor"
        xRange={[0, 20]}
        yRange={[0, 4.2]}
        height={300}
      />
    </Interactive>
  )
}
