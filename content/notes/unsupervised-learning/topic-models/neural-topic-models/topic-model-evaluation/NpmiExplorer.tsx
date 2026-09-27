import { useMemo, useState } from 'react'
import { MathText } from '@/components/content/MathText'
import {
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'

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
  const [na, setNa] = useState(100)
  const [nb, setNb] = useState(50)
  const joint = useParam(20, { min: 0, max: 500, step: 1 })
  const hi = Math.min(na, nb)
  // Keep the joint count feasible when a marginal count shrinks below it.
  const setMarginal = (set: (v: number) => void, other: number) => (v: number) => {
    set(v)
    if (joint.value > Math.min(v, other)) joint.set(Math.min(v, other))
  }
  const nab = Math.min(joint.value, hi)
  const independent = (na * nb) / N

  const curve = useMemo((): XYSeries[] => {
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
  const series: XYSeries[] = [...curve, { name: 'current pair', type: 'scatter', x: [nab], y: [npmi], emphasis: true }]
  const handles: Handle[] = [{ kind: 'x', at: nab, label: 'co-occurrences', onDrag: (x) => joint.set(Math.min(x, hi)) }]

  return (
    <Interactive
      title="Normalised pointwise mutual information"
      caption={
        <MathText
          text={`Two top words of a topic, $a$ and $b$, occur in $n_a$ and $n_b$ of $${N}$ sliding windows of a reference corpus and together in $n_{ab}$. PMI compares $n_{ab}$ with the $n_a n_b / ${N}$ co-occurrences expected under independence. NPMI divides PMI by $-\\log p(a, b)$, which pins it to $[-1, 1]$: $0$ at independence, $1$ when the words only ever occur together. Drag the vertical line to change $n_{ab}$.`}
        />
      }
      controls={
        <>
          <ParamSlider
            label="windows containing a"
            value={na}
            onChange={setMarginal(setNa, nb)}
            min={5}
            max={500}
            step={5}
          />
          <ParamSlider
            label="windows containing b"
            value={nb}
            onChange={setMarginal(setNb, na)}
            min={5}
            max={500}
            step={5}
          />
          <ParamSlider label="windows containing both" param={joint} />
        </>
      }
      readout={
        <>
          <Readout label="expected under independence" value={formatNumber(independent)} />
          <Readout label="PMI (nats)" value={Number.isFinite(pmi) ? formatNumber(pmi) : '−∞'} />
          <Readout label="NPMI" value={formatNumber(npmi)} />
        </>
      }
    >
      <XYChart
        series={series}
        handles={handles}
        xRange={[0, hi]}
        yRange={Y_RANGE}
        xLabel="windows containing both words, n_ab"
        yLabel="NPMI"
        height={280}
      />
    </Interactive>
  )
}
