import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'

/** Illustrative teacher logits for an image of a handwritten 7 that also looks a little like a 1 and a 2. */
const LOGITS = [-2, 3.5, 4, 0.5, -1, -0.5, -3, 10, -1.5, 1]
const CLASSES = LOGITS.map((_, i) => i)

const softmax = (z: number[], t: number) => {
  const m = Math.max(...z)
  const e = z.map((v) => Math.exp((v - m) / t))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / s)
}

export function SoftTargets() {
  const temp = useParam(1, { min: 0.5, max: 20, step: 0.1 })
  const p = useMemo(() => softmax(LOGITS, temp.value), [temp.value])
  const series = useMemo(
    (): XYSeries[] => [{ name: 'teacher probability', type: 'bar', x: CLASSES, y: p, slot: 0 }],
    [p],
  )
  const entropy = -p.reduce((a, v) => a + (v > 0 ? v * Math.log2(v) : 0), 0)

  return (
    <Interactive
      title="Teacher probabilities at temperature T"
      caption="At T = 1 the teacher puts almost all probability on 7, and the student learns little beyond the hard label. Raising T divides every logit by T before the softmax, which reveals that this 7 resembles a 2 and a 1 far more than a 6. The largest class stays the same at every T."
      controls={<ParamSlider label="temperature T" param={temp} />}
      readout={
        <>
          <Readout label="p(7)" value={formatNumber(p[7])} />
          <Readout label="p(2)" value={formatNumber(p[2])} />
          <Readout label="p(6)" value={formatNumber(p[6])} />
          <Readout label="entropy" value={`${formatNumber(entropy)} bits`} />
          <Readout label="soft-target gradient scale 1/T²" value={formatNumber(1 / temp.value ** 2)} />
        </>
      }
    >
      <XYChart series={series} xLabel="class" yLabel="probability" xRange={[-0.5, 9.5]} yRange={[0, 1]} height={280} />
    </Interactive>
  )
}
