import { useMemo } from 'react'
import { Bars, Figure, float, formatNumber, Plot, Readout, useAxis, useFigureState } from 'aifn-render'

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
  const state = useFigureState({
    temp: float(1, { min: 0.5, max: 20, step: 0.1, label: 'temperature T' }),
  })
  const p = useMemo(() => softmax(LOGITS, state.temp), [state.temp])
  const series = useMemo(() => [{ name: 'teacher probability', x: CLASSES, y: p, slot: 0 }] as const, [p])
  const entropy = -p.reduce((a, v) => a + (v > 0 ? v * Math.log2(v) : 0), 0)

  const xAxis = useAxis({ label: 'class', range: [-0.5, 9.5] })
  const yAxis = useAxis({ label: 'probability', range: [0, 1] })
  return (
    <Figure
      title="Teacher probabilities at temperature T"
      purpose="Raise the temperature to see the teacher's probabilities on the other classes appear."
      state={state}
      caption="At T = 1 the teacher puts almost all probability on 7, and the student learns little beyond the hard label. Raising T divides every logit by T before the softmax, which reveals that this 7 resembles a 2 and a 1 far more than a 6. The largest class stays the same at every T."

      readouts={
        <>
          <Readout label="p(7)" value={formatNumber(p[7])} />
          <Readout label="p(2)" value={formatNumber(p[2])} />
          <Readout label="p(6)" value={formatNumber(p[6])} />
          <Readout label="entropy" value={`${formatNumber(entropy)} bits`} />
          <Readout label="soft-target gradient scale 1/T²" value={formatNumber(1 / state.temp ** 2)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={280}>
        <Bars {...series[0]} />
      </Plot>
    </Figure>
  )
}
