import { useMemo } from 'react'
import { Bars, Figure, float, formatNumber, Plot, Readout, slider, useAxis, useFigureState } from 'aifn-render'

const K = 4
const X_RANGE: [number, number] = [0.5, K + 0.5]
const Y_RANGE: [number, number] = [0, 1]

/** Categorical probabilities from logits through a softmax with temperature, with the entropy of the result. */
export function SoftmaxCategorical() {
  const state = useFigureState({
    z1: slider(-4, 4, 2, { step: 0.1, label: 'logit z₁' }),
    z2: slider(-4, 4, 1, { step: 0.1, label: 'logit z₂' }),
    z3: slider(-4, 4, 0, { step: 0.1, label: 'logit z₃' }),
    z4: slider(-4, 4, -1, { step: 0.1, label: 'logit z₄' }),
    temperature: float(1, { min: 0.1, max: 5, step: 0.05, label: 'temperature T' }),
  })

  const { z1, z2, z3, z4 } = state
  const pi = useMemo(() => {
    const scaled = [z1, z2, z3, z4].map((z) => z / state.temperature)
    // Subtracting the largest logit leaves the result unchanged and keeps exp from overflowing.
    const top = Math.max(...scaled)
    const e = scaled.map((z) => Math.exp(z - top))
    const total = e.reduce((a, b) => a + b, 0)
    return e.map((v) => v / total)
  }, [z1, z2, z3, z4, state.temperature])

  const series = useMemo(() => [{ name: 'π_k', x: pi.map((_, k) => k + 1), y: pi }] as const, [pi])
  const entropy = -pi.reduce((s, p) => s + (p > 0 ? p * Math.log(p) : 0), 0)
  const mode = pi.indexOf(Math.max(...pi)) + 1

  const xAxis = useAxis({ label: 'category k', range: X_RANGE })
  const yAxis = useAxis({ label: 'π_k', range: Y_RANGE })
  return (
    <Figure
      title="Softmax to a categorical distribution"
      state={state}
      caption="Each category k has a logit z_k. The softmax π_k = exp(z_k / T) / Σ_j exp(z_j / T) turns the logits into probabilities. Adding the same constant to every logit changes nothing. Low temperature T concentrates the mass on the largest logit; high temperature flattens it towards uniform, where the entropy reaches log K."
      readouts={
        <>
          <Readout label="most likely category" value={mode} />
          <Readout label="entropy (nats)" value={formatNumber(entropy)} />
          <Readout label="maximum, log K" value={formatNumber(Math.log(K))} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={260}>
        <Bars {...series[0]} />
      </Plot>
    </Figure>
  )
}
