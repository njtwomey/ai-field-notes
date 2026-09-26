import { useMemo, useState } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, type XYSeries } from '@/components/viz'

const K = 4
const X_RANGE: [number, number] = [0.5, K + 0.5]
const Y_RANGE: [number, number] = [0, 1]

/** Categorical probabilities from logits through a softmax with temperature, with the entropy of the result. */
export function SoftmaxCategorical() {
  const [logits, setLogits] = useState([2, 1, 0, -1])
  const [temperature, setTemperature] = useState(1)

  const pi = useMemo(() => {
    const scaled = logits.map((z) => z / temperature)
    // Subtracting the largest logit leaves the result unchanged and keeps exp from overflowing.
    const top = Math.max(...scaled)
    const e = scaled.map((z) => Math.exp(z - top))
    const total = e.reduce((a, b) => a + b, 0)
    return e.map((v) => v / total)
  }, [logits, temperature])

  const series = useMemo((): XYSeries[] => [{ name: 'π_k', type: 'bar', x: pi.map((_, k) => k + 1), y: pi }], [pi])
  const entropy = -pi.reduce((s, p) => s + (p > 0 ? p * Math.log(p) : 0), 0)
  const mode = pi.indexOf(Math.max(...pi)) + 1

  return (
    <Interactive
      title="Softmax to a categorical distribution"
      caption="Each category k has a logit z_k. The softmax π_k = exp(z_k / T) / Σ_j exp(z_j / T) turns the logits into probabilities. Adding the same constant to every logit changes nothing. Low temperature T concentrates the mass on the largest logit; high temperature flattens it towards uniform, where the entropy reaches log K."
      controls={
        <>
          {logits.map((z, k) => (
            <ParamSlider
              key={k}
              label={`logit z${'₁₂₃₄'[k]}`}
              value={z}
              onChange={(v) => setLogits((prev) => prev.map((old, j) => (j === k ? v : old)))}
              min={-4}
              max={4}
              step={0.1}
            />
          ))}
          <ParamSlider
            label="temperature T"
            value={temperature}
            onChange={setTemperature}
            min={0.1}
            max={5}
            step={0.05}
          />
        </>
      }
      readout={
        <>
          <Readout label="most likely category" value={mode} />
          <Readout label="entropy (nats)" value={formatNumber(entropy)} />
          <Readout label="maximum, log K" value={formatNumber(Math.log(K))} />
        </>
      }
    >
      <XYChart height={260} series={series} xLabel="category k" yLabel="π_k" xRange={X_RANGE} yRange={Y_RANGE} />
    </Interactive>
  )
}
