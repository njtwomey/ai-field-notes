import { useMemo, useState } from 'react'
import { Interactive, ParamSlider, ParamSwitch, Readout, XYChart, formatNumber } from '@/components/viz'
import { rng } from '@/lib/math'
import { covariance, eigSymmetric } from '../../_shared/linalg'

const N = 200
const D = 10
const FACTORS = 3

/** Fixed loadings: feature m depends on the three latent factors with weights drawn once. */
const LOADINGS = (() => {
  const r = rng(12)
  return Array.from({ length: D }, () => Array.from({ length: FACTORS }, () => r.normal()))
})()

/** n points x = W z + noise·ε in 10 dimensions, with three latent factors z. */
function sample(noise: number, bigUnits: boolean): number[][] {
  const r = rng(3)
  return Array.from({ length: N }, () => {
    const z = Array.from({ length: FACTORS }, () => r.normal())
    return LOADINGS.map((w, m) => {
      const x = w.reduce((s, wk, k) => s + wk * z[k], 0) + noise * r.normal()
      // Feature 10 recorded in units 20 times smaller, so its values are 20 times larger.
      return bigUnits && m === D - 1 ? 20 * x : x
    })
  })
}

function standardise(x: number[][]): number[][] {
  const mean = Array.from({ length: D }, (_, m) => x.reduce((s, r) => s + r[m], 0) / x.length)
  const sd = Array.from({ length: D }, (_, m) => Math.sqrt(x.reduce((s, r) => s + (r[m] - mean[m]) ** 2, 0) / x.length))
  return x.map((r) => r.map((v, m) => (v - mean[m]) / sd[m]))
}

export function ScreePlot() {
  const [noise, setNoise] = useState(0.6)
  const [bigUnits, setBigUnits] = useState(false)
  const [scale, setScale] = useState(false)
  const values = useMemo(() => {
    const x = sample(noise, bigUnits)
    return eigSymmetric(covariance(scale ? standardise(x) : x)).values.map((v) => Math.max(v, 0))
  }, [noise, bigUnits, scale])
  const total = values.reduce((a, b) => a + b, 0)
  const ratio = values.map((v) => v / total)
  const cumulative = ratio.map((_, j) => ratio.slice(0, j + 1).reduce((a, b) => a + b, 0))
  const needed = cumulative.findIndex((c) => c >= 0.9) + 1
  const ks = values.map((_, j) => j + 1)

  return (
    <Interactive
      title="Scree plot"
      caption="Ten features generated from three latent factors plus independent noise. The bars are each component's share of the total variance; the line is the cumulative share. With little noise, three components hold almost everything and the bars drop sharply after the third. Record feature 10 in units 20 times smaller and the first component becomes that feature alone, until the features are standardised."
      controls={
        <>
          <ParamSlider
            label="noise standard deviation"
            value={noise}
            onChange={setNoise}
            min={0.05}
            max={3}
            step={0.05}
          />
          <ParamSwitch label="feature 10 in units 20× smaller" checked={bigUnits} onChange={setBigUnits} />
          <ParamSwitch label="standardise features" checked={scale} onChange={setScale} />
        </>
      }
      readout={
        <>
          <Readout label="first three components explain" value={`${(100 * cumulative[2]).toFixed(1)}%`} />
          <Readout label="components for 90%" value={needed} />
          <Readout label="largest eigenvalue" value={formatNumber(values[0])} />
        </>
      }
    >
      <XYChart
        height={320}
        xLabel="component"
        yLabel="share of variance"
        yRange={[0, 1]}
        series={[
          { name: 'explained variance ratio', type: 'bar', x: ks, y: ratio, slot: 0 },
          { name: 'cumulative', type: 'line', x: ks, y: cumulative, slot: 1 },
        ]}
      />
    </Interactive>
  )
}
