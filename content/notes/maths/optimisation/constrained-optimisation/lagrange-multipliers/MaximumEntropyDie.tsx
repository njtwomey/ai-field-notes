import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'
import { SensitivityPanel } from './ConstrainedExplorer'

const FACES = [1, 2, 3, 4, 5, 6]

/** The Gibbs distribution pₖ ∝ exp(−λ₁k) on the faces, with its normaliser Z. */
function gibbs(l1: number) {
  const w = FACES.map((k) => Math.exp(-l1 * k))
  const z = w.reduce((a, b) => a + b, 0)
  return { p: w.map((v) => v / z), z }
}
const meanOf = (p: number[]) => p.reduce((s, v, i) => s + v * FACES[i], 0)

/** The multiplier λ₁ whose Gibbs distribution has the target mean. The mean falls as λ₁ rises, so bisect. */
function multiplierFor(mu: number): number {
  let lo = -30
  let hi = 30
  for (let i = 0; i < 80; i++) {
    const mid = (lo + hi) / 2
    if (meanOf(gibbs(mid).p) > mu) lo = mid
    else hi = mid
  }
  return (lo + hi) / 2
}

/** The maximum entropy H*(μ) = log Z + λ₁μ, in nats. */
const maxEntropy = (mu: number) => {
  const l1 = multiplierFor(mu)
  return Math.log(gibbs(l1).z) + l1 * mu
}

export function MaximumEntropyDie() {
  const mu = useParam(4.5, { min: 1.2, max: 5.8, step: 0.05 })
  const l1 = multiplierFor(mu.value)
  const { p, z } = gibbs(l1)
  const entropy = Math.log(z) + l1 * mu.value
  const series = useMemo((): XYSeries[] => {
    const probs = gibbs(multiplierFor(mu.value)).p
    return [
      { name: 'maximum-entropy pₖ', type: 'bar', x: FACES, y: probs, slot: 0 },
      { name: 'uniform 1/6', type: 'line', x: [0.5, 6.5], y: [1 / 6, 1 / 6], muted: true, dashed: true },
    ]
  }, [mu.value])
  return (
    <Interactive
      title="Maximum entropy on a die with a known mean"
      caption="Left: the distribution on the faces 1 to 6 with the largest entropy among those with mean μ. It has the form pₖ ∝ exp(−λ₁k), where λ₁ is the multiplier of the mean constraint. Set μ with the slider, or drag the vertical line on the right. Right: the maximum entropy H*(μ); the dashed tangent has slope λ₁. At μ = 3.5 the constraint costs nothing, λ₁ = 0, and the answer is uniform."
      controls={<ParamSlider label="target mean μ" param={mu} />}
      readout={
        <>
          <Readout label="p" value={`(${p.map((v) => v.toFixed(3)).join(', ')})`} />
          <Readout label="λ₁ (mean)" value={formatNumber(l1)} />
          <Readout label="λ₀ = log Z − 1 (normalisation)" value={formatNumber(Math.log(z) - 1)} />
          <Readout label="H* (nats)" value={formatNumber(entropy)} />
          <Readout label="log 6" value={formatNumber(Math.log(6))} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart height={300} series={series} xRange={[0, 7]} yRange={[0, 1]} xLabel="face k" yLabel="pₖ" />
        <SensitivityPanel
          param={mu}
          optimum={maxEntropy}
          multiplier={multiplierFor}
          xLabel="target mean μ"
          symbol="μ"
          yLabel="H*(μ)"
          height={300}
        />
      </div>
    </Interactive>
  )
}
