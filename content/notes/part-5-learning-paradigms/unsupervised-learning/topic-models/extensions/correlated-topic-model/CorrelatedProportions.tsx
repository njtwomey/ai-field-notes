import { useMemo, useState } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber } from 'aifn-render'
import { rng } from '@/lib/math'
import { correlation, dirichlet, softmax } from '../_shared/random'

const K = 5
const SAMPLES = 400

/** Topic proportions for five topics under a symmetric Dirichlet and under a logistic normal. */
export function CorrelatedProportions() {
  const [alpha, setAlpha] = useState(0.5)
  const [rho, setRho] = useState(0.8)
  const [scale, setScale] = useState(1)

  const dir = useMemo(() => {
    const r = rng(5)
    return Array.from({ length: SAMPLES }, () => dirichlet(r, new Array<number>(K).fill(alpha)))
  }, [alpha])
  const logn = useMemo(() => {
    const r = rng(6)
    // η ~ N(0, s²Σ) with unit variances and correlation ρ between topics 1 and 2 only, via a 2 × 2 Cholesky factor.
    return Array.from({ length: SAMPLES }, () => {
      const e = Array.from({ length: K }, () => r.normal())
      const eta = e.map((v) => scale * v)
      eta[1] = scale * (rho * e[0] + Math.sqrt(1 - rho * rho) * e[1])
      return softmax(eta)
    })
  }, [rho, scale])

  const corrDir = correlation(
    dir.map((t) => t[0]),
    dir.map((t) => t[1]),
  )
  const corrLogn = correlation(
    logn.map((t) => t[0]),
    logn.map((t) => t[1]),
  )

  return (
    <Interactive
      title="Topic correlations: Dirichlet against logistic normal"
      caption="Each point is the topic proportions of one simulated document with five topics, plotted as the share of topic 1 against the share of topic 2. Left: a symmetric Dirichlet, as in LDA. Whatever α is, the correlation between two proportions stays at −1/(K − 1) = −0.25: the concentration changes how spread the points are, never how the topics co-occur. Right: the logistic normal of the correlated topic model, with correlation ρ between the log-weights of topics 1 and 2. Positive ρ makes documents that use topic 1 also use topic 2, which the Dirichlet cannot express."
      controls={
        <>
          <ParamSlider label="Dirichlet α" value={alpha} onChange={setAlpha} min={0.1} max={3} step={0.05} />
          <ParamSlider
            label="logistic-normal correlation ρ"
            value={rho}
            onChange={setRho}
            min={-0.95}
            max={0.95}
            step={0.05}
          />
          <ParamSlider
            label="logistic-normal scale s"
            value={scale}
            onChange={setScale}
            min={0.2}
            max={2.5}
            step={0.1}
          />
        </>
      }
      readout={
        <>
          <Readout label="corr(θ₁, θ₂), Dirichlet" value={formatNumber(corrDir)} />
          <Readout label="theory −1/(K − 1)" value="−0.25" />
          <Readout label="corr(θ₁, θ₂), logistic normal" value={formatNumber(corrLogn)} />
        </>
      }
    >
      <div className="grid gap-4 lg:grid-cols-2">
        <XYChart
          height={300}
          xLabel="θ₁ (Dirichlet)"
          yLabel="θ₂"
          xRange={[0, 1]}
          yRange={[0, 1]}
          series={[{ name: 'Dirichlet', type: 'scatter', x: dir.map((t) => t[0]), y: dir.map((t) => t[1]), slot: 0 }]}
        />
        <XYChart
          height={300}
          xLabel="θ₁ (logistic normal)"
          yLabel="θ₂"
          xRange={[0, 1]}
          yRange={[0, 1]}
          series={[
            { name: 'logistic normal', type: 'scatter', x: logn.map((t) => t[0]), y: logn.map((t) => t[1]), slot: 1 },
          ]}
        />
      </div>
    </Interactive>
  )
}
