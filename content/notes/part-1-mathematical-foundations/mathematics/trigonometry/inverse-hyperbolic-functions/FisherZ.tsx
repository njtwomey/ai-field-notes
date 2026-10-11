import { useMemo } from 'react'
import {
  Curve,
  Figure,
  formatNumber,
  Histogram,
  int,
  Plot,
  Plots,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { normal, stream } from 'aifn-compute/foundation/random'
import { normalPdf } from 'aifn-compute/numerics/special'
import { grid } from '../_shared/geometry'

const REPEATS = 2000

/** Sample mean, standard deviation and skewness. */
function moments(v: number[]) {
  const n = v.length
  const mean = v.reduce((a, b) => a + b, 0) / n
  const m2 = v.reduce((a, b) => a + (b - mean) ** 2, 0) / n
  const m3 = v.reduce((a, b) => a + (b - mean) ** 3, 0) / n
  return { mean, sd: Math.sqrt(m2), skew: m3 / m2 ** 1.5 }
}

/**
 * The sampling distribution of Pearson's r for bivariate normal data, before and after Fisher's transformation
 * z = artanh r. The histogram of r is skewed and its spread depends on ρ; the histogram of z is close to the normal
 * curve with mean artanh ρ and variance 1/(n − 3) whatever ρ is.
 */
export function FisherZ() {
  const state = useFigureState({
    rho: slider(-0.95, 0.95, 0.8, { step: 0.05, label: 'population correlation ρ' }),
    n: int(20, { min: 5, max: 100, step: 1, label: 'sample size n' }),
    seed: int(1, { ge: 0, label: 'seed' }),
  })
  const { rho, n, seed } = state

  const sim = useMemo(() => {
    const s = stream(seed)
    const c = Math.sqrt(1 - rho * rho)
    const r: number[] = []
    for (let k = 0; k < REPEATS; k++) {
      let sx = 0
      let sy = 0
      let sxx = 0
      let syy = 0
      let sxy = 0
      for (let i = 0; i < n; i++) {
        const u = normal(s)
        const x = u
        const y = rho * u + c * normal(s)
        sx += x
        sy += y
        sxx += x * x
        syy += y * y
        sxy += x * y
      }
      const cov = sxy - (sx * sy) / n
      const vx = sxx - (sx * sx) / n
      const vy = syy - (sy * sy) / n
      r.push(Math.max(-0.999999, Math.min(0.999999, cov / Math.sqrt(vx * vy))))
    }
    const z = r.map(Math.atanh)
    return { r, z, mr: moments(r), mz: moments(z) }
  }, [rho, n, seed])

  const centre = Math.atanh(rho)
  const se = 1 / Math.sqrt(n - 3)
  // Five standard errors either side of the centre: the z histogram fills the panel at every n.
  const span = 5 * se
  const curve = useMemo(() => {
    const zs = grid(centre - span, centre + span, 241)
    return { x: zs, y: zs.map((z) => normalPdf((z - centre) / se) / se) }
  }, [centre, se, span])

  const rx = useAxis({ label: 'sample correlation r', range: [-1, 1] })
  const ry = useAxis({ label: 'density', range: [0, undefined] })
  const zx = useAxis({ label: 'z = artanh r', range: [centre - span, centre + span] })
  const zy = useAxis({ label: 'density', range: [0, undefined] })
  return (
    <Figure
      title="Fisher's z-transformation"
      state={state}
      caption="Each panel summarises 2,000 samples of n pairs from a bivariate normal distribution with correlation ρ. Left: the sample correlations r pile up against the bound at ±1 and are skewed away from it when ρ is large, and their spread shrinks as |ρ| grows. Right: the same samples after z = artanh r. The histogram is nearly symmetric and matches the normal curve with mean artanh ρ and standard deviation 1/√(n − 3) for every ρ, which is what makes confidence intervals for a correlation simple."
      readouts={{
        r: (
          <>
            <Readout label="sd of r" value={formatNumber(sim.mr.sd)} />
            <Readout label="(1 − ρ²)/√n" value={formatNumber((1 - rho * rho) / Math.sqrt(n))} />
            <Readout label="skewness of r" value={formatNumber(sim.mr.skew)} />
          </>
        ),
        'z = artanh r': (
          <>
            <Readout label="mean of z" value={formatNumber(sim.mz.mean)} />
            <Readout label="artanh ρ" value={formatNumber(centre)} />
            <Readout label="sd of z" value={formatNumber(sim.mz.sd)} />
            <Readout label="1/√(n − 3)" value={formatNumber(se)} />
            <Readout label="skewness of z" value={formatNumber(sim.mz.skew)} />
          </>
        ),
      }}
    >
      <Plots cols={2}>
        <Plot x={rx} y={ry} title="r">
          <Histogram name="sample r" values={sim.r} range={[-1, 1]} bins={60} slot={0} />
        </Plot>
        <Plot x={zx} y={zy} title="z = artanh r">
          <Histogram name="artanh r" values={sim.z} bins={50} slot={1} />
          <Curve name="normal, sd 1/√(n − 3)" x={curve.x} y={curve.y} emphasis />
        </Plot>
      </Plots>
    </Figure>
  )
}
