import { useMemo, useState } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Player,
  Plot,
  Points,
  Readout,
  Rug,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import {
  design,
  drawSamples,
  kernelRatio,
  kliepTrace,
  logRatio,
  pickCentres,
  ratioErrors,
  type Shift,
} from '../_shared/ratio'

const LO = -4
const HI = 5
const X = toFlat(linspace(LO, HI, 241))

/**
 * KLIEP by projected gradient ascent, scrubbed step by step: the model r̂ = Σ α_l K(·, c_l) with Gaussian kernels on
 * members of the p sample starts flat (every β_l = α_l b_l equal) and moves its weight onto the centres that raise
 * the p sample's log-likelihood, while the q sample's mean of r̂ stays 1.
 */
export function KliepAscent() {
  const state = useFigureState({
    sigma: float(0.6, { min: 0.1, max: 2, step: 0.05, label: 'kernel width σ' }),
    b: int(40, { min: 3, max: 100, label: 'kernel centres b', suggestions: [10, 40, 100] }),
    mean: float(1, { min: -2, max: 3, step: 0.05, label: 'mean of p' }),
    sd: float(0.6, { min: 0.3, max: 1.6, step: 0.05, label: 'standard deviation of p' }),
    n: int(200, { min: 20, max: 1000, label: 'points in each sample', suggestions: [50, 200, 1000] }),
    steps: int(200, { min: 10, max: 2000, label: 'projected gradient steps', suggestions: [50, 200, 1000] }),
    seed: int(1, { min: 0, max: 999, label: 'seed' }),
  })
  const shift: Shift = useMemo(() => ({ mean: state.mean, sd: state.sd }), [state.mean, state.sd])

  const run = useMemo(() => {
    const { xp, xq } = drawSamples(state.seed, state.n, state.n, shift)
    const centres = pickCentres(xp, state.b)
    const tr = kliepTrace(xp, xq, centres, state.sigma, state.steps)
    // K(x, c_l) on the grid, row-major [grid, b], for drawing each weighted bump.
    const K = toFlat(design(X, centres, state.sigma))
    return { xp, xq, centres, tr, K }
  }, [state.seed, state.n, shift, state.b, state.sigma, state.steps])

  // The walk-through restarts at step 0 whenever the run changes.
  const [position, setPosition] = useState({ run, k: 0 })
  const k = position.run === run ? Math.min(position.k, run.tr.alphas.length - 1) : 0
  const alpha = run.tr.alphas[k]

  const fit = useMemo(() => kernelRatio(alpha, run.centres, state.sigma, X), [alpha, run, state.sigma])
  const truth = useMemo(() => X.map((x) => Math.exp(logRatio(x, shift))), [shift])
  const bumps = useMemo(() => {
    const b = run.centres.length
    const x: number[] = []
    const y: number[] = []
    for (let l = 0; l < b; l++) {
      if (alpha[l] <= 1e-9) continue
      X.forEach((xv, i) => {
        x.push(xv)
        y.push(alpha[l] * run.K[i * b + l])
      })
      x.push(NaN)
      y.push(NaN)
    }
    return { x, y }
  }, [alpha, run])

  const active = alpha.filter((a) => a > 1e-9).length
  const errors = ratioErrors(
    X,
    fit.map((v) => Math.log(v)),
    shift,
  )
  const activeCentres = run.centres.filter((_, l) => alpha[l] > 1e-9)

  const xa = useAxis({ label: 'x', range: [LO, HI] })
  const ya = useAxis({ label: 'r(x)', range: [0, undefined] })
  const xb = useAxis({ label: 'step' })
  const yb = useAxis({ label: 'mean log r̂ over the p sample' })
  return (
    <Figure
      title="KLIEP by projected gradient ascent"
      state={state}
      caption="Left: the KLIEP model r̂ = Σ α_l K(x, c_l), with each of its Gaussian bumps α_l K(x, c_l) drawn thin, against the true ratio (dashed). Ticks mark the kernel centres, members of the p sample; filled dots mark those still in use. Right: the objective, the mean of log r̂ over the p sample, which every step raises. Play to watch the ascent. The model starts flat and moves weight towards the centres where p is dense relative to q; the constraint keeps the mean of r̂ over the q sample at exactly 1, so weight added in one place is taken from another. The simplex projection sets many weights to exactly zero."
      controls={
        <Player
          value={k}
          onChange={(v) => setPosition({ run, k: v })}
          count={run.tr.alphas.length}
          label="step"
          format={(v) => String(run.tr.index[v])}
        />
      }
      readouts={
        <>
          <Readout label="step" value={run.tr.index[k]} />
          <Readout label="mean log r̂ over p sample" value={formatNumber(run.tr.objective[k])} />
          <Readout label="centres in use" value={`${active} of ${run.centres.length}`} />
          <Readout label="KL error" value={formatNumber(errors.kl)} />
          <Readout label="squared error" value={formatNumber(errors.squared)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <div className="md:col-span-2">
          <Plot x={xa} y={ya} height={300}>
            <Curve name="α_l K(x, c_l)" x={bumps.x} y={bumps.y} slot={2} thin silent />
            <Curve name="r(x)" x={X} y={truth} emphasis dashed />
            <Curve name="r̂(x)" x={X} y={fit} slot={2} width={3} />
            <Rug name="kernel centres" values={run.centres} muted />
            <Points name="centres in use" x={activeCentres} y={activeCentres.map(() => 0)} slot={2} size={6} />
          </Plot>
        </div>
        <Plot x={xb} y={yb} height={300}>
          <Curve name="objective" x={run.tr.index} y={run.tr.objective} slot={2} />
          <Points name="now" x={[run.tr.index[k]]} y={[run.tr.objective[k]]} emphasis />
        </Plot>
      </div>
    </Figure>
  )
}
