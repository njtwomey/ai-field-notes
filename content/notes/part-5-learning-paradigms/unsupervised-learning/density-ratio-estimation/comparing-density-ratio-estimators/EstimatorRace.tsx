import { useMemo, useState } from 'react'
import {
  choice,
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
  useComputed,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { logRatio, type Shift } from '../_shared/ratio'
import { METHODS, raceFrames, SIZES } from './race'

const LO = -4
const HI = 5
const X = toFlat(linspace(LO, HI, 241))
const ERROR_RANGE: [number, number] = [1e-3, 1e2]
const LOG_R_RANGE: [number, number] = [-8, 4]
const clip = (v: number, [lo, hi]: [number, number]) => Math.min(hi, Math.max(lo, v))

type Metric = 'squared' | 'kl'

/**
 * Four estimators of the same ratio from nested samples of growing size, scrubbed by sample size: dividing kernel
 * density estimates, logistic regression on Gaussian bumps, KLIEP and uLSIF, the last three with the same kernel width.
 */
export function EstimatorRace() {
  const state = useFigureState({
    metric: choice<Metric>(
      [
        { value: 'squared', label: 'squared error ½ E_q[(r̂ − r)²]' },
        { value: 'kl', label: 'KL error, from p to r̂ q' },
      ],
      'squared',
      { label: 'error' },
    ),
    mean: float(1, { min: -2, max: 3, step: 0.05, label: 'mean of p' }),
    sd: float(0.6, { min: 0.3, max: 1.6, step: 0.05, label: 'standard deviation of p' }),
    sigma: float(0.5, { min: 0.15, max: 1.5, step: 0.05, label: 'kernel width σ (shared)' }),
    seed: int(1, { min: 0, max: 999, label: 'seed' }),
  })
  const shift: Shift = useMemo(() => ({ mean: state.mean, sd: state.sd }), [state.mean, state.sd])

  const race = useComputed(() => raceFrames(state.seed, shift, state.sigma, X), [state.seed, shift, state.sigma], {
    mode: 'release',
  })
  const frames = race.value

  // The walk-through restarts at the smallest sample whenever the race is recomputed.
  const [position, setPosition] = useState({ frames, k: 0 })
  const k = position.frames === frames ? position.k : 0
  const frame = frames[k]

  const truth = useMemo(() => X.map((x) => logRatio(x, shift)), [shift])
  // Each estimator's error at every size, clipped to the chart.
  const curves = METHODS.map((_, m) => frames.map((f) => clip(f.errors[m][state.metric], ERROR_RANGE)))

  const xa = useAxis({ label: 'x', range: [LO, HI] })
  const ya = useAxis({ label: 'log r(x)', range: LOG_R_RANGE })
  const xb = useAxis({ label: 'points in each sample', log: true, range: [SIZES[0], SIZES.at(-1)!] })
  const yb = useAxis({ label: 'error', log: true, range: ERROR_RANGE })
  return (
    <Figure
      title="Four estimators as the samples grow"
      state={state}
      caption="Left: each estimator's log r̂ at the current sample size against the true log ratio (dashed), with the two samples as rugs. Right: each estimator's error against the size of each sample, on log scales, with the current size marked; errors beyond the chart sit on its edge. Play to grow both samples from 10 to 1000 points; each larger sample contains the smaller ones. Dividing kernel density estimates fails in the tails while the samples are small. uLSIF, whose penalty is cross-validated, is steady from the start; KLIEP and logistic regression, unpenalised or barely penalised, overfit small samples. Switch the error between the squared error, which weights mistakes by q, and the Kullback–Leibler error, which weights the log ratio by p."
      controls={
        <Player
          value={k}
          onChange={(v) => setPosition({ frames, k: v })}
          count={frames.length}
          label="sample size"
          format={(v) => String(frames[v].n)}
        />
      }
      readouts={
        <>
          <Readout label="points in each sample" value={frame.n} />
          {METHODS.map((name, m) => (
            <Readout key={name} label={name} value={formatNumber(frame.errors[m][state.metric])} />
          ))}
          <Readout label="uLSIF's cross-validated λ" value={formatNumber(frame.lambda)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Plot x={xa} y={ya} height={320}>
          <Curve name="log r(x)" x={X} y={truth} emphasis dashed />
          {METHODS.map((name, m) => (
            <Curve
              key={name}
              name={name}
              x={X}
              y={frame.logR[m].map((v) => clip(v, [LOG_R_RANGE[0] - 2, LOG_R_RANGE[1] + 2]))}
              slot={m}
              stale={race.stale}
            />
          ))}
          <Rug name="q sample" values={frame.xq} muted />
          <Rug name="p sample" values={frame.xp} emphasis />
        </Plot>
        <Plot x={xb} y={yb} height={320}>
          {METHODS.map((name, m) => (
            <Curve key={name} name={name} x={SIZES} y={curves[m]} slot={m} stale={race.stale} />
          ))}
          {METHODS.map((name, m) => (
            <Points key={name} name={name} x={[frame.n]} y={[curves[m][k]]} slot={m} size={8} />
          ))}
        </Plot>
      </div>
    </Figure>
  )
}
