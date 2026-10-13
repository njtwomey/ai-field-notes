import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Readout,
  toggle,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { normals, stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { jamesStein, jamesSteinExactRisk, jamesSteinRisk } from 'aifn-methods/theory/shrinkage'

const TARGETS = [
  { value: 'zero', label: 'the origin (d − 2)' },
  { value: 'mean', label: 'the grand mean (d − 3)' },
] as const
const MAX_DISTANCE = 8
const NORMS = Array.from({ length: 33 }, (_, i) => (i * MAX_DISTANCE) / 32)
const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—')

/** A fixed direction for the means: an evenly spread ramp with mean zero and unit length. */
const direction = (d: number) => {
  const ramp = Array.from({ length: d }, (_, j) => j - (d - 1) / 2)
  const length = Math.sqrt(ramp.reduce((a, v) => a + v * v, 0))
  return ramp.map((v) => v / length)
}

/**
 * Stein's paradox on the engine: the risk of the observations, the James–Stein estimator (exact and simulated) and
 * its positive part against the distance of the means from the target, and one draw at a chosen distance.
 */
export function SteinRisk() {
  const state = useFigureState({
    d: int(10, { ge: 4, le: 100, suggestions: [4, 10, 50], label: 'dimension d' }),
    target: choice(TARGETS, 'zero', { label: 'shrink towards' }),
    positive: toggle(true, { label: 'positive part in the draw' }),
    seed: int(1, { ge: 1, le: 999, label: 'draw' }),
    distance: float(2, { min: 0, max: MAX_DISTANCE, step: 0.25, label: 'distance ‖θ − target‖ / σ' }),
  })
  const { d, target, positive, seed, distance } = state
  const risk = useComputed(
    () => jamesSteinRisk(stream('stein-risk'), { p: d, norms: NORMS, repeats: 2000, target }),
    [d, target],
    { mode: 'release' },
  )
  const r = risk.value
  const draw = useMemo(() => {
    const v = direction(d)
    const theta = v.map((u) => distance * u)
    const noise = toFlat(normals(stream(`stein-draw/${seed}`), [d]))
    const x = theta.map((t, j) => t + noise[j])
    const js = jamesStein(x, { target, positivePart: positive })
    const loss = (e: ArrayLike<number>) => theta.reduce((a, t, j) => a + (e[j] - t) ** 2, 0)
    return { theta, x, js, lossX: loss(x), lossJs: loss(js.estimate) }
  }, [d, distance, seed, target, positive])
  const exact = jamesSteinExactRisk(d, distance, { target: target === 'mean' ? 'mean' : 'fixed' })
  const index = Array.from({ length: d }, (_, j) => j + 1)

  const rx = useAxis({
    label: 'distance of the means from the target ‖θ − t‖ / σ',
    range: [0, MAX_DISTANCE],
    nice: false,
  })
  const ry = useAxis({ label: 'risk E‖θ̂ − θ‖² / σ²', hold: 'union', key: `${d}/${target}` })
  const jx = useAxis({ label: 'coordinate j', range: [0, d + 1], nice: false })
  const jy = useAxis({ label: 'value', hold: 'union', key: `${d}/${seed}` })

  return (
    <Figure
      title="The James–Stein estimator beats the observations"
      state={state}
      hoverReadout={false}
      caption="Left: the risk, in units of σ², of three estimators of d Gaussian means observed once each, against the distance of the means from the shrinkage target. The observations x have risk d at every distance. The James–Stein estimator's exact risk (dashed), from the noncentral chi-squared formula, starts at 2 (3 towards the grand mean) and rises towards d without reaching it; the dots are a simulation of 2,000 draws per distance with common noise, and the positive-part estimator lies below both. Right: one draw at the distance marked on the left. Each coordinate's true mean, its observation and its James–Stein estimate. The estimate pulls every observation towards the target by the same factor; the total squared error falls, while some single coordinates get worse. Drag the vertical line to move the distance."
      readouts={
        <>
          <Readout label="exact risk at this distance (observations: d)" value={`${fmt(exact)} (${d})`} />
          <Readout label="shrinkage factor of the draw" value={fmt(draw.js.factor)} />
          <Readout label="squared error of x / of James–Stein" value={`${fmt(draw.lossX)} / ${fmt(draw.lossJs)}`} />
          <Readout
            label="coordinates made worse"
            value={`${draw.theta.filter((t, j) => (draw.js.estimate[j] - t) ** 2 > (draw.x[j] - t) ** 2).length} of ${d}`}
          />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Plot x={rx} y={ry} title="risk against the distance">
          <Curve name="observations x" x={Array.from(r.norms)} y={Array.from(r.mle)} slot={0} stale={risk.stale} />
          <Points
            name="James–Stein, simulated"
            x={Array.from(r.norms)}
            y={Array.from(r.jamesStein)}
            slot={1}
            stale={risk.stale}
          />
          <Curve
            name="James–Stein, exact"
            x={Array.from(r.norms)}
            y={Array.from(r.exact)}
            emphasis
            dashed
            stale={risk.stale}
          />
          <Curve
            name="positive part, simulated"
            x={Array.from(r.norms)}
            y={Array.from(r.positivePart)}
            slot={2}
            stale={risk.stale}
          />
          <Handle
            kind="x"
            at={distance}
            label="draw"
            onDrag={(x) => state.set('distance', Math.max(0, Math.min(MAX_DISTANCE, Math.round(x * 4) / 4)))}
          />
        </Plot>
        <Plot x={jx} y={jy} title="one draw at the marked distance">
          <Points name="observation x_j" x={index} y={draw.x} slot={0} />
          <Points name="James–Stein estimate" x={index} y={Array.from(draw.js.estimate)} slot={1} />
          <Curve name="true mean θ_j" x={index} y={draw.theta} emphasis dashed showPoints />
          <Curve name="target" x={index} y={Array.from(draw.js.target)} muted />
        </Plot>
      </div>
    </Figure>
  )
}
