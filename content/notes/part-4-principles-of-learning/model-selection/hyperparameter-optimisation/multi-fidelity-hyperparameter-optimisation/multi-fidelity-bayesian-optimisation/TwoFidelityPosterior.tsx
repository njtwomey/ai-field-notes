import { useMemo } from 'react'
import {
  Area,
  Curve,
  Figure,
  formatNumber,
  int,
  Plot,
  Points,
  Readout,
  setting,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { tensor, toFlat } from 'aifn-compute/foundation/tensor'
import { forrester } from 'aifn-methods/data/objectives'
import { fitMultiFidelityGp } from 'aifn-methods/learning/gaussian-processes'

const HIGH = forrester()
const LOW = forrester({ fidelity: 'low' })
/** The full-fidelity loss f and its cheap version at one point. */
const f = (x: number) => HIGH.value(tensor([x]))
const fLow = (x: number) => LOW.value(tensor([x]))
const XS = Array.from({ length: 201 }, (_, i) => i / 200)
const GRID = tensor(XS)
const TRUE_HIGH = XS.map(f)
const TRUE_LOW = XS.map(fLow)
const Y_RANGE = [-12, 20] as const

/** n evenly spaced points on [0, 1], both ends included. */
const evenly = (n: number) => Array.from({ length: n }, (_, i) => i / (n - 1))

/** The posterior mean and ±2 sd band of one fidelity on the grid, its error against f, and where its mean is lowest. */
function summarise(p: { mean: ArrayLike<number>; sd: ArrayLike<number> }) {
  const mean = Array.from(p.mean)
  const sd = Array.from(p.sd)
  const rmse = Math.sqrt(mean.reduce((a, m, i) => a + (m - TRUE_HIGH[i]) ** 2, 0) / XS.length)
  const at = mean.indexOf(Math.min(...mean))
  return {
    mean,
    upper: mean.map((m, i) => m + 2 * sd[i]),
    lower: mean.map((m, i) => m - 2 * sd[i]),
    rmse,
    argmin: XS[at],
    trueAtArgmin: TRUE_HIGH[at],
  }
}

/**
 * The autoregressive two-fidelity Gaussian process on Forrester's function: many cheap evaluations and a few
 * expensive ones, against a Gaussian process fitted to the expensive evaluations alone.
 */
export function TwoFidelityPosterior() {
  const state = useFigureState({
    high: int(4, { ge: 3, le: 10, suggestions: [3, 4, 5, 6], label: 'expensive evaluations' }),
    low: int(11, { ge: 4, le: 41, suggestions: [6, 11, 21], label: 'cheap evaluations' }),
    alone: setting(true, 'show a GP on the expensive evaluations alone'),
  })
  const { high, low, alone } = state

  const fit = useMemo(() => {
    const xLow = evenly(low)
    const xHigh = evenly(high)
    const x = tensor([...xLow, ...xHigh])
    const fidelity = [...xLow.map(() => 0), ...xHigh.map(() => 1)]
    const y = tensor([...xLow.map(fLow), ...xHigh.map(f)])
    const gp = fitMultiFidelityGp(x, fidelity, y)
    const p = gp.predict(GRID, 1)
    const single = fitMultiFidelityGp(
      tensor(xHigh),
      xHigh.map(() => 0),
      tensor(xHigh.map(f)),
    )
    const q = single.predict(GRID, 0)
    return {
      xLow,
      yLow: xLow.map(fLow),
      xHigh,
      yHigh: xHigh.map(f),
      rho: gp.rho[0],
      joint: summarise({ mean: toFlat(p.mean), sd: toFlat(p.sd) }),
      single: summarise({ mean: toFlat(q.mean), sd: toFlat(q.sd) }),
    }
  }, [high, low])

  const x = useAxis({ label: 'hyperparameter x', range: [0, 1] })
  const y = useAxis({ label: 'loss', range: Y_RANGE })
  const where = (s: { argmin: number; trueAtArgmin: number }) =>
    `x = ${formatNumber(s.argmin)}, f = ${formatNumber(s.trueAtArgmin)}`
  return (
    <Figure
      title="Cheap evaluations through a learned relation"
      state={state}
      caption="The Forrester function f stands for the full-fidelity loss (ink, dashed) and its tilted cheap version for a low-fidelity evaluation (muted). Orange points are cheap evaluations and ink points expensive ones, both evenly spaced on [0, 1]. Blue is the posterior of the autoregressive model f = ρ·f_lo + δ, fitted by marginal likelihood: its mean and a band of two standard deviations at full fidelity. Green, dashed, is a Gaussian process fitted to the expensive evaluations alone. The readouts give the fitted scale ρ (the truth is 2), the root-mean-square error of each mean against f over [0, 1], and where each mean is lowest with the true loss there. With four expensive evaluations the joint model finds the narrow minimum near 0.757 and the expensive-only model does not. With three, at 0, 0.5 and 1, the fit cannot tell the scale from the discrepancy and settles on the wrong ρ."
      readouts={{
        'joint model': (
          <>
            <Readout label="fitted ρ" value={formatNumber(fit.rho)} />
            <Readout label="error of the mean (RMS)" value={formatNumber(fit.joint.rmse)} />
            <Readout label="lowest mean at" value={where(fit.joint)} />
          </>
        ),
        'expensive evaluations alone': (
          <>
            <Readout label="error of the mean (RMS)" value={formatNumber(fit.single.rmse)} />
            <Readout label="lowest mean at" value={where(fit.single)} />
          </>
        ),
        truth: <Readout label="minimum of f" value={`x = 0.757, f = ${formatNumber(HIGH.minimumValue)}`} />,
      }}
    >
      <Plot x={x} y={y}>
        <Area
          name="joint model ± 2 sd"
          x={XS}
          y={fit.joint.upper}
          base={fit.joint.lower}
          slot={0}
          opacity={0.18}
          line={false}
        />
        <Curve name="cheap version f_lo" x={XS} y={TRUE_LOW} muted thin />
        <Curve name="full fidelity f" x={XS} y={TRUE_HIGH} emphasis dashed thin />
        <Curve name="joint model, mean" x={XS} y={fit.joint.mean} slot={0} />
        {alone && <Curve name="expensive alone, mean" x={XS} y={fit.single.mean} slot={2} dashed />}
        <Points name="cheap evaluations" x={fit.xLow} y={fit.yLow} slot={1} size={6} />
        <Points name="expensive evaluations" x={fit.xHigh} y={fit.yHigh} emphasis size={8} />
      </Plot>
    </Figure>
  )
}
