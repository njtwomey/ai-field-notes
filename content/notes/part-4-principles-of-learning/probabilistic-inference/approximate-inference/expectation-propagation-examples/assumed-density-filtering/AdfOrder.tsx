import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, Handle, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import {
  ep1d,
  grid,
  logNormalCdf,
  normalLogPdf,
  normalPdf,
  normaliseOnGrid,
  probitTilted,
  toMoments,
  toNat,
} from '../_shared/ep'

const XS = grid(-5, 6, 441)
const X_RANGE: [number, number] = [-5, 6]
const Y_RANGE: [number | undefined, number | undefined] = [0, undefined]
const PRIOR = { mean: 0, variance: 4 }
const SWEEPS = 5

/**
 * Two censored readings of θ absorbed in both orders. The first two updates of EP are assumed density filtering; later
 * updates revisit each site and remove the dependence on order.
 */
export function AdfOrder() {
  const state = useFigureState({
    step: float(2, { min: 0, max: 2 * SWEEPS, step: 1, label: 'site updates', format: (v) => String(v) }),
    above: float(2, { min: -2, max: 5, step: 0.05, label: 'reading above' }),
    below: float(0, { min: -3, max: 4, step: 0.05, label: 'reading below' }),
  })

  const runs = useMemo(() => {
    // Factor 0: the reading θ + noise exceeded `above`. Factor 1: it fell below `below`. Unit noise variance.
    const facs: [number, number][] = [
      [1, state.above],
      [-1, state.below],
    ]
    const tilted = (i: number, c: { mean: number; variance: number }) =>
      probitTilted(c.mean, c.variance, facs[i][0], facs[i][1], 1)
    const run = (order: number[]) => ep1d(toNat(PRIOR), 2, tilted, { sweeps: SWEEPS, order })
    const exact = normaliseOnGrid(
      XS,
      XS.map(
        (x) =>
          normalLogPdf(x, PRIOR.mean, PRIOR.variance) + facs.reduce((s, [y, c]) => s + logNormalCdf(y * (x - c)), 0),
      ),
    )
    return { ab: run([0, 1]), ba: run([1, 0]), exact }
  }, [state.above, state.below])

  const at = (steps: typeof runs.ab) => (state.step === 0 ? PRIOR : toMoments(steps[state.step - 1].q))
  const qa = at(runs.ab)
  const qb = at(runs.ba)
  const series = [
    { name: 'prior', x: XS, y: XS.map((x) => normalPdf(x, PRIOR.mean, PRIOR.variance)), muted: true },
    { name: 'exact posterior', x: XS, y: runs.exact.density, emphasis: true },
    { name: '"above" first', x: XS, y: XS.map((x) => normalPdf(x, qa.mean, qa.variance)), slot: 0 },
    {
      name: '"below" first',
      x: XS,
      y: XS.map((x) => normalPdf(x, qb.mean, qb.variance)),
      slot: 1,
      dashed: true,
    },
  ] as const
  const phase =
    state.step === 0 ? 'prior' : state.step <= 2 ? 'ADF (first pass)' : `EP sweep ${Math.ceil(state.step / 2)}`

  const xAxis = useAxis({ label: 'θ', range: X_RANGE })
  const yAxis = useAxis({ label: 'density', range: Y_RANGE })
  return (
    <Figure
      title="Assumed density filtering depends on order; EP does not"
      state={state}
      caption="Prior θ ~ N(0, 4). Two readings of θ + N(0, 1) noise: one above a threshold, one below another. Each line is the Gaussian approximation after the given number of site updates, absorbing the “above” reading first (solid) or the “below” reading first (dashed). Update 2 ends the single ADF pass, where the orders disagree. Later updates are EP revisiting each site; both orders reach the same Gaussian. Drag either threshold."

      readouts={
        <>
          <Readout label="phase" value={phase} />
          <Readout label="above first: mean, var" value={`${formatNumber(qa.mean)}, ${formatNumber(qa.variance)}`} />
          <Readout label="below first: mean, var" value={`${formatNumber(qb.mean)}, ${formatNumber(qb.variance)}`} />
          <Readout
            label="exact: mean, var"
            value={`${formatNumber(runs.exact.mean)}, ${formatNumber(runs.exact.variance)}`}
          />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Curve {...series[2]} />
        <Curve {...series[3]} />
        <Handle {...state.handle('above', { label: 'reading above' })} />
        <Handle {...state.handle('below', { label: 'reading below' })} />
      </Plot>
    </Figure>
  )
}
