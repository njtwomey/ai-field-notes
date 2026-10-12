import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  Rug,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { drawSamples, kdeLogDensity, kdeLogRatio, logP, logQ, logRatio, type Shift } from '../_shared/ratio'

const LO = -4
const HI = 5
const X = toFlat(linspace(LO, HI, 241))
const Y_RANGE: [number, number] = [-10, 6]
const clampY = (v: number) => Math.min(Y_RANGE[1] + 2, Math.max(Y_RANGE[0] - 2, v))

/** Many draws as one curve broken by NaN, so they share a legend entry and a colour. */
function joined(lines: readonly number[][]) {
  const x: number[] = []
  const y: number[] = []
  for (const line of lines) {
    x.push(...X, NaN)
    y.push(...line.map(clampY), NaN)
  }
  return { x, y }
}

/**
 * The plug-in estimate log p̂ − log q̂ from two kernel density estimates, over repeated draws of both samples, against
 * the true log ratio. Inside the bulk of q the quotient is close; where either sample runs out, the two estimated tails
 * divide into a ratio set by the bandwidth and the last few points, not by p and q.
 */
export function DividingDensities() {
  const state = useFigureState({
    mean: float(1, { min: -2, max: 3, step: 0.05, label: 'mean of p', onChart: true }),
    sd: float(0.6, { min: 0.3, max: 1.6, step: 0.05, label: 'standard deviation of p' }),
    n: int(200, { min: 20, max: 2000, label: 'points in each sample', suggestions: [50, 200, 2000] }),
    scale: float(1, { min: 0.25, max: 3, step: 0.05, label: "bandwidth (× Scott's rule)" }),
    draws: int(8, { min: 1, max: 40, label: 'repeated draws', suggestions: [1, 8, 40] }),
    seed: int(1, { min: 0, max: 999, label: 'seed' }),
  })
  const shift: Shift = useMemo(() => ({ mean: state.mean, sd: state.sd }), [state.mean, state.sd])

  const truth = useMemo(
    () => ({
      logR: X.map((x) => logRatio(x, shift)),
      logP: X.map((x) => logP(x, shift)),
      logQ: X.map((x) => logQ(x)),
    }),
    [shift],
  )

  // One pair of samples per draw; the first is also shown as rugs and as its two density estimates.
  const runs = useMemo(
    () =>
      Array.from({ length: state.draws }, (_, k) => {
        const { xp, xq } = drawSamples(state.seed * 1000 + k, state.n, state.n, shift)
        return { xp, xq, logR: kdeLogRatio(xp, xq, X, state.scale) }
      }),
    [state.draws, state.seed, state.n, state.scale, shift],
  )
  const first = runs[0]
  const estimates = useMemo(
    () => ({ logP: kdeLogDensity(first.xp, X, state.scale), logQ: kdeLogDensity(first.xq, X, state.scale) }),
    [first, state.scale],
  )

  const lines = useMemo(() => joined(runs.map((r) => r.logR)), [runs])

  // Mean absolute error of log r̂, inside the central 95% of q (|x| ≤ 1.96) and outside it, over the draws.
  const errors = useMemo(() => {
    let inner = 0
    let outer = 0
    let ni = 0
    let no = 0
    for (const r of runs)
      X.forEach((x, i) => {
        const e = Math.abs(r.logR[i] - truth.logR[i])
        if (Math.abs(x) <= 1.96) {
          inner += e
          ni++
        } else {
          outer += e
          no++
        }
      })
    return { inner: inner / ni, outer: outer / no }
  }, [runs, truth])

  const xa = useAxis({ label: 'x', range: [LO, HI] })
  const ya = useAxis({ label: 'log density', range: [-12, 1] })
  const xb = useAxis({ label: 'x', range: [LO, HI] })
  const yb = useAxis({ label: 'log r(x)', range: Y_RANGE })
  const peak = logRatio(state.mean, shift)
  return (
    <Figure
      title="Dividing two density estimates"
      state={state}
      caption="Left: for one draw, the kernel density estimates of p and q on a log scale, with the true log densities dashed. Each estimate follows its density near its sample and falls off as a Gaussian of the bandwidth beyond it. Right: the log of their quotient for each repeated draw (thin lines) against the true log ratio. Drag the marker on the true curve to move p. In the bulk of q the quotient is close; in the tails it is the ratio of two kernel tails, set by the bandwidth and the last few points of each sample, and it swings by orders of magnitude from draw to draw."
      readouts={
        <>
          <Readout label="mean |error| of log r̂, |x| ≤ 1.96" value={formatNumber(errors.inner)} />
          <Readout label="mean |error| of log r̂, |x| > 1.96" value={formatNumber(errors.outer)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Plot x={xa} y={ya} height={300}>
          <Curve name="log q(x)" x={X} y={truth.logQ} slot={0} dashed />
          <Curve name="log p(x)" x={X} y={truth.logP} slot={1} dashed />
          <Curve name="log q̂(x)" x={X} y={estimates.logQ} slot={0} />
          <Curve name="log p̂(x)" x={X} y={estimates.logP} slot={1} />
          <Rug name="q sample" values={first.xq} slot={0} />
          <Rug name="p sample" values={first.xp} slot={1} />
        </Plot>
        <Plot x={xb} y={yb} height={300}>
          <Curve name="log p̂ − log q̂, one per draw" x={lines.x} y={lines.y} slot={2} thin silent />
          <Curve name="log r(x)" x={X} y={truth.logR} emphasis />
          <Handle
            kind="point"
            at={[state.mean, peak]}
            label="p"
            onDrag={([x]) => state.set('mean', Math.min(3, Math.max(-2, x)))}
          />
        </Plot>
      </div>
    </Figure>
  )
}
