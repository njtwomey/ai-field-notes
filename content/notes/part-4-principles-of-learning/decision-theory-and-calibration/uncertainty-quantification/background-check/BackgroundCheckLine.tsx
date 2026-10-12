import { useMemo } from 'react'
import {
  Area,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  Plot,
  Points,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { backgroundCheck, bias, decide, logBackground, type Bias } from './background'

const LO = -6
const HI = 12
const X = toFlat(linspace(LO, HI, 721))
const Z = toFlat(linspace(0, 1, 201))
const LABELS = ['class 1', 'class 2', 'background']
const clamp01 = (v: number) => Math.min(1, Math.max(0, v))

/** log N(x; m, s²). */
const HALF_LOG_2PI = 0.5 * Math.log(2 * Math.PI)
const logNormal = (x: number, m: number, s: number) => -0.5 * ((x - m) / s) ** 2 - Math.log(s) - HALF_LOG_2PI

/**
 * Background Check on a line with the true foreground known: two Gaussian classes with equal priors and a shared
 * standard deviation. q_f is the foreground density relative to its peak; the bias μ(z) = μ0 + (μ1 − μ0) z^γ gives
 * q_b; the familiarity r = q_f / q_b turns the two-class posterior into three probabilities.
 */
export function BackgroundCheckLine() {
  const state = useFigureState({
    m1: float(-1.5, { min: LO + 1, max: HI - 1, step: 0.05, label: 'class 1 mean', onChart: true }),
    m2: float(1.5, { min: LO + 1, max: HI - 1, step: 0.05, label: 'class 2 mean', onChart: true }),
    sd: float(1, { min: 0.3, max: 2.5, step: 0.05, label: 'class standard deviation' }),
    mu0: float(0.1, { min: 0, max: 1, step: 0.01, label: 'μ(0): background far from the foreground' }),
    mu1: float(0.5, { min: 0, max: 1, step: 0.01, label: 'μ(1): background at the foreground peak' }),
    gamma: float(1, { min: 0.2, max: 5, step: 0.05, label: 'bias shape γ (1 = affine)' }),
    probe: float(6, { min: LO, max: HI, step: 0.05, label: 'test point x', onChart: true }),
  })
  const b: Bias = useMemo(
    () => ({ mu0: state.mu0, mu1: state.mu1, gamma: state.gamma }),
    [state.mu0, state.mu1, state.gamma],
  )

  // log p(x | f) and the within-foreground posterior on the grid; q_f is relative to the grid's peak.
  const fg = useMemo(() => {
    const at = (x: number) => {
      const l1 = Math.log(0.5) + logNormal(x, state.m1, state.sd)
      const l2 = Math.log(0.5) + logNormal(x, state.m2, state.sd)
      const top = Math.max(l1, l2)
      const lse = top + Math.log(Math.exp(l1 - top) + Math.exp(l2 - top))
      return { logP: lse, within: [Math.exp(l1 - lse), Math.exp(l2 - lse)] }
    }
    const grid = Array.from(X, at)
    const peak = Math.max(...grid.map((g) => g.logP))
    return { at, grid, peak }
  }, [state.m1, state.m2, state.sd])

  const curves = useMemo(() => {
    const p1: number[] = []
    const p2: number[] = []
    const pb: number[] = []
    const qf: number[] = []
    const qb: number[] = []
    const bgWins: number[] = []
    for (const g of fg.grid) {
      const lq = g.logP - fg.peak
      const r = backgroundCheck(lq, g.within, b)
      p1.push(r.probs[0])
      p2.push(r.probs[1])
      pb.push(r.probs[2])
      qf.push(Math.exp(lq))
      qb.push(Math.exp(logBackground(lq, b)))
      bgWins.push(decide(r.probs) === 2 ? 1 : 0)
    }
    return { p1, p2, pb, qf, qb, bgWins }
  }, [fg, b])

  const muCurve = useMemo(() => Z.map((z) => bias(z, b)), [b])

  const at = (x: number) => {
    const g = fg.at(x)
    const lq = g.logP - fg.peak
    return { lq, ...backgroundCheck(lq, g.within, b) }
  }
  const probe = at(state.probe)
  const peak1 = at(state.m1).probs[0]
  const peak2 = at(state.m2).probs[1]
  const qfProbe = Math.exp(probe.lq)
  const qbProbe = Math.exp(logBackground(probe.lq, b))

  const tiny = 0.005
  const regime =
    b.mu0 < tiny && b.mu1 < tiny
      ? 'no background: plain classifier'
      : b.mu0 < tiny
        ? Math.abs(b.gamma - 1) < 1e-9
          ? `cautious: Chow's rule with θ = ${formatNumber(b.mu1)}`
          : 'background tied to the foreground'
        : b.mu1 < tiny
          ? 'outlier detection'
          : Math.abs(b.mu0 - b.mu1) < tiny
            ? 'constant background'
            : b.mu0 > b.mu1
              ? 'outliers and ambiguity'
              : 'ambiguity and outliers, background rising with q_f'

  const xAxis = useAxis({ label: 'x', range: [LO, HI] })
  const yAxis = useAxis({ label: 'probability', range: [0, 1] })
  const zAxis = useAxis({ label: 'q_f, relative foreground density', range: [0, 1] })
  const bAxis = useAxis({ label: 'q_b = μ(q_f)', range: [0, 1], equal: zAxis })
  return (
    <Figure
      title="Background Check on a line"
      state={state}
      caption="Left: two Gaussian foreground classes and the three probabilities Background Check gives them: p(class, f | x) for each class and p(b | x) for the background. The shading marks where the background is the most probable label. Drag the class markers to move the classes and the vertical line to move the test point. Right: the background bias μ, which maps the relative foreground density q_f to a relative background density q_b. Drag its two ends, μ(0) and μ(1). Above the dashed diagonal the background is denser than the foreground. μ(0) = 0 gives cautious classification (Chow's rule), μ(1) = 0 outlier detection, and μ(0) = μ(1) a constant background."
      readouts={
        <>
          <Readout label="regime" value={regime} />
          <Readout label="q_f(x)" value={formatNumber(qfProbe)} />
          <Readout label="q_b(x)" value={formatNumber(qbProbe)} />
          <Readout label="familiarity r" value={probe.logR === Infinity ? '∞' : formatNumber(Math.exp(probe.logR))} />
          <Readout label="p(class 1 | x)" value={formatNumber(probe.probs[0])} />
          <Readout label="p(class 2 | x)" value={formatNumber(probe.probs[1])} />
          <Readout label="p(b | x)" value={formatNumber(probe.probs[2])} />
          <Readout label="decision" value={LABELS[decide(probe.probs)]} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <div className="md:col-span-2">
          <Plot x={xAxis} y={yAxis} height={300}>
            <Area name="background wins" x={X} y={curves.bgWins} slot={2} opacity={0.12} line={false} />
            <Curve name="q_f(x)" x={X} y={curves.qf} dashed muted />
            <Curve name="q_b(x)" x={X} y={curves.qb} dashed slot={2} width={1} />
            <Curve name="p(class 1, f | x)" x={X} y={curves.p1} slot={0} />
            <Curve name="p(class 2, f | x)" x={X} y={curves.p2} slot={1} />
            <Curve name="p(b | x)" x={X} y={curves.pb} slot={2} width={3} />
            <Handle
              kind="point"
              at={[state.m1, peak1]}
              label="class 1"
              slot={0}
              onDrag={([x]) => state.set('m1', Math.min(HI - 1, Math.max(LO + 1, x)))}
            />
            <Handle
              kind="point"
              at={[state.m2, peak2]}
              label="class 2"
              slot={1}
              onDrag={([x]) => state.set('m2', Math.min(HI - 1, Math.max(LO + 1, x)))}
            />
            <Handle kind="x" at={state.probe} label="test point" onDrag={(x) => state.set('probe', x)} />
          </Plot>
        </div>
        <Plot x={zAxis} y={bAxis} height={300}>
          <Curve name="r = 1" x={[0, 1]} y={[0, 1]} dashed muted />
          <Curve name="μ(z)" x={Z} y={muCurve} slot={2} />
          <Points name="test point" x={[qfProbe]} y={[qbProbe]} emphasis />
          <Handle kind="point" at={[0, b.mu0]} label="μ(0)" onDrag={([, y]) => state.set('mu0', clamp01(y))} />
          <Handle kind="point" at={[1, b.mu1]} label="μ(1)" onDrag={([, y]) => state.set('mu1', clamp01(y))} />
        </Plot>
      </div>
    </Figure>
  )
}
