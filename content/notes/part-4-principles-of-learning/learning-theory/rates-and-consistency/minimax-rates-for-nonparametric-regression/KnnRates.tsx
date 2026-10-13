import { type BiasVarianceResult } from 'aifn-methods/theory/bias-variance'
import {
  call,
  choice,
  Curve,
  Figure,
  formatNumber,
  Plot,
  Readout,
  slider,
  useAxis,
  useComputed,
  useFigureState,
  type Task,
} from 'aifn-render'

const SIZES = [20, 40, 80, 160, 320, 640]
const REPEATS = 20
const NOISE = 0.5
type Totals = BiasVarianceResult['totals']
const fmt = (v: number | undefined) =>
  v !== undefined && Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—'
const neighbours = (m: number, exponent: number) => Math.max(1, Math.min(m, Math.round(m ** exponent)))

/**
 * k-nearest-neighbour regression with k growing like m^a: the squared bias, the variance and their sum (the excess
 * risk) against m on log–log axes. Small a leaves the variance stuck, large a leaves the bias stuck; a near 2/3
 * balances them and the excess risk falls like m^(−2/3), the minimax rate for Lipschitz functions of one variable.
 */
export function KnnRates() {
  const state = useFigureState({
    target: choice(
      [
        { value: 'sine', label: 'smooth: sin(πx)' },
        { value: 'step', label: 'jump: a step at 0' },
      ],
      'sine',
      { label: 'regression function' },
    ),
    exponent: slider(0, 1, 0.65, { step: 0.05, label: 'k = m^a, exponent a' }),
  })
  const { target, exponent } = state
  const run = useComputed(
    () =>
      SIZES.map((m) =>
        call<BiasVarianceResult>(
          'theory/bias-variance/biasVariance',
          call('foundation/random/stream', `nonparametric-rates/${target}`),
          {
            target,
            noise: NOISE,
            n: m,
            repeats: REPEATS,
            grid: 51,
            keep: 0,
            model: { kind: 'knn', k: neighbours(m, exponent) },
          },
        ),
      ) as unknown as Task<BiasVarianceResult[]>,
    [target, exponent],
    { mode: 'worker', initial: [] as Totals[], then: (results) => results.map((r) => r.totals) },
  )
  const totals = run.value
  const ready = totals.length === SIZES.length
  // A log axis cannot show zeros; a zero term is left out of its curve.
  const positive = (v: number) => (v > 0 ? v : NaN)
  const bias2 = ready ? totals.map((t) => positive(t.bias2)) : []
  const variance = ready ? totals.map((t) => positive(t.variance)) : []
  const excess = ready ? totals.map((t) => positive(t.bias2 + t.variance)) : []
  const first = excess[0]
  const last = excess[SIZES.length - 1]
  const slope =
    ready && first > 0 && last > 0 ? Math.log(last / first) / Math.log(SIZES[SIZES.length - 1] / SIZES[0]) : NaN
  // A guide of slope −2/3 through the first measured point.
  const guide = ready ? [first, first * (SIZES[SIZES.length - 1] / SIZES[0]) ** (-2 / 3)] : []

  const mAxis = useAxis({
    label: 'training points m',
    range: [10, 1000],
    log: true,
    format: (v) => String(Math.round(v)),
  })
  const eAxis = useAxis({ label: 'squared error above the noise', log: true, range: [1e-4, 1] })
  return (
    <Figure
      title="Balancing bias and variance sets the rate"
      purpose="A local average over k neighbours has variance about σ²/k and a bias that grows with the neighbourhood k/m. Growing k as a power of m trades one against the other; the best power gives the minimax rate."
      state={state}
      defaultSize="M"
      readouts={
        <>
          <Readout label="k at m = 20 and 640" value={`${neighbours(20, exponent)}, ${neighbours(640, exponent)}`} />
          <Readout label="excess risk at m = 640" value={fmt(last)} />
          <Readout label="measured slope" value={fmt(slope)} />
        </>
      }
      caption="k-nearest-neighbour regression of y = f(x) + 0.5ε with x uniform on [−1, 1], averaged over 20 training sets at each m. The ink curve is the excess risk, the expected squared error above the noise variance σ² = 0.25, which is the squared bias plus the variance. The dashed line has slope −2/3, drawn through the first point. With a = 0 (fixed k) the variance stalls near σ²/k; with a = 1 (k a fixed fraction of m) the bias stalls. Near a = 2/3 both fall and the sine's excess risk has slope close to −2/3. The step function is not Lipschitz: its bias is concentrated near the jump, the best exponent is near ½, and the slope is near −½. Computed in the background; the curves appear when the run finishes."
    >
      <Plot x={mAxis} y={eAxis} height={340}>
        {ready && (
          <>
            <Curve name="slope −2/3" x={[SIZES[0], SIZES[SIZES.length - 1]]} y={guide} muted dashed />
            <Curve name="squared bias" x={SIZES} y={bias2} slot={0} stale={run.stale} />
            <Curve name="variance" x={SIZES} y={variance} slot={1} stale={run.stale} />
            <Curve name="excess risk" x={SIZES} y={excess} emphasis showPoints stale={run.stale} />
          </>
        )}
      </Plot>
    </Figure>
  )
}
