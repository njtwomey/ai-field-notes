import {
  Curve,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Readout,
  useAxis,
  slider,
  useFigureState,
  variants,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn/foundation/tensor'

type FamilyId = 'bernoulli' | 'poisson' | 'exponential'

type Family = {
  label: string
  /** Natural parameter range shown on the axis. */
  range: [number, number]
  initial: number
  A: (eta: number) => number
  /** A′(η) = E[T(X)]. */
  mean: (eta: number) => number
  /** A″(η) = var[T(X)]. */
  variance: (eta: number) => number
  /** The usual parameter, as text. */
  usual: (eta: number) => string
}

const FAMILIES: Record<FamilyId, Family> = {
  bernoulli: {
    label: 'Bernoulli',
    range: [-6, 6],
    initial: 1,
    A: (eta) => Math.log1p(Math.exp(eta)),
    mean: (eta) => 1 / (1 + Math.exp(-eta)),
    variance: (eta) => {
      const p = 1 / (1 + Math.exp(-eta))
      return p * (1 - p)
    },
    usual: (eta) => `p = ${formatNumber(1 / (1 + Math.exp(-eta)))}`,
  },
  poisson: {
    label: 'Poisson',
    range: [-3, 3],
    initial: 1,
    A: (eta) => Math.exp(eta),
    mean: (eta) => Math.exp(eta),
    variance: (eta) => Math.exp(eta),
    usual: (eta) => `λ = ${formatNumber(Math.exp(eta))}`,
  },
  exponential: {
    label: 'exponential',
    range: [-5, -0.2],
    initial: -1,
    A: (eta) => -Math.log(-eta),
    mean: (eta) => -1 / eta,
    variance: (eta) => 1 / (eta * eta),
    usual: (eta) => `λ = ${formatNumber(-eta)}`,
  },
}

/** The log-partition function A(η) of a one-parameter family, with its tangent: the slope is the mean of T(X). */
export function LogPartition() {
  // One η per family (a variants case each), so switching family keeps each family's own value inside its own range.
  const eta = (id: FamilyId) => ({
    label: FAMILIES[id].label,
    params: {
      eta: slider(FAMILIES[id].range[0], FAMILIES[id].range[1], FAMILIES[id].initial, {
        step: 0.05,
        label: 'natural parameter η',
      }),
    },
  })
  const state = useFigureState({
    family: variants(
      { bernoulli: eta('bernoulli'), poisson: eta('poisson'), exponential: eta('exponential') },
      { choiceLabel: 'family' },
    ),
  })
  const f = FAMILIES[state.family.key]
  const [lo, hi] = f.range
  const setEta = (v: number) => state.set('family.eta', v)

  const xs = toFlat(linspace(lo, hi, 200))
  const e = state.family.values.eta
  const slope = f.mean(e)
  const series = [
    { name: 'A(η)', x: xs, y: xs.map(f.A), slot: 0 },
    {
      name: 'tangent, slope E[T(X)]',
      x: xs,
      y: xs.map((x) => f.A(e) + slope * (x - e)),
      dashed: true,
      slot: 1,
    },
  ] as const
  const ys = xs.map(f.A)
  const yRange: [number, number] = [Math.min(...ys) - 0.5, Math.max(...ys) + 0.5]

  const xAxis = useAxis({ label: 'η', range: f.range })
  const yAxis = useAxis({ label: 'A(η)', range: yRange })
  return (
    <Figure
      title="The log-partition function generates the moments"
      state={state}
      caption="A(η) is convex. Its slope at η is the mean of the sufficient statistic, and its curvature is the variance. Drag the point along the curve, or use the slider. For the Poisson, A = A′ = A″ = e^η, so the mean equals the variance."
      readouts={
        <>
          <Readout label="usual parameter" value={f.usual(e)} />
          <Readout label="A′(η) = E[T(X)]" value={formatNumber(slope)} />
          <Readout label="A″(η) = var[T(X)]" value={formatNumber(f.variance(e))} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Handle kind="point" at={[e, f.A(e)]} onDrag={([x]) => setEta(x)} />
      </Plot>
    </Figure>
  )
}
