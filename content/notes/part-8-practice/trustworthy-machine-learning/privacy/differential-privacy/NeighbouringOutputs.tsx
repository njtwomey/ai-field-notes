import { useMemo } from 'react'
import {
  Area,
  choice,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { normalCdf } from 'aifn-compute/numerics/special'

type Mechanism = 'laplace' | 'gaussian'

const COUNT = 10
const XS = toFlat(linspace(-10, 30, 801))

/**
 * The output densities of a noisy count on two neighbouring data sets whose true counts differ by the sensitivity Δ.
 * Laplace noise of scale Δ/ε keeps the density ratio within e^ε everywhere. Gaussian noise with the same standard
 * deviation has a ratio that grows without bound in the tails, so it needs the extra slack δ.
 */
export function NeighbouringOutputs() {
  const state = useFigureState({
    mech: choice<Mechanism>(
      [
        { value: 'laplace', label: 'Laplace' },
        { value: 'gaussian', label: 'Gaussian' },
      ],
      'laplace',
      { label: 'noise' },
    ),
    eps: float(0.5, { min: 0.1, max: 2, step: 0.05, label: 'privacy budget ε' }),
    delta: int(1, { min: 1, max: 3, step: 1, label: 'sensitivity Δ' }),
    z: slider(-10, 30, 13, { step: 0.1, onChart: true }),
  })

  const b = state.delta / state.eps
  const sd = Math.SQRT2 * b
  const a = COUNT
  const a2 = COUNT + state.delta
  const pdf = useMemo(() => {
    if (state.mech === 'laplace') return (x: number, m: number) => Math.exp(-Math.abs(x - m) / b) / (2 * b)
    return (x: number, m: number) => Math.exp(-0.5 * ((x - m) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI))
  }, [state.mech, b, sd])
  const curves = useMemo(() => ({ d: XS.map((x) => pdf(x, a)), d2: XS.map((x) => pdf(x, a2)) }), [pdf, a, a2])

  // Privacy loss at the output z: log of the ratio of the two densities.
  const loss =
    state.mech === 'laplace'
      ? (Math.abs(state.z - a2) - Math.abs(state.z - a)) / b
      : ((state.z - a2) ** 2 - (state.z - a) ** 2) / (2 * sd * sd)
  // Probability that the privacy loss exceeds ε when the output is drawn on D. Zero for Laplace; for Gaussian the loss
  // is normal with mean Δ²/(2s²) and standard deviation Δ/s.
  const tail =
    state.mech === 'laplace' ? 0 : 1 - normalCdf((state.eps - state.delta ** 2 / (2 * sd * sd)) / (state.delta / sd))

  const series = [
    { name: `output on D (count ${a})`, x: XS, y: curves.d, slot: 0 },
    { name: `output on D′ (count ${a2})`, x: XS, y: curves.d2, slot: 1 },
  ] as const

  const xAxis = useAxis({ label: 'released count', range: [-10, 30] })
  const yAxis = useAxis({ label: 'density', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="Noisy counts on neighbouring data sets"
      purpose="Compare the output distributions of a noisy count on two neighbouring data sets and read the privacy loss at a dragged output z."
      state={state}
      caption="Two data sets differ in one person, which changes a count by at most the sensitivity Δ. Each curve is the distribution of the released count. Drag the output z: the privacy loss is the log ratio of the two densities at z. With Laplace noise of scale Δ/ε it never exceeds ε, so no output betrays whether the person is in the data by more than a factor e^ε. Switch to Gaussian noise of the same standard deviation: the loss is small near the centre but grows without bound in the tails, and the probability of exceeding ε becomes the δ of (ε, δ)-privacy."

      readouts={
        <>
          <Readout label="noise standard deviation" value={formatNumber(sd)} />
          <Readout label="privacy loss at z" value={formatNumber(Math.abs(loss))} />
          <Readout label="bound ε" value={formatNumber(state.eps)} />
          <Readout label="P(loss > ε) on D" value={tail === 0 ? '0' : tail.toExponential(2)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} ariaLabel={'Output densities of a noisy count on two neighbouring data sets'}>
        <Area {...series[0]} />
        <Area {...series[1]} />
        <Handle {...state.handle('z', { label: 'output z' })} />
      </Plot>
    </Figure>
  )
}
