import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'
import { linspace } from '@/lib/math'
import { normalCdf } from '@/lib/math/special'

type Mechanism = 'laplace' | 'gaussian'

const COUNT = 10
const XS = linspace(-10, 30, 801)

/**
 * The output densities of a noisy count on two neighbouring data sets whose true counts differ by the sensitivity Δ.
 * Laplace noise of scale Δ/ε keeps the density ratio within e^ε everywhere. Gaussian noise with the same standard
 * deviation has a ratio that grows without bound in the tails, so it needs the extra slack δ.
 */
export function NeighbouringOutputs() {
  const eps = useParam(0.5, { min: 0.1, max: 2, step: 0.05 })
  const delta = useParam(1, { min: 1, max: 3, step: 1 })
  const z = useParam(13, { min: -10, max: 30, step: 0.1 })
  const [mech, setMech] = useState<Mechanism>('laplace')

  const b = delta.value / eps.value
  const sd = Math.SQRT2 * b
  const a = COUNT
  const a2 = COUNT + delta.value
  const pdf = useMemo(() => {
    if (mech === 'laplace') return (x: number, m: number) => Math.exp(-Math.abs(x - m) / b) / (2 * b)
    return (x: number, m: number) => Math.exp(-0.5 * ((x - m) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI))
  }, [mech, b, sd])
  const curves = useMemo(() => ({ d: XS.map((x) => pdf(x, a)), d2: XS.map((x) => pdf(x, a2)) }), [pdf, a, a2])

  // Privacy loss at the output z: log of the ratio of the two densities.
  const loss =
    mech === 'laplace'
      ? (Math.abs(z.value - a2) - Math.abs(z.value - a)) / b
      : ((z.value - a2) ** 2 - (z.value - a) ** 2) / (2 * sd * sd)
  // Probability that the privacy loss exceeds ε when the output is drawn on D. Zero for Laplace; for Gaussian the loss
  // is normal with mean Δ²/(2s²) and standard deviation Δ/s.
  const tail =
    mech === 'laplace' ? 0 : 1 - normalCdf((eps.value - delta.value ** 2 / (2 * sd * sd)) / (delta.value / sd))

  const series: XYSeries[] = [
    { name: `output on D (count ${a})`, type: 'line', x: XS, y: curves.d, slot: 0, area: true },
    { name: `output on D′ (count ${a2})`, type: 'line', x: XS, y: curves.d2, slot: 1, area: true },
  ]
  const handles: Handle[] = [{ kind: 'x', at: z.value, label: 'output z', onDrag: z.set }]

  return (
    <Interactive
      title="Noisy counts on neighbouring data sets"
      caption="Two data sets differ in one person, which changes a count by at most the sensitivity Δ. Each curve is the distribution of the released count. Drag the output z: the privacy loss is the log ratio of the two densities at z. With Laplace noise of scale Δ/ε it never exceeds ε, so no output betrays whether the person is in the data by more than a factor e^ε. Switch to Gaussian noise of the same standard deviation: the loss is small near the centre but grows without bound in the tails, and the probability of exceeding ε becomes the δ of (ε, δ)-privacy."
      controls={
        <>
          <ParamChoice
            label="noise"
            value={mech}
            onChange={setMech}
            options={[
              { value: 'laplace', label: 'Laplace' },
              { value: 'gaussian', label: 'Gaussian' },
            ]}
          />
          <ParamSlider label="privacy budget ε" param={eps} />
          <ParamSlider label="sensitivity Δ" param={delta} />
        </>
      }
      readout={
        <>
          <Readout label="noise standard deviation" value={formatNumber(sd)} />
          <Readout label="privacy loss at z" value={formatNumber(Math.abs(loss))} />
          <Readout label="bound ε" value={formatNumber(eps.value)} />
          <Readout label="P(loss > ε) on D" value={tail === 0 ? '0' : tail.toExponential(2)} />
        </>
      }
    >
      <XYChart
        series={series}
        handles={handles}
        xRange={[-10, 30]}
        yRange={[0, undefined]}
        xLabel="released count"
        yLabel="density"
        ariaLabel="Output densities of a noisy count on two neighbouring data sets"
      />
    </Interactive>
  )
}
