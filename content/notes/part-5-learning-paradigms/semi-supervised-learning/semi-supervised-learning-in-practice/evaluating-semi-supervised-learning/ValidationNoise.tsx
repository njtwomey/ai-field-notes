import { Curve, Figure, float, formatNumber, Handle, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { normalCdf } from 'aifn-compute/numerics/special'

const N_MIN = 10
const N_MAX = 100000
const NS = Array.from({ length: 81 }, (_, i) => N_MIN * (N_MAX / N_MIN) ** (i / 80))
const Z95 = 1.6448536269514722

/**
 * How often a validation set of n points ranks two models correctly, when their true accuracies differ by Δ: the
 * normal approximation to the difference of two measured accuracies, with errors correlated across the two models.
 */
export function ValidationNoise() {
  const fig = useFigureState({
    accuracy: float(0.9, { min: 0.5, max: 0.99, step: 0.01, label: 'accuracy of the worse model' }),
    gap: float(0.01, {
      min: 0.001,
      max: 0.1,
      step: 0.001,
      label: 'true gap Δ',
      suggestions: [0.005, 0.01, 0.02, 0.05],
    }),
    correlation: float(0, { min: 0, max: 0.95, step: 0.05, label: 'correlation of the two models’ errors ρ' }),
    n: int(250, { min: N_MIN, max: N_MAX, scale: 'log10', label: 'validation points n', onChart: true }),
  })
  const { accuracy: a, gap, correlation: rho, n } = fig
  const b = Math.min(0.999, a + gap)
  const va = a * (1 - a)
  const vb = b * (1 - b)
  const varianceOf = (r: number) => Math.max(1e-12, va + vb - 2 * r * Math.sqrt(va * vb))
  const correct = (r: number, m: number) => normalCdf(gap / Math.sqrt(varianceOf(r) / m))
  const needed = (r: number) => (Z95 ** 2 * varianceOf(r)) / gap ** 2
  const hoeffding = Math.log(2 / 0.05) / (2 * gap ** 2)

  const xa = useAxis({ label: 'validation points n', log: true, range: [N_MIN, N_MAX] })
  const ya = useAxis({ label: 'P(validation ranks the better model first)', range: [0.5, 1] })

  return (
    <Figure
      title="Small validation sets cannot rank close methods"
      state={fig}
      caption="Two models have true accuracies a and a + Δ. A validation set of n points measures each; the measured difference is approximately normal with mean Δ and variance (a(1 − a) + b(1 − b) − 2ρ√(a(1 − a)b(1 − b)))/n, where b = a + Δ and ρ is the correlation between the two models' per-point errors. The curves give the probability that the better model also measures better, for uncorrelated errors (dashed) and for the chosen ρ (solid). Drag the vertical line to set n. With a = 0.9 and Δ = 0.01, a validation set of 250 points ranks the models correctly about two times in three; about 4,700 points are needed for 95%, and Hoeffding's bound asks for about 18,000 to pin each accuracy to ±Δ."
      readouts={
        <>
          <Readout label="P(correct ranking) at n" value={formatNumber(correct(rho, n))} />
          <Readout label="n for 95%, ρ = 0" value={formatNumber(Math.ceil(needed(0)))} />
          <Readout label="n for 95%, chosen ρ" value={formatNumber(Math.ceil(needed(rho)))} />
          <Readout label="Hoeffding n for ±Δ at 95%" value={formatNumber(Math.ceil(hoeffding))} />
        </>
      }
    >
      <Plot x={xa} y={ya}>
        <Curve name="uncorrelated errors" x={NS} y={NS.map((m) => correct(0, m))} slot={0} dashed />
        <Curve name="chosen correlation ρ" x={NS} y={NS.map((m) => correct(rho, m))} slot={0} />
        <Curve name="95%" x={[N_MIN, N_MAX]} y={[0.95, 0.95]} muted dashed />
        <Handle
          kind="x"
          at={n}
          label="n"
          onDrag={(v) => fig.set('n', Math.round(Math.min(N_MAX, Math.max(N_MIN, v))))}
        />
      </Plot>
    </Figure>
  )
}
