import { useMemo, useState } from 'react'
import { child, normal, stream } from 'aifn-compute/foundation/random'
import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { Normal } from 'aifn-compute/probability/distributions'
import { ecdf, ecdfAt } from 'aifn-compute/probability/stats'
import { kolmogorovNull, kolmogorovSf, ksStatistic } from 'aifn-compute/probability/tests'
import { tailBounds } from 'aifn-methods/theory/concentration'
import {
  Area,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Plots,
  Readout,
  Segments,
  useAxis,
  useFigureState,
} from 'aifn-render'

const LO = -3.5
const HI = 3.5
const law = Normal(0, 1)
const cdf = (t: number) => law.cdf(t) as number
const GRID = Array.from({ length: 141 }, (_, i) => LO + ((HI - LO) * i) / 140)
const fmt = (v: number) => formatNumber(Number(v.toPrecision(3)))

/**
 * The empirical cdf of m standard normal draws against the true cdf: the largest gap (the uniform deviation over all
 * half-lines), the gap at one draggable point, the exact 1 − δ band, and the exact tail of the largest gap beside the
 * DKW bound 2 exp(−2 m t²).
 */
export function EcdfBand() {
  const state = useFigureState({
    m: int(50, { ge: 5, le: 2000, suggestions: [10, 50, 200, 1000], label: 'sample size m' }),
    delta: float(0.05, { ge: 0.001, le: 0.5, suggestions: [0.01, 0.05, 0.1], label: 'δ' }),
    seed: int(1, { ge: 0, le: 999, label: 'seed' }),
  })
  const { m, delta, seed } = state
  const [t0, setT0] = useState(-0.8)

  const sample = useMemo(
    () => toFlat(normal(child(stream('dkw'), 'sample', seed), 0, 1, { shape: [m] }) as Tensor),
    [m, seed],
  )
  const steps = useMemo(() => {
    const { values, probabilities } = ecdf(sample)
    const v = toFlat(values)
    const p = toFlat(probabilities)
    const x: number[] = [LO]
    const y: number[] = [0]
    v.forEach((value, i) => {
      x.push(value, value)
      y.push(i === 0 ? 0 : p[i - 1], p[i])
    })
    x.push(HI)
    y.push(1)
    return { x, y }
  }, [sample])
  const ks = useMemo(() => ksStatistic(sample, cdf), [sample])
  const half = useMemo(() => kolmogorovNull(m).isf(delta) as number, [m, delta])
  const truth = GRID.map(cdf)
  const bandLow = truth.map((v) => Math.max(0, v - half))
  const bandHigh = truth.map((v) => Math.min(1, v + half))
  const atT0 = toFlat(ecdfAt(sample, [t0]))[0]
  const atSup = toFlat(ecdfAt(sample, [ks.location]))[0]
  // The sup is approached from the left at a draw when the empirical cdf is below the truth there.
  const supEcdf = ks.sign > 0 ? atSup : cdf(ks.location) - ks.statistic

  const tails = useMemo(() => {
    const tMax = Math.min(0.9, 2.6 / Math.sqrt(m))
    const t = Array.from({ length: 60 }, (_, i) => (tMax * (i + 1)) / 60)
    return { t, exact: t.map((d) => kolmogorovSf(d, m)), dkw: Array.from(tailBounds(t, m, 0.5, 0.25).hoeffding) }
  }, [m])
  const pExact = kolmogorovSf(ks.statistic, m)
  const pDkw = tailBounds([ks.statistic], m, 0.5, 0.25).hoeffding[0]

  const xa = useAxis({ label: 'x', range: [LO, HI] })
  const ya = useAxis({ label: 'cumulative probability', range: [0, 1] })
  const ta = useAxis({ label: 'deviation t', range: [0, tails.t[tails.t.length - 1]] })
  const pa = useAxis({ label: 'P(sup gap ≥ t)', range: [1e-4, 1], log: true })
  return (
    <Figure
      title="The empirical cdf converges uniformly"
      state={state}
      defaultSize="M"
      readouts={
        <>
          <Readout label="largest gap D" value={fmt(ks.statistic)} />
          <Readout label="gap at the handle" value={fmt(Math.abs(atT0 - cdf(t0)))} />
          <Readout label="exact 1 − δ half-width" value={fmt(half)} />
          <Readout label="P(D ≥ observed), exact" value={fmt(pExact)} />
          <Readout label="DKW bound on it" value={fmt(pDkw)} />
        </>
      }
      caption="Left: m draws from N(0, 1), their empirical cdf (slot 0), the true cdf and the band that contains the empirical cdf everywhere with probability 1 − δ, from the exact law of the largest gap (the engine's kolmogorovNull). The thick segment is the largest gap D, the uniform deviation over every half-line (−∞, t]; the thin one is the gap at the point t you drag. Right: the probability that the largest gap reaches t, exactly (kolmogorovSf) and as bounded by Dvoretzky, Kiefer, Wolfowitz and Massart, 2 exp(−2mt²) (the engine's tailBounds; the same expression as Hoeffding's bound for one fixed t). The two curves nearly coincide in the tail, so the uniform bound costs almost nothing over the pointwise one."
    >
      <Plots cols={2}>
        <Plot x={xa} y={ya} title={`m = ${m}`}>
          <Area
            name={`exact ${Math.round((1 - delta) * 100)}% band`}
            x={GRID}
            y={bandHigh}
            base={bandLow}
            opacity={0.2}
            line={false}
            slot={2}
          />
          <Curve name="true cdf F" x={GRID} y={truth} emphasis thin dashed />
          <Curve name="empirical cdf" x={steps.x} y={steps.y} slot={0} />
          <Segments
            name="largest gap D"
            segments={[{ from: [ks.location, cdf(ks.location)], to: [ks.location, supEcdf] }]}
            emphasis
            width={3}
          />
          <Segments name="gap at t" segments={[{ from: [t0, cdf(t0)], to: [t0, atT0] }]} slot={1} width={2} />
          <Handle kind="x" at={t0} label="t" onDrag={(v) => setT0(Math.max(LO, Math.min(HI, v)))} />
        </Plot>
        <Plot x={ta} y={pa} title="tail of the largest gap">
          <Curve name="exact" x={tails.t} y={tails.exact} slot={0} />
          <Curve name="DKW bound 2e^(−2mt²)" x={tails.t} y={tails.dkw} emphasis dashed />
          <Curve name="δ" x={[0, tails.t[tails.t.length - 1]]} y={[delta, delta]} thin dashed silent />
        </Plot>
      </Plots>
    </Figure>
  )
}
