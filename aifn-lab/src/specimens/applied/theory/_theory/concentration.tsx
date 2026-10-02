/**
 * Concentration inequalities against simulated tails, McDiarmid on a function that is not a sum, and the law of large
 * numbers and central limit theorem by simulation (`aifn-applied/theory/concentration`).
 */
import { useMemo } from 'react'
import {
  concentrationStudy,
  lawMoments,
  mcdiarmidStudy,
  runningMeans,
  standardisedSums,
  tailBounds,
  type SummandLaw,
} from 'aifn-applied/theory/concentration'
import { stream } from 'aifn/foundation/random'
import { Normal } from 'aifn/probability/distributions'
import { Figure } from '@lab/layout'
import { choice, int, row, slider, useComputed, useFigureState, type AnyValues } from '@lab/state'
import { Curve, Density, formatNumber, Handle, Histogram, Plot, Plots, Readout, useAxis } from '@lab/viz'

const fmt = (v: number, digits = 3) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(digits))) : '—')
const BOUNDED = [
  { value: 'bernoulli', label: 'Bernoulli(p)' },
  { value: 'uniform', label: 'uniform on [0, 1]' },
  { value: 'arcsine', label: 'arcsine on [0, 1]' },
] as const
const ALL = [
  ...BOUNDED,
  { value: 'exponential', label: 'exponential (unbounded)' },
  { value: 'pareto', label: 'Pareto α = 1.5 (infinite variance)' },
] as const
const FLOOR = 1e-5

// ── 1 · Tail bounds ──────────────────────────────────────────────────────────────────────────────────────────────────

export function TailBoundsSpecimen() {
  const state = useFigureState({
    law: row('1 · the mean of n variables', {
      law: choice(BOUNDED, 'bernoulli', { label: 'law' }),
      p: slider(0.01, 0.99, 0.1, { step: 0.01, label: 'Bernoulli p', when: (v: AnyValues) => v.law === 'bernoulli' }),
      n: int(50, { ge: 1, le: 5000, suggestions: [10, 50, 200, 1000], label: 'n' }),
      trials: int(20_000, { ge: 100, le: 200_000, suggestions: [5000, 20_000, 100_000], label: 'Monte Carlo trials' }),
      seed: int(1, { ge: 0, le: 9999, label: 'seed' }),
    }),
    probe: row('2 · deviation', { t: slider(0.01, 0.5, 0.1, { step: 0.005, label: 'deviation t' }) }),
  })
  const { law, p, n, trials, seed } = state.law
  const { t } = state.probe
  const study = useComputed(
    () => concentrationStudy(stream(`concentration/${seed}`), { law: law as SummandLaw, p, n, trials }),
    [law, p, n, trials, seed],
    { mode: 'release' },
  )
  const c = study.value
  const at = tailBounds([t], n, c.mean, c.variance)
  const empiricalAt = useMemo(() => {
    let k = 0
    for (const m of c.means) if (Math.abs(m - c.mean) >= t - 1e-12) k++
    return k / c.means.length
  }, [c, t])
  const tAxis = useAxis({ label: 'deviation t', range: [0, 0.5] })
  const pAxis = useAxis({ label: 'P(|X̄ₙ − μ| ≥ t)', log: true, range: [FLOOR, 1] })
  const mAxis = useAxis({ label: 'sample mean X̄ₙ', range: [0, 1] })
  const densityAxis = useAxis({ label: 'density', hold: 'union', key: `${law}/${p}/${n}` })
  const floor = (a: ArrayLike<number>) => Array.from(a, (v) => Math.max(FLOOR, v))
  return (
    <Figure
      title="Concentration inequalities against the truth"
      purpose="The mean of n bounded variables rarely strays far from μ; Chebyshev's bound falls like 1/(nt²), while Hoeffding's, Bernstein's and the relative-entropy Chernoff bound fall exponentially in n t², and the closer a bound uses the law (its variance, its mean), the tighter it sits on the simulated tail."
      state={state}
      defaultSize="L"
      readouts={{
        [`at t = ${fmt(t, 2)}`]: (
          <>
            <Readout label="simulated" value={fmt(empiricalAt)} />
            <Readout label="Chernoff (KL)" value={fmt(at.chernoff[0])} />
            <Readout label="Bernstein" value={fmt(at.bernstein[0])} />
            <Readout label="Hoeffding" value={fmt(at.hoeffding[0])} />
            <Readout label="Chebyshev" value={fmt(at.chebyshev[0])} />
          </>
        ),
        law: (
          <>
            <Readout label="μ" value={fmt(c.mean)} />
            <Readout label="σ²" value={fmt(c.variance)} />
          </>
        ),
      }}
      caption="aifn concentrationStudy draws the trials' sample means and counts how often |X̄ₙ − μ| ≥ t; tailBounds gives Chebyshev σ²/(nt²), Hoeffding 2e^{−2nt²}, Bernstein 2e^{−nt²/(2σ² + 2t/3)} and Chernoff e^{−n KL(μ+t‖μ)} + e^{−n KL(μ−t‖μ)} (KL between Bernoulli laws, valid for any law on [0, 1]). Left: tails on a log scale; the simulated tail ends where no trial deviated that far, and the bounds are floored at 10⁻⁵. Right: the sample means. With a small Bernoulli p the variance is small and Bernstein and Chernoff beat Hoeffding by orders of magnitude. Drag t on the left chart."
    >
      <Plots cols={2}>
        <Plot x={tAxis} y={pAxis} title="tail probability">
          <Curve
            name="simulated"
            x={c.t}
            y={Array.from(c.empirical, (v) => (v > 0 ? v : NaN))}
            emphasis
            showPoints
            stale={study.stale}
          />
          <Curve name="Chernoff (KL)" slot={0} x={c.t} y={floor(c.chernoff)} />
          <Curve name="Bernstein" slot={1} x={c.t} y={floor(c.bernstein)} />
          <Curve name="Hoeffding" slot={2} x={c.t} y={floor(c.hoeffding)} />
          <Curve name="Chebyshev" slot={3} x={c.t} y={floor(c.chebyshev)} dashed />
          <Handle kind="x" at={t} onDrag={(v) => state.set('probe.t', Math.max(0.01, Math.min(0.5, v)))} label="t" />
        </Plot>
        <Plot x={mAxis} y={densityAxis} title="sample means">
          <Histogram name="X̄ₙ" values={c.means} slot={0} normalize="density" />
        </Plot>
      </Plots>
    </Figure>
  )
}

// ── 2 · McDiarmid ────────────────────────────────────────────────────────────────────────────────────────────────────

export function McdiarmidSpecimen() {
  const state = useFigureState({
    bins: row('1 · balls into bins', {
      balls: int(100, { ge: 1, le: 5000, suggestions: [50, 100, 500], label: 'balls n' }),
      bins: int(100, { ge: 2, le: 2000, suggestions: [20, 100, 500], label: 'bins m' }),
      trials: int(5000, { ge: 100, le: 50_000, suggestions: [1000, 5000, 20_000], label: 'trials' }),
      seed: int(1, { ge: 0, le: 9999, label: 'seed' }),
    }),
  })
  const { balls, bins, trials, seed } = state.bins
  const study = useComputed(
    () => mcdiarmidStudy(stream(`mcdiarmid/${seed}`), { balls, bins, trials }),
    [balls, bins, trials, seed],
    { mode: 'release' },
  )
  const m = study.value
  const tAxis = useAxis({ label: 'deviation t', range: [0, 0.25] })
  const pAxis = useAxis({ label: 'P(|f − E f| ≥ t)', log: true, range: [1e-4, 1] })
  const fAxis = useAxis({ label: 'fraction of empty bins f', range: [0, 1] })
  const fDensityAxis = useAxis({ label: 'density', hold: 'union', key: `${balls}/${bins}` })
  return (
    <Figure
      title="McDiarmid's inequality beyond sums"
      purpose="The fraction of empty bins after n balls is not a sum of independent terms, but moving one ball changes it by at most 1/m, so McDiarmid's bounded-differences inequality gives P(|f − E f| ≥ t) ≤ 2 exp(−2t²m²/n)."
      state={state}
      defaultSize="M"
      readouts={{
        f: (
          <>
            <Readout label="E f = (1 − 1/m)ⁿ" value={fmt(m.expected)} />
            <Readout label="bounded difference cᵢ" value={fmt(1 / bins)} />
          </>
        ),
      }}
      caption="aifn mcdiarmidStudy throws n balls into m bins per trial. Left: the simulated tail (ink) against McDiarmid's bound, floored at 10⁻⁴. Right: the fractions of empty bins, centred near (1 − 1/m)ⁿ ≈ e^{−n/m}. The bound is loose by the factor the variance would buy (a Bernstein-type bound), but it needs nothing but the bounded differences."
    >
      <Plots cols={2}>
        <Plot x={tAxis} y={pAxis} title="tail probability">
          <Curve
            name="simulated"
            x={m.t}
            y={Array.from(m.empirical, (v) => Math.max(1e-4, v))}
            emphasis
            showPoints
            stale={study.stale}
          />
          <Curve name="McDiarmid" slot={0} x={m.t} y={Array.from(m.bound, (v) => Math.max(1e-4, v))} />
        </Plot>
        <Plot x={fAxis} y={fDensityAxis} title="fraction of empty bins">
          <Histogram name="f" values={m.values} slot={1} normalize="density" />
        </Plot>
      </Plots>
    </Figure>
  )
}

// ── 3 · LLN and CLT ──────────────────────────────────────────────────────────────────────────────────────────────────

export function LimitTheoremsSpecimen() {
  const state = useFigureState({
    law: row('1 · summands', {
      law: choice(ALL, 'exponential', { label: 'law' }),
      p: slider(0.01, 0.99, 0.1, { step: 0.01, label: 'Bernoulli p', when: (v: AnyValues) => v.law === 'bernoulli' }),
      n: int(10, { ge: 1, le: 2000, suggestions: [1, 2, 5, 10, 30, 100], label: 'n per sum' }),
      seed: int(1, { ge: 0, le: 9999, label: 'seed' }),
    }),
  })
  const { law, p, n, seed } = state.law
  const moments = lawMoments(law as SummandLaw, p)
  const paths = useComputed(() => runningMeans(stream(`lln/${seed}`), law as SummandLaw, 2000, 8, p), [law, p, seed], {
    mode: 'release',
  })
  const sums = useComputed(
    () => standardisedSums(stream(`clt/${seed}`), law as SummandLaw, n, 20_000, p),
    [law, p, n, seed],
    { mode: 'release' },
  )
  const steps = useMemo(() => Float64Array.from({ length: 2000 }, (_, i) => i + 1), [])
  const nAxis = useAxis({ label: 'n', range: [1, 2000], log: true })
  const meanAxis = useAxis({
    label: 'running mean X̄ₙ',
    range: [moments.mean - 2, moments.mean + 2],
    key: `${law}/${p}`,
  })
  const zAxis = useAxis({
    label: Number.isFinite(moments.variance) ? '√n (X̄ₙ − μ)/σ' : 'n^{1/3} (X̄ₙ − μ)',
    range: [-4, 4],
  })
  const normal = useMemo(() => Normal(0, 1), [])
  const zDensityAxis = useAxis({ label: 'density', range: [0, 0.8] })
  const clipped = useMemo(() => sums.value.filter((v) => v >= -4 && v <= 4), [sums.value])
  return (
    <Figure
      title="The law of large numbers and the central limit theorem"
      purpose="Running means settle on μ (the law of large numbers), and their fluctuations, scaled by √n/σ, approach the standard normal whatever the summands' law (the central limit theorem) — but only when the variance is finite: a Pareto law with α = 1.5 has a skewed, heavy-tailed stable limit instead."
      state={state}
      defaultSize="L"
      readouts={{
        law: (
          <>
            <Readout label="μ" value={fmt(moments.mean)} />
            <Readout label="σ²" value={Number.isFinite(moments.variance) ? fmt(moments.variance) : '∞'} />
            <Readout label="shown sums in [−4, 4]" value={`${clipped.length} / ${sums.value.length}`} />
          </>
        ),
      }}
      caption="aifn runningMeans (left, 8 paths of 2000 draws, log n axis, μ dashed) and standardisedSums (right, 20 000 sums of n draws, histogram against the N(0, 1) density). Raise n from 1: the exponential's skewed histogram turns normal by n ≈ 30, the Bernoulli's lattice fills in, and the Pareto's stays skewed with occasional huge values (its sums are scaled by n^{1/3}, the stable law's rate)."
    >
      <Plots cols={2}>
        <Plot x={nAxis} y={meanAxis} title="running means">
          <Curve name="μ" x={[1, 2000]} y={[moments.mean, moments.mean]} emphasis dashed />
          {Array.from({ length: 8 }, (_, r) => (
            <Curve key={r} name="paths" x={steps} y={paths.value.subarray(r * 2000, (r + 1) * 2000)} slot={0} thin />
          ))}
        </Plot>
        <Plot x={zAxis} y={zDensityAxis} title={`standardised sums, n = ${n}`}>
          <Histogram name="sums" values={clipped} slot={1} normalize="density" range={[-4, 4]} />
          <Density name="N(0, 1)" dist={normal} emphasis />
        </Plot>
      </Plots>
    </Figure>
  )
}
