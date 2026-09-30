import * as D from 'aifn/probability/distributions'
import * as Bij from 'aifn/probability/bijectors'
import { normalProjection } from 'aifn-applied/information/projection'
import { trapezoidSamples } from 'aifn/numerics/quadrature'
import { child, stream } from 'aifn/foundation/random'
import { linspace, tensor, toFlat, unwrap, type Value } from 'aifn/foundation/tensor'
import { useMemo, useState } from 'react'
import { Choice, defineVariants, ParamControls, Player, slider, Slider, useVariants } from '@lab/controls'
import { diverging, useTheme } from '@lab/design'
import { ControlRow, Figure, Tex } from '@lab/layout'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@lab/ui/table'
import { Toggle } from '@lab/ui/toggle'
import { Panel, Readout, Subplots, XYChart, type Handle, type XYSeries } from '@lab/viz'
import { formatValue } from '@lab/views'

/** A Value as numbers. */
const numbers = (v: Value): number[] => {
  const r = unwrap(v)
  return typeof r === 'number' ? [r] : toFlat(r)
}
const num = (v: Value) => numbers(v)[0]
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))
/** Non-finite values become gaps in lines. */
const gap = (v: number) => (Number.isFinite(v) ? v : NaN)
const grid = (lo: number, hi: number, n: number) => toFlat(linspace(lo, hi, n))
const fmt = (v: number) => (v === Infinity ? '∞' : formatValue(Number(v.toPrecision(4))))
const texNum = (v: number) => formatValue(Number(v.toPrecision(3)))

/** A divergence or entropy with how it was found, e.g. "0.2258 (quadrature)". */
function methodValue(r: D.MethodResult): string {
  if (r.method === 'quadrature' && !r.converged)
    // A diverging integral (e.g. a Cauchy-like p against a normal q) leaves quadrature with a huge, unconverged value.
    return Math.abs(r.value) > 1e6 ? '∞ (quadrature diverges)' : `${fmt(r.value)} (quadrature, not converged)`
  return `${fmt(r.value)} (${r.method})`
}

// ── Figure 1: the expected log-ratio ────────────────────────────────────────────────────────────────────────────────

const FAMILIES = defineVariants({
  normal: {
    label: 'Normal',
    params: { mu: slider(-4, 4, 0, { label: 'mean μ' }), sigma: slider(0.2, 3, 1, { label: 'sd σ' }) },
    f: (_: number, p): D.Univariate => D.Normal(p.mu, p.sigma),
  },
  laplace: {
    label: 'Laplace',
    params: { loc: slider(-4, 4, 0, { label: 'location μ' }), b: slider(0.2, 3, 1, { label: 'scale b' }) },
    f: (_: number, p): D.Univariate => D.Laplace(p.loc, p.b),
  },
  studentT: {
    label: 'Student t',
    params: {
      df: slider(1, 30, 3, { label: 'degrees of freedom ν' }),
      loc: slider(-4, 4, 0, { label: 'location μ' }),
      scale: slider(0.2, 3, 1, { label: 'scale σ' }),
    },
    f: (_: number, p): D.Univariate => D.StudentT(p.df, p.loc, p.scale),
  },
  gamma: {
    label: 'Gamma',
    params: { shape: slider(0.5, 10, 2, { label: 'shape k' }), rate: slider(0.2, 4, 1, { label: 'rate λ' }) },
    f: (_: number, p): D.Univariate => D.Gamma(p.shape, p.rate),
  },
  beta: {
    label: 'Beta',
    params: { a: slider(0.3, 8, 2, { label: 'α' }), b: slider(0.3, 8, 3, { label: 'β' }) },
    f: (_: number, p): D.Univariate => D.Beta(p.a, p.b),
  },
  mixture: {
    label: 'Two-component normal mixture',
    params: {
      w: slider(0, 1, 0.5, { label: 'weight w of the left component' }),
      m1: slider(-4, 4, -1.5, { label: 'left mean μ₁' }),
      m2: slider(-4, 4, 1.5, { label: 'right mean μ₂' }),
      sd: slider(0.2, 2, 0.6, { label: 'component sd σ' }),
    },
    f: (_: number, p): D.Univariate => D.Mixture([p.w, 1 - p.w], [D.Normal(p.m1, p.sd), D.Normal(p.m2, p.sd)]),
  },
})

const FAMILY_TEX: Record<string, (p: Record<string, number>) => string> = {
  normal: (p) => String.raw`\mathcal{N}(${texNum(p.mu)}, ${texNum(p.sigma)}^2)`,
  laplace: (p) => String.raw`\mathrm{Laplace}(${texNum(p.loc)}, ${texNum(p.b)})`,
  studentT: (p) => String.raw`t_{${texNum(p.df)}}(${texNum(p.loc)}, ${texNum(p.scale)})`,
  gamma: (p) => String.raw`\mathrm{Gamma}(${texNum(p.shape)}, ${texNum(p.rate)})`,
  beta: (p) => String.raw`\mathrm{Beta}(${texNum(p.a)}, ${texNum(p.b)})`,
  mixture: (p) =>
    String.raw`${texNum(p.w)}\,\mathcal{N}(${texNum(p.m1)}, ${texNum(p.sd)}^2) + ${texNum(1 - p.w)}\,\mathcal{N}(${texNum(p.m2)}, ${texNum(p.sd)}^2)`,
}

/** The x window: both distributions' central 99.8%, clipped to the union of their supports. */
function windowOf(p: D.Univariate, q: D.Univariate): [number, number] {
  const ends = (d: D.Univariate) => {
    const set = Bij.supportInterval(d.support)
    return [Math.max(set.lower, num(d.quantile(0.001))), Math.min(set.upper, num(d.quantile(0.999)))]
  }
  const [a, b] = ends(p)
  const [c, e] = ends(q)
  const lo = Math.min(a, c)
  const hi = Math.max(b, e)
  const pad = 0.04 * (hi - lo)
  const bounds = [Bij.supportInterval(p.support), Bij.supportInterval(q.support)]
  return [
    Math.max(lo - pad, Math.min(...bounds.map((s) => s.lower))),
    Math.min(hi + pad, Math.max(...bounds.map((s) => s.upper))),
  ]
}

const DRAWS = ['100', '1000', '10000'] as const

/**
 * KL(p ‖ q) as the area under p log(p/q), after the site's KL note: two densities chosen from six families, the
 * integrand of the chosen direction split into its positive and negative parts, and the numbers that go with it
 * (both directions, a Monte Carlo estimate, the cross-entropy identity and the symmetric Jensen–Shannon divergence).
 */
function ExpectedLogRatio() {
  const pv = useVariants(FAMILIES, { key: 'mixture' })
  const qv = useVariants(FAMILIES, {
    state: { key: 'normal', values: { normal: { mu: 0.5, sigma: 1.6 } }, shared: {} },
  })
  const [reverse, setReverse] = useState(false)
  const [draws, setDraws] = useState<(typeof DRAWS)[number]>('1000')
  const [seed, setSeed] = useState(1)
  const { resolved: mode } = useTheme()
  const ramp = diverging(mode)
  const negative = ramp[1]
  const positive = ramp[ramp.length - 2]

  const pair = useMemo(() => {
    try {
      return { ok: true as const, p: pv.f!(0), q: qv.f!(0) }
    } catch (e) {
      return { ok: false as const, error: (e as Error).message }
    }
  }, [pv, qv])

  // The first argument of the chosen direction, a, against the second, b.
  const [aName, bName] = reverse ? ['q', 'p'] : ['p', 'q']
  const klLabel = `KL(${aName} ‖ ${bName})`

  const numbersOf = useMemo(() => {
    if (!pair.ok) return null
    const { p, q } = pair
    const [a, b] = reverse ? [q, p] : [p, q]
    const safe = <T,>(f: () => T): T | null => {
      try {
        return f()
      } catch {
        return null
      }
    }
    return {
      pq: D.klAuto(p, q),
      qp: D.klAuto(q, p),
      entropy: D.entropyAuto(a),
      cross: D.crossEntropyAuto(a, b),
      js: D.jensenShannonNumerical(p, q),
      mc: safe(() => D.klMonteCarloWithError(stream(`kl/${seed}`), a, b, Number(draws))),
    }
  }, [pair, reverse, seed, draws])

  const curves = useMemo(() => {
    if (!pair.ok) return null
    const { p, q } = pair
    const [lo, hi] = windowOf(p, q)
    const xs = grid(lo, hi, 801)
    const pd = numbers(p.prob(tensor(xs))).map(gap)
    const qd = numbers(q.prob(tensor(xs))).map(gap)
    const [a, b] = reverse ? [q, p] : [p, q]
    const f = numbers(D.klIntegrand(a, b, tensor(xs)))
    const up = f.map((v) => (v > 0 ? gap(v) : 0))
    const down = f.map((v) => (v < 0 ? v : 0))
    const infinite = f.some((v) => v === Infinity)
    return {
      xs,
      pd,
      qd,
      up,
      down,
      infinite,
      upArea: trapezoidSamples(
        up.map((v) => (Number.isFinite(v) ? v : 0)),
        xs,
      ),
      downArea: trapezoidSamples(down, xs),
    }
  }, [pair, reverse])

  const densities = useMemo(
    (): XYSeries[] =>
      curves
        ? [
            { name: 'p(x)', type: 'line', x: curves.xs, y: curves.pd, slot: 0, area: true },
            { name: 'q(x)', type: 'line', x: curves.xs, y: curves.qd, slot: 1 },
          ]
        : [],
    [curves],
  )
  const net = numbersOf ? (reverse ? numbersOf.qp : numbersOf.pq) : null
  const integrand = useMemo((): XYSeries[] => {
    if (!curves || !net) return []
    const [lo, hi] = [curves.xs[0], curves.xs[curves.xs.length - 1]]
    return [
      {
        name: `${aName} log(${aName}/${bName}) > 0: area +${fmt(curves.upArea)}`,
        type: 'area',
        x: curves.xs,
        y: curves.up,
        color: positive,
      },
      {
        name: `${aName} log(${aName}/${bName}) < 0: area ${fmt(curves.downArea)}`,
        type: 'area',
        x: curves.xs,
        y: curves.down,
        color: negative,
      },
      { name: `net area = ${klLabel} = ${fmt(net.value)}`, type: 'line', x: [lo, hi], y: [0, 0], muted: true },
    ]
  }, [curves, net, aName, bName, klLabel, positive, negative])

  const familyOptions = (Object.keys(FAMILIES.specs) as (keyof typeof FAMILIES.specs)[]).map((k) => ({
    value: k,
    label: FAMILIES.specs[k].label,
  }))
  const row = (v: typeof pv, name: 'p' | 'q', index: number) => (
    <ControlRow label={`${index} · ${name}`}>
      <div className="flex flex-col gap-1">
        <Choice label={`family of ${name}`} value={v.key} onChange={v.setKey} options={familyOptions} />
        <Tex className="text-sm">{`${name} = ${FAMILY_TEX[v.key](v.values as Record<string, number>)}`}</Tex>
      </div>
      <ParamControls defs={FAMILIES.specs[v.key].params} values={v.values} set={v.set} />
    </ControlRow>
  )
  const controls = (
    <>
      {row(pv, 'p', 1)}
      {row(qv, 'q', 2)}
      <ControlRow label="3 · direction and Monte Carlo">
        <div className="flex flex-col gap-1">
          <Toggle
            variant="outline"
            pressed={reverse}
            onPressedChange={setReverse}
            className="w-fit aria-pressed:border-primary aria-pressed:bg-primary aria-pressed:text-primary-foreground"
          >
            {reverse ? 'Reverse: KL(q ‖ p)' : 'Forward: KL(p ‖ q)'}
          </Toggle>
          <Tex className="text-sm">
            {reverse
              ? String.raw`\mathrm{KL}(q \,\|\, p) = \int q(x) \log \frac{q(x)}{p(x)}\, dx`
              : String.raw`\mathrm{KL}(p \,\|\, q) = \int p(x) \log \frac{p(x)}{q(x)}\, dx`}
          </Tex>
        </div>
        <Choice label="Monte Carlo draws" value={draws} onChange={setDraws} options={DRAWS} />
        <Slider label="seed" value={seed} min={1} max={20} step={1} onChange={setSeed} />
      </ControlRow>
    </>
  )

  const readouts = numbersOf ? (
    <>
      <Readout label="KL(p ‖ q)" value={methodValue(numbersOf.pq)} />
      <Readout label="KL(q ‖ p)" value={methodValue(numbersOf.qp)} />
      <Readout
        label={`Monte Carlo ${klLabel}, ${draws} draws from ${aName}`}
        value={
          numbersOf.mc
            ? `${fmt(numbersOf.mc.value)} ± ${fmt(numbersOf.mc.standardError)} (standard error)`
            : 'no draws for this family'
        }
      />
      <Readout label={`H(${aName})`} value={methodValue(numbersOf.entropy)} />
      <Readout label={`cross-entropy H(${aName}, ${bName})`} value={methodValue(numbersOf.cross)} />
      {net && <Readout label={`H(${aName}) + ${klLabel}`} value={fmt(numbersOf.entropy.value + net.value)} />}
      <Readout label="Jensen–Shannon JS(p, q) (symmetric, ≤ log 2)" value={`${fmt(numbersOf.js.value)} (quadrature)`} />
    </>
  ) : undefined

  return (
    <Figure
      title="KL divergence is an expected log-ratio"
      description="KL(p ‖ q) = E_p[log p(X) − log q(X)]: the area under p(x) log(p(x)/q(x))."
      defaultSize="L"
      controls={controls}
      readouts={readouts}
      caption={
        <>
          The top panel shows the densities p (filled) and q. The bottom panel shows the integrand of the chosen
          direction: red where the first argument is more probable than the second (log-ratio above 0), blue where it is
          less. The net area is the KL divergence, and it is never negative: the blue area never outweighs the red, and
          both vanish only when p = q. Turning the direction toggle swaps the roles of p and q and changes the value,
          because KL is not symmetric; the Jensen–Shannon divergence is. Where q ≈ 0 while p &gt; 0 (a narrow or
          light-tailed q under a wide or heavy-tailed p, or q&apos;s support missing part of p&apos;s, e.g. a Gamma or
          Beta q under a Normal p) the log-ratio grows without bound and KL(p ‖ q) blows up, to ∞ when the supports
          differ. Pairs with a closed form (Normal, Gamma or Beta with itself) are marked closed form; the others are
          integrated numerically. The Monte Carlo estimate averages the log-ratio over draws from the first argument,
          with its standard error. H(p, q) = H(p) + KL(p ‖ q) checks the cross-entropy identity. The axes hold while
          parameters move and refit when a family or the direction changes.
        </>
      }
    >
      {!pair.ok ? (
        <div className="text-sm text-muted-foreground">{pair.error}</div>
      ) : (
        <Subplots
          rows={2}
          sharex
          heightRatios={[1, 1]}
          hoverGroup
          rescaleOnChange={false}
          holdFit="union"
          axisKey={`${pv.key}|${qv.key}|${reverse}`}
        >
          <Panel>
            <XYChart series={densities} xLabel="x" yLabel="density" />
          </Panel>
          <Panel>
            <XYChart
              series={integrand}
              xLabel="x"
              yLabel={`${aName} log(${aName}/${bName})${curves?.infinite ? ' (∞ where q = 0)' : ''}`}
            />
          </Panel>
        </Subplots>
      )}
    </Figure>
  )
}

// ── Figure 2: forward against reverse ───────────────────────────────────────────────────────────────────────────────

const X_RANGE: [number, number] = [-8, 8]

/**
 * The best single normal for a bimodal mixture under each direction of KL, both fitted by L-BFGS with autodiff
 * gradients (`normalProjection` from aifn/info), the reverse fit from a start the reader drags.
 */
function ForwardReverse() {
  const [separation, setSeparation] = useState(4)
  const [weight, setWeight] = useState(0.5)
  const [sd, setSd] = useState(0.7)
  const [start, setStart] = useState(1)
  // The reverse fit's optimiser step on show; null follows the last step.
  const [stepWanted, setStep] = useState<number | null>(null)

  const p = useMemo(
    () => D.Mixture([weight, 1 - weight], [D.Normal(-separation / 2, sd), D.Normal(separation / 2, sd)]),
    [weight, separation, sd],
  )
  const fits = useMemo(() => {
    const init = { loc: start, scale: 1 }
    const forward = normalProjection(p, 'forward', { init })
    const reverse = normalProjection(p, 'reverse', { init })
    const qf = D.Normal(forward.loc, forward.scale)
    const qr = D.Normal(reverse.loc, reverse.scale)
    return {
      forward,
      reverse,
      qf,
      qr,
      moments: { mean: num(p.mean()), sd: num(p.stddev()) },
      table: {
        ff: D.klAuto(p, qf),
        fr: D.klAuto(qf, p),
        rf: D.klAuto(p, qr),
        rr: D.klAuto(qr, p),
      },
    }
  }, [p, start])
  const path = fits.reverse.path
  const step = stepWanted === null ? path.length - 1 : clamp(stepWanted, 0, path.length - 1)
  const at = path[step]

  const xs = useMemo(() => grid(X_RANGE[0], X_RANGE[1], 641), [])
  const series = useMemo(
    (): XYSeries[] => [
      { name: 'p (mixture)', type: 'line', x: xs, y: numbers(p.prob(tensor(xs))), slot: 0, area: true },
      {
        name: 'q minimising KL(p ‖ q): forward, mode covering',
        type: 'line',
        x: xs,
        y: numbers(fits.qf.prob(tensor(xs))),
        slot: 1,
      },
      {
        name: 'q minimising KL(q ‖ p): reverse, mode seeking',
        type: 'line',
        x: xs,
        y: numbers(fits.qr.prob(tensor(xs))),
        slot: 2,
      },
      {
        name: 'start N(μ₀, 1)',
        type: 'line',
        x: xs,
        y: numbers(D.Normal(start, 1).prob(tensor(xs))),
        muted: true,
        dashed: true,
      },
    ],
    [xs, p, fits, start],
  )
  const live = useMemo(
    (): XYSeries[] => [
      {
        name: 'reverse fit at the step shown',
        type: 'line',
        x: xs,
        y: numbers(D.Normal(at.loc, at.scale).prob(tensor(xs))),
        slot: 2,
        dashed: true,
        thin: true,
      },
    ],
    [xs, at],
  )
  const handles = useMemo(
    (): Handle[] => [
      {
        kind: 'x',
        at: start,
        label: 'start μ₀',
        onDrag: (x) => {
          setStart(clamp(x, -6, 6))
          setStep(null)
        },
      },
    ],
    [start],
  )
  const { forward, reverse, table, moments } = fits

  return (
    <Figure
      title="Forward against reverse KL: mode covering and mode seeking"
      description="The best single normal q for a bimodal p depends on which KL is minimised."
      defaultSize="L"
      controls={
        <>
          <ControlRow label="1 · p, an equal-width two-component normal mixture">
            <Slider label="separation of the means" value={separation} min={0} max={8} onChange={setSeparation} />
            <Slider label="weight of the left component" value={weight} min={0.05} max={0.95} onChange={setWeight} />
            <Slider label="component sd" value={sd} min={0.3} max={1.5} onChange={setSd} />
          </ControlRow>
          <ControlRow label="2 · the fits (L-BFGS on (μ, log σ), autodiff gradients)">
            <Slider
              label="start μ₀ (σ₀ = 1)"
              value={start}
              min={-6}
              max={6}
              onChange={(v) => {
                setStart(v)
                setStep(null)
              }}
            />
            <Player
              label="reverse-fit step"
              value={step}
              onChange={setStep}
              count={path.length}
              format={(i) => `${i} / ${path.length - 1}`}
            />
          </ControlRow>
        </>
      }
      readouts={
        <>
          <Readout
            label="forward fit"
            value={`N(${fmt(forward.loc)}, ${fmt(forward.scale)}²), ${forward.path.length - 1} steps`}
          />
          <Readout label="moment match E_p[X], sd_p[X]" value={`${fmt(moments.mean)}, ${fmt(moments.sd)}`} />
          <Readout
            label="reverse fit"
            value={`N(${fmt(reverse.loc)}, ${fmt(reverse.scale)}²), ${reverse.path.length - 1} steps`}
          />
          <Readout
            label={`reverse step ${step}`}
            value={`N(${fmt(at.loc)}, ${fmt(at.scale)}²), KL(q ‖ p) ${fmt(at.objective)}`}
          />
          <Readout label="at the forward fit: KL(p ‖ q)" value={methodValue(table.ff)} />
          <Readout label="KL(q ‖ p)" value={methodValue(table.fr)} />
          <Readout label="at the reverse fit: KL(p ‖ q)" value={methodValue(table.rf)} />
          <Readout label="KL(q ‖ p)" value={methodValue(table.rr)} />
        </>
      }
      caption={
        <>
          Drag the start μ₀ on the chart (or use its slider): both fits begin at N(μ₀, 1). Minimising the forward KL(p ‖
          q) = E_p[log p − log q] is maximum likelihood in the limit of many draws from p: q is charged wherever p has
          mass and q does not, so it spreads to cover both modes. Over normals the forward fit is moment matching, and
          it lands on the mean and sd of p from any start (compare the moment-match readout). Minimising the reverse
          KL(q ‖ p) = E_q[log q − log p], the objective of variational inference, charges q only where q itself has
          mass: putting mass between the modes, where p is small, costs a lot, while ignoring a mode costs nothing. The
          reverse fit therefore seeks one mode, and which one depends on the start; with modes far apart it sits on one
          mode with KL(q ‖ p) near −log of that mode&apos;s weight. Each fit has the lower value of its own objective
          (compare the four KL readouts). The player steps through the reverse fit&apos;s L-BFGS iterates, drawn dashed.
          Starting exactly at the centre of a symmetric p stalls at the saddle between the modes.
        </>
      }
    >
      <XYChart
        series={series}
        live={live}
        handles={handles}
        xRange={X_RANGE}
        xLabel="x"
        yLabel="density"
        rescaleOnChange={false}
        holdFit="union"
      />
    </Figure>
  )
}

// ── Figure 3: the closed forms against quadrature and Monte Carlo ───────────────────────────────────────────────────

const PAIRS: { label: string; p: D.Distribution; q: D.Distribution; continuous: boolean }[] = [
  { label: 'N(1, 2²) ‖ N(0, 1.5²)', p: D.Normal(1, 2), q: D.Normal(0, 1.5), continuous: true },
  { label: 'LogNormal(0.2, 0.7) ‖ LogNormal(0, 1)', p: D.LogNormal(0.2, 0.7), q: D.LogNormal(0, 1), continuous: true },
  { label: 'Beta(2, 3) ‖ Beta(1.5, 0.7)', p: D.Beta(2, 3), q: D.Beta(1.5, 0.7), continuous: true },
  { label: 'Gamma(2, 3) ‖ Gamma(1.5, 0.7)', p: D.Gamma(2, 3), q: D.Gamma(1.5, 0.7), continuous: true },
  { label: 'Exponential(2) ‖ Exponential(3.5)', p: D.Exponential(2), q: D.Exponential(3.5), continuous: true },
  { label: 'Poisson(2) ‖ Poisson(3.5)', p: D.Poisson(2), q: D.Poisson(3.5), continuous: false },
  { label: 'Bernoulli(0.3) ‖ Bernoulli(0.6)', p: D.Bernoulli(0.3), q: D.Bernoulli(0.6), continuous: false },
  {
    label: 'Categorical(0.2, 0.5, 0.3) ‖ Categorical(0.4, 0.4, 0.2)',
    p: D.Categorical(tensor([0.2, 0.5, 0.3])),
    q: D.Categorical(tensor([0.4, 0.4, 0.2])),
    continuous: false,
  },
  {
    label: 'Dirichlet(2, 3, 1.5) ‖ Dirichlet(1, 1, 4)',
    p: D.Dirichlet(tensor([2, 3, 1.5])),
    q: D.Dirichlet(tensor([1, 1, 4])),
    continuous: false,
  },
]

/** Each registered closed form against quadrature (continuous univariate pairs) and a Monte Carlo estimate. */
function ClosedFormPairs() {
  const [draws, setDraws] = useState<(typeof DRAWS)[number]>('10000')
  const [seed, setSeed] = useState(1)
  const rows = useMemo(
    () =>
      PAIRS.map(({ label, p, q, continuous }, i) => {
        const closed = num(D.kl(p, q))
        const quad = continuous ? D.klNumerical(p, q).value : null
        const mc = D.klMonteCarloWithError(child(stream(`kl-pairs/${seed}`), i), p, q, Number(draws))
        return { label, closed, quad, mc, z: (mc.value - closed) / mc.standardError }
      }),
    [draws, seed],
  )
  return (
    <Figure
      title="Closed-form KL pairs against quadrature and Monte Carlo"
      description="Every registered closed form, checked two independent ways."
      controls={
        <>
          <Choice label="Monte Carlo draws" value={draws} onChange={setDraws} options={DRAWS} />
          <Slider label="seed" value={seed} min={1} max={20} step={1} onChange={setSeed} />
        </>
      }
      caption="kl(p, q) dispatches on the two family names to a closed form. For continuous univariate pairs, quadrature of p log(p/q) should agree to many digits. The Monte Carlo estimate should fall within a few standard errors of the closed form: the z column is (Monte Carlo − closed form) / standard error, and |z| above about 3 would point at a wrong formula."
    >
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>p ‖ q</TableHead>
            <TableHead className="text-right">closed form</TableHead>
            <TableHead className="text-right">quadrature</TableHead>
            <TableHead className="text-right">Monte Carlo ± se</TableHead>
            <TableHead className="text-right">z</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((r) => (
            <TableRow key={r.label}>
              <TableCell>{r.label}</TableCell>
              <TableCell className="text-right font-mono tabular-nums">{fmt(r.closed)}</TableCell>
              <TableCell className="text-right font-mono tabular-nums">
                {r.quad === null ? '—' : formatValue(Number(r.quad.toPrecision(8)))}
              </TableCell>
              <TableCell className="text-right font-mono tabular-nums">
                {fmt(r.mc.value)} ± {fmt(r.mc.standardError)}
              </TableCell>
              <TableCell className="text-right font-mono tabular-nums">{r.z.toFixed(2)}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Figure>
  )
}

export function KlSpecimen() {
  return (
    <>
      <ExpectedLogRatio />
      <ForwardReverse />
      <ClosedFormPairs />
    </>
  )
}
