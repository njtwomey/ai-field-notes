import * as D from 'aifn/probability/distributions'
import * as Bij from 'aifn/probability/bijectors'
import { normalProjection } from 'aifn-methods/information/projection'
import { trapezoidSamples } from 'aifn/numerics/quadrature'
import { child, stream } from 'aifn/foundation/random'
import { linspace, tensor, toFlat, unwrap, type Value } from 'aifn/foundation/tensor'
import { useMemo, useState } from 'react'
import { Player } from '@lab/controls'
import { Equation, Figure, live, Tex, tex } from '@lab/layout'
import { choice, row, slider, toggle, useComputed, useFigureState, variants } from '@lab/state'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@lab/ui/table'
import { Area, Curve, Handle, Plot, Plots, Readout, SignedArea, useAxis } from '@lab/viz'
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

/** A case's slider values. */
type Num = Record<string, number>

/** The six families as variants cases; `normal` sets the normal's initial mean and sd (p and q differ). */
const families = (normal: { mu: number; sigma: number }) => ({
  normal: {
    label: 'Normal',
    params: {
      mu: slider(-4, 4, normal.mu, { label: 'mean μ' }),
      sigma: slider(0.2, 3, normal.sigma, { label: 'sd σ' }),
    },
  },
  laplace: {
    label: 'Laplace',
    params: { loc: slider(-4, 4, 0, { label: 'location μ' }), b: slider(0.2, 3, 1, { label: 'scale b' }) },
  },
  studentT: {
    label: 'Student t',
    params: {
      df: slider(1, 30, 3, { label: 'degrees of freedom ν' }),
      loc: slider(-4, 4, 0, { label: 'location μ' }),
      scale: slider(0.2, 3, 1, { label: 'scale σ' }),
    },
  },
  gamma: {
    label: 'Gamma',
    params: { shape: slider(0.5, 10, 2, { label: 'shape k' }), rate: slider(0.2, 4, 1, { label: 'rate λ' }) },
  },
  beta: {
    label: 'Beta',
    params: { a: slider(0.3, 8, 2, { label: 'α' }), b: slider(0.3, 8, 3, { label: 'β' }) },
  },
  mixture: {
    label: 'Two-component normal mixture',
    params: {
      w: slider(0, 1, 0.5, { label: 'weight w of the left component' }),
      m1: slider(-4, 4, -1.5, { label: 'left mean μ₁' }),
      m2: slider(-4, 4, 1.5, { label: 'right mean μ₂' }),
      sd: slider(0.2, 2, 0.6, { label: 'component sd σ' }),
    },
  },
})

/** Each family's distribution from its values. */
const MAKE: Record<string, (p: Num) => D.Univariate> = {
  normal: (p) => D.Normal(p.mu, p.sigma),
  laplace: (p) => D.Laplace(p.loc, p.b),
  studentT: (p) => D.StudentT(p.df, p.loc, p.scale),
  gamma: (p) => D.Gamma(p.shape, p.rate),
  beta: (p) => D.Beta(p.a, p.b),
  mixture: (p) => D.Mixture([p.w, 1 - p.w], [D.Normal(p.m1, p.sd), D.Normal(p.m2, p.sd)]),
}

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

const DRAWS = [100, 1000, 10000] as const

/**
 * KL(p ‖ q) as the area under p log(p/q), after the site's KL note: two densities chosen from six families, the
 * integrand of the chosen direction split into its positive and negative parts, and the numbers that go with it
 * (both directions, a Monte Carlo estimate, the cross-entropy identity and the symmetric Jensen–Shannon divergence).
 */
function ExpectedLogRatio() {
  const state = useFigureState({
    p: variants(families({ mu: 0, sigma: 1 }), { label: '1 · p', choiceLabel: 'family of p', initial: 'mixture' }),
    q: variants(families({ mu: 0.5, sigma: 1.6 }), { label: '2 · q', choiceLabel: 'family of q', initial: 'normal' }),
    reveal: row('3 · direction and Monte Carlo', {
      reverse: toggle(false, 'reverse: KL(q ‖ p)'),
      draws: choice(DRAWS, 1000, { label: 'Monte Carlo draws' }),
      seed: slider(1, 20, 1, { step: 1, label: 'seed' }),
    }),
  })
  const pv = state.p
  const qv = state.q
  const { reverse, draws, seed } = state.reveal

  const pair = useMemo(() => {
    try {
      return { ok: true as const, p: MAKE[pv.key](pv.values as Num), q: MAKE[qv.key](qv.values as Num) }
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
      mc: safe(() => D.klMonteCarloWithError(stream(`kl/${seed}`), a, b, draws)),
    }
  }, [pair, reverse, seed, draws])

  const curves = useMemo(() => {
    if (!pair.ok) return null
    const { p, q } = pair
    const [lo, hi] = windowOf(p, q)
    const xs = grid(lo, hi, 801)
    const [a, b] = reverse ? [q, p] : [p, q]
    const f = numbers(D.klIntegrand(a, b, tensor(xs)))
    return {
      xs,
      pd: numbers(p.prob(tensor(xs))).map(gap),
      qd: numbers(q.prob(tensor(xs))).map(gap),
      f: f.map(gap),
      infinite: f.some((v) => v === Infinity),
      upArea: trapezoidSamples(
        f.map((v) => (v > 0 && Number.isFinite(v) ? v : 0)),
        xs,
      ),
      downArea: trapezoidSamples(
        f.map((v) => (v < 0 ? v : 0)),
        xs,
      ),
    }
  }, [pair, reverse])
  const net = numbersOf ? (reverse ? numbersOf.qp : numbersOf.pq) : null
  const axisKey = `${pv.key}|${qv.key}|${reverse}`
  const x = useAxis({ label: 'x', hold: 'union', key: axisKey })
  const yd = useAxis({ label: 'density', hold: 'union', key: axisKey })
  const yf = useAxis({
    label: `${aName} log(${aName}/${bName})${curves?.infinite ? ' (∞ where q = 0)' : ''}`,
    hold: 'union',
    key: axisKey,
  })

  return (
    <Figure
      title="KL divergence is an expected log-ratio"
      purpose="KL(p ‖ q) = E_p[log p(X) − log q(X)] is the net signed area under p(x) log(p(x)/q(x)); it is never negative, zero only when p = q, and not symmetric."
      description={
        <span className="flex flex-wrap gap-x-6 text-base">
          <Tex>{`p = ${FAMILY_TEX[pv.key](pv.values as Num)}`}</Tex>
          <Tex>{`q = ${FAMILY_TEX[qv.key](qv.values as Num)}`}</Tex>
        </span>
      }
      defaultSize="L"
      state={state}
      equation={
        curves && net ? (
          <Equation>
            {tex`\mathrm{KL}(${aName} \,\|\, ${bName}) = \int ${aName}(x) \log \frac{${aName}(x)}{${bName}(x)}\, dx = ${live(curves.upArea, { digits: 4 })} ${live(curves.downArea, { digits: 4 })} = ${live(net.value, { digits: 4, strong: true })}`}
          </Equation>
        ) : undefined
      }
      readouts={
        numbersOf
          ? {
              'both directions': (
                <>
                  <Readout label="KL(p ‖ q)" value={methodValue(numbersOf.pq)} />
                  <Readout label="KL(q ‖ p)" value={methodValue(numbersOf.qp)} />
                  <Readout
                    label="Jensen–Shannon JS(p, q) (symmetric, ≤ log 2)"
                    value={`${fmt(numbersOf.js.value)} (quadrature)`}
                  />
                </>
              ),
              [`checks of ${klLabel}`]: (
                <>
                  <Readout
                    label={`Monte Carlo, ${draws} draws from ${aName}`}
                    value={
                      numbersOf.mc
                        ? `${fmt(numbersOf.mc.value)} ± ${fmt(numbersOf.mc.standardError)} (standard error)`
                        : 'no draws for this family'
                    }
                  />
                  <Readout label={`H(${aName})`} value={methodValue(numbersOf.entropy)} />
                  <Readout label={`cross-entropy H(${aName}, ${bName})`} value={methodValue(numbersOf.cross)} />
                  {net && (
                    <Readout label={`H(${aName}) + ${klLabel}`} value={fmt(numbersOf.entropy.value + net.value)} />
                  )}
                </>
              ),
            }
          : undefined
      }
      caption={
        <>
          The top panel shows the densities p (filled) and q. The bottom panel shows the integrand of the chosen
          direction: one colour where the first argument is more probable than the second (log-ratio above 0), the other
          where it is less. The net area is the KL divergence, and it is never negative. The reveal toggle swaps the
          roles of p and q and changes the value, because KL is not symmetric; the Jensen–Shannon divergence is. Where q
          ≈ 0 while p &gt; 0 (a narrow or light-tailed q under a wide or heavy-tailed p, or q&apos;s support missing
          part of p&apos;s, e.g. a Gamma or Beta q under a Normal p) the log-ratio grows without bound and KL(p ‖ q)
          blows up, to ∞ when the supports differ. Pairs with a closed form (Normal, Gamma or Beta with itself) are
          marked closed form; the others are integrated numerically. H(p, q) = H(p) + KL(p ‖ q) checks the cross-entropy
          identity. The axes hold while parameters move and refit when a family or the direction changes.
        </>
      }
    >
      {!pair.ok || !curves ? (
        <div className="text-sm text-muted-foreground">{pair.ok ? '' : pair.error}</div>
      ) : (
        <Plots rows={2} heights={[1, 1]} hoverGroup>
          <Plot x={x} y={yd}>
            <Area name="p(x)" x={curves.xs} y={curves.pd} slot={0} />
            <Curve name="q(x)" x={curves.xs} y={curves.qd} slot={1} />
          </Plot>
          <Plot x={x} y={yf}>
            <SignedArea name={`${aName} log(${aName}/${bName})`} x={curves.xs} y={curves.f} label={klLabel} />
          </Plot>
        </Plots>
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
  const state = useFigureState({
    p: row('1 · p, an equal-width two-component normal mixture', {
      separation: slider(0, 8, 4, { label: 'separation of the means' }),
      weight: slider(0.05, 0.95, 0.5, { label: 'weight of the left component' }),
      sd: slider(0.3, 1.5, 0.7, { label: 'component sd' }),
    }),
    fits: row('2 · the fits (L-BFGS on (μ, log σ), autodiff gradients)', {
      start: slider(-6, 6, 1, { label: 'start μ₀ (σ₀ = 1)' }),
    }),
  })
  const { separation, weight, sd } = state.p
  const start = state.fits.start
  // The reverse fit's optimiser step on show.
  const [stepWanted, setStep] = useState(0)

  const p = useMemo(
    () => D.Mixture([weight, 1 - weight], [D.Normal(-separation / 2, sd), D.Normal(separation / 2, sd)]),
    [weight, separation, sd],
  )
  // Two L-BFGS fits per change, too slow for every pointer move: they follow on release, while the start curve moves
  // with the pointer and the old fits are drawn faded.
  const fitsComputed = useComputed(
    () => {
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
    },
    [p, start],
    { mode: 'release' },
  )
  const fits = fitsComputed.value
  const stale = fitsComputed.stale
  const path = fits.reverse.path
  const step = clamp(stepWanted, 0, path.length - 1)
  const at = path[step]

  const xs = useMemo(() => grid(X_RANGE[0], X_RANGE[1], 641), [])
  const curves = useMemo(
    () => ({
      p: numbers(p.prob(tensor(xs))),
      forward: numbers(fits.qf.prob(tensor(xs))),
      reverse: numbers(fits.qr.prob(tensor(xs))),
    }),
    [xs, p, fits],
  )
  const startCurve = useMemo(() => numbers(D.Normal(start, 1).prob(tensor(xs))), [xs, start])
  const stepCurve = useMemo(() => numbers(D.Normal(at.loc, at.scale).prob(tensor(xs))), [xs, at])
  const { forward, reverse, table, moments } = fits
  const x = useAxis({ label: 'x', range: X_RANGE })
  const y = useAxis({ label: 'density', hold: 'union' })

  return (
    <Figure
      title="Forward against reverse KL: mode covering and mode seeking"
      purpose="The best single normal for a bimodal p depends on the direction: minimising KL(p ‖ q) covers both modes (it matches moments), minimising KL(q ‖ p) seeks one mode, chosen by the start."
      defaultSize="L"
      state={state}
      controls={
        <div className="col-span-full">
          <Player
            label="reverse-fit step"
            value={step}
            onChange={setStep}
            count={path.length}
            format={(i) => `${i} / ${path.length - 1}`}
          />
        </div>
      }
      readouts={{
        'forward fit': (
          <>
            <Readout
              label="q"
              value={`N(${fmt(forward.loc)}, ${fmt(forward.scale)}²), ${forward.path.length - 1} steps`}
            />
            <Readout label="moment match E_p[X], sd_p[X]" value={`${fmt(moments.mean)}, ${fmt(moments.sd)}`} />
            <Readout label="KL(p ‖ q)" value={methodValue(table.ff)} />
            <Readout label="KL(q ‖ p)" value={methodValue(table.fr)} />
          </>
        ),
        'reverse fit': (
          <>
            <Readout
              label="q"
              value={`N(${fmt(reverse.loc)}, ${fmt(reverse.scale)}²), ${reverse.path.length - 1} steps`}
            />
            <Readout
              label={`step ${step}`}
              value={`N(${fmt(at.loc)}, ${fmt(at.scale)}²), KL(q ‖ p) ${fmt(at.objective)}`}
            />
            <Readout label="KL(p ‖ q)" value={methodValue(table.rf)} />
            <Readout label="KL(q ‖ p)" value={methodValue(table.rr)} />
          </>
        ),
      }}
      caption={
        <>
          Drag the start μ₀ on the chart (or use its slider): both fits begin at N(μ₀, 1). Minimising the forward KL(p ‖
          q) = E_p[log p − log q] is maximum likelihood in the limit of many draws from p: q is charged wherever p has
          mass and q does not, so it spreads to cover both modes. Over normals the forward fit is moment matching, and
          it lands on the mean and sd of p from any start. Minimising the reverse KL(q ‖ p) = E_q[log q − log p], the
          objective of variational inference, charges q only where q itself has mass: putting mass between the modes,
          where p is small, costs a lot, while ignoring a mode costs nothing. The reverse fit therefore seeks one mode,
          and which one depends on the start. Each fit has the lower value of its own objective. Play from step 0 to
          watch the reverse fit&apos;s L-BFGS iterates (dashed) move from the start to its mode. Starting exactly at the
          centre of a symmetric p stalls at the saddle between the modes.
        </>
      }
    >
      <Plot x={x} y={y}>
        <Area name="p (mixture)" x={xs} y={curves.p} slot={0} />
        <Curve name="q minimising KL(p ‖ q): forward, mode covering" x={xs} y={curves.forward} slot={1} stale={stale} />
        <Curve name="q minimising KL(q ‖ p): reverse, mode seeking" x={xs} y={curves.reverse} slot={2} stale={stale} />
        <Curve name="start N(μ₀, 1)" x={xs} y={startCurve} muted dashed live />
        <Curve name="reverse fit at the step shown" x={xs} y={stepCurve} slot={2} dashed thin live stale={stale} />
        <Handle {...state.handle('fits.start', { label: 'start μ₀' })} />
      </Plot>
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
  const state = useFigureState({
    draws: choice(DRAWS, 10000, { label: 'Monte Carlo draws' }),
    seed: slider(1, 20, 1, { step: 1, label: 'seed' }),
  })
  const { draws, seed } = state
  const rows = useMemo(
    () =>
      PAIRS.map(({ label, p, q, continuous }, i) => {
        const closed = num(D.kl(p, q))
        const quad = continuous ? D.klNumerical(p, q).value : null
        const mc = D.klMonteCarloWithError(child(stream(`kl-pairs/${seed}`), i), p, q, draws)
        return { label, closed, quad, mc, z: (mc.value - closed) / mc.standardError }
      }),
    [draws, seed],
  )
  return (
    <Figure
      title="Closed-form KL pairs against quadrature and Monte Carlo"
      purpose="Every registered closed form agrees with quadrature to many digits and with a Monte Carlo estimate to within a few standard errors."
      state={state}
      defaultSize="L"
      hoverReadout={false}
      caption="kl(p, q) dispatches on the two family names to a closed form. For continuous univariate pairs, quadrature of p log(p/q) should agree to many digits. The z column is (Monte Carlo − closed form) / standard error; |z| above about 3 would point at a wrong formula."
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
