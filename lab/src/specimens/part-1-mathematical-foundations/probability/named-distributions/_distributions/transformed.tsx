import { grad } from 'aifn-compute/foundation/autodiff'
import * as D from 'aifn-compute/probability/bijectors'
import * as Distributions from 'aifn-compute/probability/distributions'
import { integrate } from 'aifn-compute/numerics/quadrature'
import { stream } from 'aifn-compute/foundation/random'
import { histogram } from 'aifn-compute/probability/stats'
import { linspace, tensor, toFlat, unwrap, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { useMemo, useState } from 'react'
import { Equation, Figure, live, Tex, tex } from 'aifn-render/layout'
import { choice, row, slider, toggle, useFigureState, variants } from 'aifn-render/state'
import { Area, Bars, Curve, Handle, Plot, Plots, Points, Readout, SupportBand, useAxis } from 'aifn-render/viz'
import { formatValue, histogramBars } from '@lab/views'

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

type Range = [number, number]
/** A case's slider values. */
type Num = Record<string, number>
type AnyMap = D.Bijector | D.ManyToOneMap

const BINS = 48
/** Width of the interval dx around x₀, as a share of the input axis. */
const DX = 0.04

// ── Input distributions and maps ────────────────────────────────────────────────────────────────────────────────────

const BASES = variants(
  {
    normal: {
      label: 'Normal',
      params: { mu: slider(-2, 2, 0, { label: 'mean μ' }), sigma: slider(0.2, 2, 0.8, { label: 'sd σ' }) },
      f: (_: number, p: Num): Distributions.Univariate => Distributions.Normal(p.mu, p.sigma),
    },
    uniform: {
      label: 'Uniform',
      params: { a: slider(-2, 2, -1, { label: 'low a' }), b: slider(-1.5, 3, 1, { label: 'high b' }) },
      f: (_: number, p: Num): Distributions.Univariate => Distributions.Uniform(p.a, p.b),
    },
    exponential: {
      label: 'Exponential',
      params: { rate: slider(0.2, 4, 1, { label: 'rate λ' }) },
      f: (_: number, p: Num): Distributions.Univariate => Distributions.Exponential(p.rate),
    },
    beta: {
      label: 'Beta',
      params: { alpha: slider(0.3, 6, 2, { label: 'α' }), beta: slider(0.3, 6, 3, { label: 'β' }) },
      f: (_: number, p: Num): Distributions.Univariate => Distributions.Beta(p.alpha, p.beta),
    },
    gamma: {
      label: 'Gamma',
      params: { shape: slider(0.5, 8, 2, { label: 'shape k' }), rate: slider(0.2, 4, 1, { label: 'rate λ' }) },
      f: (_: number, p: Num): Distributions.Univariate => Distributions.Gamma(p.shape, p.rate),
    },
  },
  { label: '1 · input distribution', choiceLabel: 'distribution of X' },
)

/** Each base as TeX, from its values. */
const BASE_TEX: Record<string, (p: Record<string, number>) => string> = {
  normal: (p) => String.raw`\mathcal{N}(${texNum(p.mu)}, ${texNum(p.sigma)}^2)`,
  uniform: (p) => String.raw`\mathrm{Uniform}(${texNum(p.a)}, ${texNum(p.b)})`,
  exponential: (p) => String.raw`\mathrm{Exponential}(${texNum(p.rate)})`,
  beta: (p) => String.raw`\mathrm{Beta}(${texNum(p.alpha)}, ${texNum(p.beta)})`,
  gamma: (p) => String.raw`\mathrm{Gamma}(${texNum(p.shape)}, ${texNum(p.rate)})`,
}

const MAPS = variants(
  {
    affine: {
      label: 'affine a·x + b',
      params: { a: slider(-3, 3, 1.5, { label: 'slope a' }), b: slider(-3, 3, 0, { label: 'shift b' }) },
      f: (_: number, p: Num): AnyMap => D.affineBijector(p.b, p.a),
    },
    exp: { label: 'exp', params: {}, f: (): AnyMap => D.expBijector },
    log: { label: 'log', params: {}, f: (): AnyMap => D.logBijector },
    sigmoid: {
      label: 'sigmoid σ(x / T)',
      params: { T: slider(0.2, 3, 1, { label: 'temperature T' }) },
      f: (_: number, p: Num): AnyMap => D.chainBijectors(D.affineBijector(0, 1 / p.T), D.sigmoidBijector),
    },
    tanh: { label: 'tanh', params: {}, f: (): AnyMap => D.tanhBijector },
    softplus: { label: 'softplus', params: {}, f: (): AnyMap => D.softplusBijector },
    power: {
      label: 'power xᵖ',
      params: { p: slider(-2, 3, 0.5, { label: 'power p' }) },
      f: (_: number, p: Num): AnyMap => D.powerBijector(p.p),
    },
    square: { label: 'square x² (two preimages)', params: {}, f: (): AnyMap => D.squareMap },
    probit: {
      label: 'normal cdf Φ (probability integral transform)',
      params: {},
      f: (): AnyMap => D.normalCdfBijector,
    },
  },
  { label: '2 · map', choiceLabel: 'map g', initial: 'exp' },
)

const FORMULA: Record<string, string> = {
  affine: 'g(x) = ax + b',
  exp: 'g(x) = e^x',
  log: String.raw`g(x) = \log x`,
  sigmoid: String.raw`g(x) = \sigma(x / T)`,
  tanh: String.raw`g(x) = \tanh x`,
  softplus: String.raw`g(x) = \log(1 + e^x)`,
  power: 'g(x) = x^p',
  square: 'g(x) = x^2',
  probit: String.raw`g(x) = \Phi(x)`,
}

// ── Helpers ──────────────────────────────────────────────────────────────────────────────────────────────────────────

/** A number for TeX. */
const texNum = (v: number) => formatValue(Number(v.toPrecision(3)))

/** An interval as TeX: \mathbb{R}, (0, \infty), [0, 1]. */
function intervalTex(i: D.Interval): string {
  if (i.lower === -Infinity && i.upper === Infinity) return String.raw`\mathbb{R}`
  const end = (v: number) => (v === Infinity ? String.raw`\infty` : v === -Infinity ? String.raw`-\infty` : texNum(v))
  return `${i.lowerOpen ? '(' : '['}${end(i.lower)}, ${end(i.upper)}${i.upperOpen ? ')' : ']'}`
}

/** The axis range for a support: its finite ends as hard limits, quantiles where it is infinite. */
function supportWindow(d: Distributions.Univariate, set: D.Interval, lo: number, hi: number): Range {
  const a = Number.isFinite(set.lower) ? set.lower : num(d.quantile(lo))
  const b = Number.isFinite(set.upper) ? set.upper : num(d.quantile(hi))
  return [a, b]
}

/**
 * A range held while parameters move: reset when `key` changes (the base, the map or a support), otherwise grown to
 * include each new range and never shrunk (the lab's `holdFit="union"`).
 */
function useHeldRange(key: string, range: Range | null): Range | null {
  const [held, setHeld] = useState<{ key: string; range: Range } | null>(null)
  let next = held
  if (range && (!held || held.key !== key)) next = { key, range }
  else if (range && held) {
    const lo = Math.min(held.range[0], range[0])
    const hi = Math.max(held.range[1], range[1])
    if (lo !== held.range[0] || hi !== held.range[1]) next = { key, range: [lo, hi] }
  }
  if (next !== held) setHeld(next)
  return next && next.key === key ? next.range : range
}

/** Histogram heights as a density of all n draws (draws outside the window count towards n but are not drawn). */
function densityHistogram(values: readonly number[], [lo, hi]: Range) {
  const bars = histogramBars(histogram(values as number[], { bins: BINS, range: [lo, hi] }))
  const width = (hi - lo) / BINS
  const heights = bars.counts.map((c) => c / (values.length * width))
  return { edges: bars.edges, centres: bars.x, heights }
}

/** A peak for a density axis: the curve's, but no more than 2.5 times the histogram's (densities can be infinite). */
const densityPeak = (curve: readonly number[], hist: readonly number[]) => {
  const h = Math.max(...hist)
  const c = Math.max(0, ...curve.filter(Number.isFinite))
  return 1.1 * Math.max(h, Math.min(c, 2.5 * h))
}

// ── The figure ───────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The change of variables as three aligned panels, after the site's change-of-variables figure: the map y = g(x) in
 * the middle, the input density p_X along the bottom (sharing x), and the output density p_Y rotated on the right
 * (sharing y), each with a histogram of draws. Densities, supports, images, draws, derivatives and integrals all come
 * from aifn: `Transformed` for bijectors, `Pushforward` for x² (a sum over both preimages).
 */
export function TransformedSpecimen() {
  const state = useFigureState({
    base: BASES,
    map: MAPS,
    reveal: row('3 · density of Y and draws', {
      // Off: the naive density p_X(g⁻¹(y)) alone, which misses the histogram. On: the Jacobian-scaled p_Y, with the
      // naive curve kept dashed so the difference shows.
      jacobian: toggle(false, 'Jacobian scaling'),
      size: choice([1000, 5000, 20000], 5000, { label: 'draws' }),
      seed: slider(1, 20, 1, { step: 1, label: 'seed' }),
    }),
    // Placed on the chart and clamped to the input window below; a fine step so a drag is smooth on any window.
    x0: slider(-100, 100, 0.5, { onChart: true, step: 0.0001, label: 'x₀' }),
  })
  const { base, map } = state
  const { jacobian, size: n, seed } = state.reveal

  // The base, the map and the pushforward; a base whose support is not inside the map's domain is an aifn error.
  const setup = useMemo(() => {
    let X: Distributions.Univariate | null = null
    let g: AnyMap | null = null
    try {
      X = (base.f as unknown as (x: number) => Distributions.Univariate)(0)
      g = (map.f as unknown as (x: number) => AnyMap)(0)
      const Y = 'branches' in g ? Distributions.Pushforward(X, g) : Distributions.Transformed(X, g)
      const xSet = D.supportInterval(X.support)
      const ySet = D.supportInterval(Y.support)
      return { ok: true as const, X, g, Y, xSet, ySet }
    } catch (e) {
      return { ok: false as const, X, g, error: (e as Error).message }
    }
  }, [base, map])

  const key = setup.ok
    ? `${base.key}|${map.key}|${D.formatInterval(setup.xSet)}|${D.formatInterval(setup.ySet)}`
    : `${base.key}|${map.key}|error`

  // Axis windows: supports with finite ends as limits, quantiles elsewhere; held (union) until the key changes.
  const windows = useMemo(() => {
    if (!setup.ok) return null
    return {
      x: supportWindow(setup.X, setup.xSet, 0.001, 0.999),
      y: supportWindow(setup.Y, setup.ySet, 0.002, 0.99),
    }
  }, [setup])
  const xw = useHeldRange(`${key}|x`, windows?.x ?? null)
  const yw = useHeldRange(`${key}|y`, windows?.y ?? null)

  // Draws of X and their images g(xᵢ).
  const draws = useMemo(() => {
    if (!setup.ok) return null
    const xs = setup.X.sample(stream(`change-of-variables/${seed}`), { shape: [n] }) as Tensor
    return { xs: numbers(xs), ys: numbers(setup.g.forward(xs)) }
  }, [setup, seed, n])

  // Curves on the held windows. The output curve is evaluated on the images of an even grid in x and on an even grid
  // in y, so that it is resolved both where g squashes and where it stretches.
  const curves = useMemo(() => {
    if (!setup.ok || !xw || !yw || !draws) return null
    const { X, g, Y, xSet } = setup
    const xs = grid(xw[0], xw[1], 401)
    const pX = numbers(X.prob(tensor(xs))).map(gap)
    const gx = numbers(g.forward(tensor(xs))).map(gap)
    const ys = [...gx.filter((y) => y >= yw[0] && y <= yw[1]), ...grid(yw[0], yw[1], 301)].sort((a, b) => a - b)
    const pY = numbers(Y.prob(tensor(ys))).map(gap)
    // Without the Jacobian: p_X at each preimage, summed over the branches whose image holds y.
    const pieces = D.branchImages(g, xSet)
    const naiveAt = (y: number) =>
      pieces.reduce(
        (t, { branch, image }) => (y >= image.lower && y <= image.upper ? t + num(X.prob(branch.inverse(y))) : t),
        0,
      )
    const naive = ys.map((y) => gap(naiveAt(y)))
    const hx = densityHistogram(draws.xs, xw)
    const hy = densityHistogram(draws.ys, yw)
    return {
      xs,
      pX,
      gx,
      ys,
      pY,
      naive,
      naiveAt,
      hx,
      hy,
      peakX: densityPeak(pX, hx.heights),
      peakY: densityPeak(pY, hy.heights),
    }
  }, [setup, xw, yw, draws])
  const dX = useHeldRange(`${key}|dx`, curves ? [0, curves.peakX] : null)
  const dY = useHeldRange(`${key}|dy`, curves ? [0, curves.peakY] : null)

  // ∫ p_Y dy and ∫ p_X(g⁻¹(y)) dy over the whole image, by adaptive quadrature.
  const integrals = useMemo(() => {
    if (!setup.ok || !curves) return null
    const { Y, ySet } = setup
    const run = (h: (y: number) => number) => {
      try {
        // A density can be infinite (or 0 · ∞) at a support end, which a node rounded onto the end would hit; single
        // points carry no area, so they count as 0.
        const r = integrate((y) => (Number.isFinite(h(y)) ? h(y) : 0), ySet.lower, ySet.upper, { maxIntervals: 400 })
        if (!r.converged && !(Math.abs(r.value) < 1e6)) return '∞ (diverges)'
        return `${formatValue(Number(r.value.toPrecision(4)))}${r.converged ? '' : ' (not converged)'}`
      } catch {
        return '—'
      }
    }
    return { withJacobian: run((y) => num(Y.prob(y))), without: run(curves.naiveAt) }
  }, [setup, curves])

  const ready = setup.ok && xw && yw && dX && dY && curves && draws
  const x0 = ready ? clamp(state.x0, Math.max(xw[0], setup.xSet.lower), Math.min(xw[1], setup.xSet.upper)) : 0

  // The point x₀, its image and the interval dx around it; the slope from autodiff, log |g′| from the map.
  const probe = useMemo(() => {
    if (!ready) return null
    const { X, Y, g } = setup
    const lo = Math.max(xw[0], setup.xSet.lower)
    const hi = Math.min(xw[1], setup.xSet.upper)
    const half = (DX * (xw[1] - xw[0])) / 2
    const a = clamp(x0 - half, lo, hi - 2 * half)
    const b = a + 2 * half
    const forward = (x: number) => num(g.forward(x))
    const y0 = forward(x0)
    const slope = num(grad((x: Value) => g.forward(x))(x0))
    const branch = D.asManyToOne(g).branches.find((br) => x0 >= br.domain.lower && x0 <= br.domain.upper)
    const logJ = branch ? num(branch.logAbsDetJacobian(x0)) : NaN
    const band = grid(a, b, 16)
    const [yA, yB] = forward(a) <= forward(b) ? [forward(a), forward(b)] : [forward(b), forward(a)]
    const bandY = grid(yA, yB, 16)
    return {
      a,
      b,
      ya: forward(a),
      yb: forward(b),
      yA,
      yB,
      y0,
      slope,
      logJ,
      branch,
      band,
      bandG: band.map(forward),
      bandP: numbers(X.prob(tensor(band))).map(gap),
      bandY,
      bandPY: numbers(Y.prob(tensor(bandY))).map(gap),
      pX0: num(X.prob(x0)),
      pY0: num(Y.prob(y0)),
    }
  }, [ready, setup, xw, x0])

  // Dragging y₀ on the output axis moves x₀ to its preimage on the branch x₀ is on.
  const setY0 = (y: number) => {
    if (!probe?.branch || !setup.ok) return
    const images = D.branchImages(setup.g, setup.xSet)
    const piece = images.find((p) => p.branch === probe.branch) ?? images[0]
    const yy = clamp(y, piece.image.lower, piece.image.upper)
    const x = num(piece.branch.inverse(yy))
    if (Number.isFinite(x)) state.set('x0', x)
  }

  const xAxis = useAxis({ label: 'x', range: xw ?? undefined, key })
  const yAxis = useAxis({ label: 'y = g(x)', range: yw ?? undefined, key })
  const pxAxis = useAxis({ label: 'p_X(x)', range: dX ?? undefined, key })
  const pyAxis = useAxis({ label: 'p_Y(y)', range: dY ?? undefined, key })

  // Each selector's statement: the support of X, the map's domain and codomain.
  const X = setup.X
  const g = setup.g
  const baseTex = X
    ? String.raw`X \sim ${BASE_TEX[base.key](base.values as Record<string, number>)}, \; X \in ${intervalTex(D.supportInterval(X.support))}`
    : String.raw`\text{invalid parameters}`
  const mapTex = g
    ? String.raw`${FORMULA[map.key]}, \quad g: ${intervalTex(g.domain)} \to ${intervalTex(g.codomain)}`
    : FORMULA[map.key]

  const lognormal =
    setup.ok && base.key === 'normal' && map.key === 'exp' && probe
      ? num(Distributions.LogNormal(base.values.mu as number, base.values.sigma as number).prob(probe.y0))
      : null

  const ylo = yw?.[0] ?? 0
  const xhi = xw?.[1] ?? 0
  const span = xw ? 0.12 * (xw[1] - xw[0]) : 0
  return (
    <Figure
      id="change-of-variables"
      title="Change of variables: a density pushed through a map"
      purpose="Pushing X through y = g(x) spreads the probability in dx over dy ≈ |g′(x)| dx, so p_Y(y) = p_X(x) / |g′(x)|: without that Jacobian factor the density misses the draws and does not integrate to 1."
      description={
        <span className="flex flex-wrap gap-x-6 text-base">
          <Tex>{baseTex}</Tex>
          <Tex>{mapTex}</Tex>
        </span>
      }
      defaultSize="L"
      hoverReadout={false}
      state={state}
      equation={
        ready && probe ? (
          <Equation>
            {map.key === 'square'
              ? tex`p_Y(${live(probe.y0, { digits: 3 })}) = \sum_{x = \pm\sqrt{y}} \frac{p_X(x)}{|g'(x)|} = ${live(probe.pY0, { digits: 4, strong: true })}`
              : tex`p_Y(${live(probe.y0, { digits: 3 })}) = \frac{p_X(${live(x0, { digits: 3 })})}{|g'(${live(x0, { digits: 3 })})|} = \frac{${live(probe.pX0, { digits: 4 })}}{${live(Math.abs(probe.slope), { digits: 4 })}} = ${live(probe.pY0, { digits: 4, strong: true })}`}
          </Equation>
        ) : undefined
      }
      readouts={
        ready && probe && integrals
          ? {
              supports: (
                <>
                  <Readout label="X" value={D.formatInterval(setup.xSet)} />
                  <Readout label="Y = g(X)" value={D.formatInterval(setup.ySet)} />
                </>
              ),
              'at x₀': (
                <>
                  <Readout label="g′(x₀) (autodiff)" value={formatValue(probe.slope)} />
                  <Readout label="log |g′(x₀)| (log-det-Jacobian)" value={formatValue(probe.logJ)} />
                  <Readout
                    label="stretch Δy / Δx over dx"
                    value={formatValue((probe.yb - probe.ya) / (probe.b - probe.a))}
                  />
                  {lognormal !== null && (
                    <Readout label="LogNormal(μ, σ) density at y₀" value={formatValue(lognormal)} />
                  )}
                </>
              ),
              'total probability': (
                <>
                  <Readout label="∫ p_Y dy (with the Jacobian)" value={integrals.withJacobian} />
                  <Readout label="∫ p_X(g⁻¹(y)) dy (without)" value={integrals.without} />
                </>
              ),
            }
          : undefined
      }
      caption={
        <>
          X is drawn along the bottom (density and histogram), the map y = g(x) in the middle, and Y = g(X) rotated on
          the right, sharing the y-axis of g. Drag x₀ on the bottom or middle panel, or y₀ on the right panel: the
          interval dx around x₀ maps to an interval dy whose length is about |g′(x₀)| dx. With Jacobian scaling off, the
          right panel shows p_X(g⁻¹(y)) alone: it misses the histogram of the draws g(xᵢ), and its integral is not 1.
          Turn it on and the solid curve includes the factor and follows the histogram, with the unscaled curve kept
          dashed. For x² every y &gt; 0 has two preimages ±√y and the density adds both terms. Supports are marked along
          each density axis. The axes hold while parameters move and refit when the input distribution, the map or a
          support changes.
        </>
      }
    >
      {!setup.ok ? (
        <div className="flex min-h-40 flex-col items-start justify-center gap-2 rounded-md border border-dashed p-4 text-sm">
          <div className="font-medium">This map cannot be applied to this input distribution.</div>
          <div className="text-muted-foreground">{setup.error}</div>
          {X && g && (
            <Tex>{String.raw`X \in ${intervalTex(D.supportInterval(X.support))}, \quad \text{but } g \text{ is defined on } ${intervalTex(g.domain)}`}</Tex>
          )}
        </div>
      ) : ready && probe ? (
        <Plots rows={2} cols={2} heights={[3, 2]} widths={[2, 1]}>
          <Plot x={xAxis} y={yAxis}>
            <Curve name="y = g(x)" x={curves.xs} y={curves.gx} emphasis />
            <Curve
              name="guides"
              x={[probe.a, probe.a, NaN, probe.b, probe.b, NaN, probe.a, xhi, NaN, probe.b, xhi]}
              y={[ylo, probe.ya, NaN, ylo, probe.yb, NaN, probe.ya, probe.ya, NaN, probe.yb, probe.yb]}
              muted
              dashed
              live
            />
            <Curve
              name="tangent"
              x={[x0 - span, x0 + span]}
              y={[probe.y0 - probe.slope * span, probe.y0 + probe.slope * span]}
              slot={3}
              dashed
              live
            />
            <Curve name="g over dx" x={probe.band} y={probe.bandG} slot={0} live />
            <Points name="(x₀, y₀)" x={[x0]} y={[probe.y0]} emphasis live />
            <Handle {...state.handle('x0', { label: 'x₀' })} />
          </Plot>
          <Plot x={pyAxis} y={yAxis}>
            <Bars
              name="draws g(xᵢ)"
              x={curves.hy.centres}
              y={curves.hy.heights}
              edges={curves.hy.edges}
              orient="y"
              slot={1}
            />
            {jacobian ? (
              <>
                <Curve name="p_Y(y), with the Jacobian" x={curves.pY} y={curves.ys} slot={1} />
                <Curve name="p_X(g⁻¹(y)), without" x={curves.naive} y={curves.ys} slot={2} dashed />
              </>
            ) : (
              <Curve name="p_X(g⁻¹(y)), without the Jacobian" x={curves.naive} y={curves.ys} slot={2} />
            )}
            <SupportBand interval={setup.ySet} orient="y" />
            <Curve
              name="dy"
              x={[0, dY[1], NaN, 0, dY[1]]}
              y={[probe.yA, probe.yA, NaN, probe.yB, probe.yB]}
              muted
              dashed
              live
            />
            <Curve name="p_Y over dy" x={probe.bandPY} y={probe.bandY} emphasis live />
            <Points name="y₀" x={[gap(probe.pY0)]} y={[probe.y0]} emphasis live />
            <Handle kind="y" at={probe.y0} label="y₀" onDrag={setY0} />
          </Plot>
          <Plot x={xAxis} y={pxAxis}>
            <Bars name="draws xᵢ" x={curves.hx.centres} y={curves.hx.heights} edges={curves.hx.edges} slot={0} />
            <Curve name="p_X(x)" x={curves.xs} y={curves.pX} slot={0} />
            <SupportBand interval={setup.xSet} />
            <Area name="P(X in dx)" x={probe.band} y={probe.bandP} slot={0} live />
            <Handle {...state.handle('x0', { label: 'x₀' })} />
          </Plot>
          <div />
        </Plots>
      ) : null}
    </Figure>
  )
}
