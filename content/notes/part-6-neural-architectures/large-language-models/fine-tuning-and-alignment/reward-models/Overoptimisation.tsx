import { useContext, useMemo, type ReactNode } from 'react'
import {
  choice,
  Contours,
  Curve,
  Figure,
  float,
  formatNumber,
  FrameContext,
  Handle,
  int,
  Plot,
  Points,
  Raster,
  Readout,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import {
  bestOfNCurve,
  bestOfNKl,
  bestOfNWeights,
  fitBradleyTerry,
  syntheticPreferences,
} from 'aifn-methods/neural/reward-models'
import { child, integers, standardNormals, stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'

/** Responses in the pool that best-of-n draws from. */
const POOL = 4096
/** The largest n offered. */
const MAX_N = 1024
/** Held-out comparisons for the pair accuracy. */
const HELD_OUT = 2000
/** Pool responses drawn in the feature-space panel (all of them are used in the computation). */
const SHOWN = 1500
/** Half-width of the feature-space panel. */
const SPAN = 3.5
const PANEL = 300

/**
 * The gold reward of a response with features x = (x₁, x₂): it rises with both features, but less when both are large
 * at once, and it peaks at x₁ = x₂ = 1/0.9.
 */
const gold = (x: Float64Array) => x[0] + x[1] - 0.4 * x[0] * x[1] - 0.25 * (x[0] * x[0] + x[1] * x[1])
const GOLD_OPTIMUM = 1 / 0.9
/** Every monomial of degree at most 2, so a quadratic proxy can represent the gold reward exactly. */
const quadratic = (x: Float64Array) => Float64Array.from([x[0], x[1], x[0] * x[0], x[0] * x[1], x[1] * x[1]])

/** The values of n on the curve: about eight per doubling, each once. */
const NS = [...new Set(Array.from({ length: 81 }, (_, i) => Math.round(2 ** (i / 8))))]
const distance = (n: number) => Math.sqrt(bestOfNKl(n))

/** The n whose distance √KL is nearest d. */
function nearestN(d: number) {
  let lo = 1
  let hi = MAX_N
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1
    if (distance(mid) < d) lo = mid
    else hi = mid
  }
  return Math.abs(distance(lo) - d) <= Math.abs(distance(hi) - d) ? lo : hi
}

/** The share of pairs whose chosen row scores higher than the rejected row. */
function pairAccuracy(score: (x: Float64Array) => number, chosen: number[], rejected: number[]) {
  let right = 0
  const pairs = chosen.length / 2
  for (let i = 0; i < pairs; i++) {
    const w = Float64Array.of(chosen[2 * i], chosen[2 * i + 1])
    const l = Float64Array.of(rejected[2 * i], rejected[2 * i + 1])
    if (score(w) > score(l)) right++
  }
  return right / pairs
}

/** Plots inside a Figure take the frame's height; this gives the plots inside it a fixed height instead. */
function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}

export function Overoptimisation() {
  const state = useFigureState({
    proxy: choice(
      [
        { value: 'linear', label: 'linear in x₁, x₂ (misspecified)' },
        { value: 'quadratic', label: 'quadratic in x₁, x₂ (can match gold)' },
      ],
      'linear',
      { label: 'proxy reward model' },
    ),
    comparisons: int(1000, {
      ge: 20,
      le: 20000,
      scale: 'log10',
      suggestions: [100, 300, 1000, 3000, 10000],
      label: 'comparisons N',
    }),
    temperature: float(1, { ge: 0.1, le: 5, suggestions: [0.25, 0.5, 1, 2, 3], label: 'label noise T' }),
    n: int(16, { ge: 1, le: MAX_N, scale: 'log10', suggestions: [1, 4, 16, 64, 256, 1024], label: 'best of n' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
  })
  const { proxy, comparisons, temperature, n, seed } = state

  const fit = useComputed(
    () => {
      const root = stream(`reward-models/overoptimisation/${seed}`)
      const features = proxy === 'quadratic' ? quadratic : undefined
      const prefs = syntheticPreferences({
        seed: integers(child(root, 'comparisons'), 2 ** 31),
        n: comparisons,
        dim: 2,
        gold,
        noise: 'gumbel',
        temperature,
      })
      const model = fitBradleyTerry(prefs, { features })
      const held = syntheticPreferences({
        seed: integers(child(root, 'held-out'), 2 ** 31),
        n: HELD_OUT,
        dim: 2,
        gold,
        noise: 'gumbel',
        temperature,
      })
      const heldW = toFlat(held.chosen)
      const heldL = toFlat(held.rejected)

      // The proxy is scaled by T: the Bradley–Terry fit estimates g / T, so T times it is in the gold reward's units.
      const pool = standardNormals(child(root, 'pool'), 2 * POOL)
      const proxyScore = new Float64Array(POOL)
      const goldScore = new Float64Array(POOL)
      for (let i = 0; i < POOL; i++) {
        const x = pool.subarray(2 * i, 2 * i + 2)
        proxyScore[i] = temperature * model.score(x)
        goldScore[i] = gold(x)
      }
      const curve = bestOfNCurve({ proxy: proxyScore, gold: goldScore, ns: NS })
      const base = curve[0]

      // Best-of-n's mean response for each n: the pool sorted by proxy, weighted by each rank's chance of winning.
      const order = Array.from({ length: POOL }, (_, i) => i).sort((a, b) => proxyScore[a] - proxyScore[b])
      const path = NS.map((m) => {
        const w = bestOfNWeights(POOL, m)
        let x1 = 0
        let x2 = 0
        order.forEach((i, rank) => {
          x1 += w[rank] * pool[2 * i]
          x2 += w[rank] * pool[2 * i + 1]
        })
        return [x1, x2] as const
      })

      let peak = 0
      curve.forEach((p, i) => {
        if (p.gold > curve[peak].gold) peak = i
      })
      const grid = Array.from({ length: 71 }, (_, i) => -SPAN + (2 * SPAN * i) / 70)
      const proxyField = grid.map((b) => grid.map((a) => temperature * model.score(Float64Array.of(a, b))))
      const sorted = [...proxyScore].sort((a, b) => a - b)
      const levels = [0.5, 0.9, 0.99, 0.999].map((q) => sorted[Math.floor(q * (POOL - 1))])

      return {
        proxyScore,
        goldScore,
        order,
        pool,
        d: curve.map((p) => Math.sqrt(p.kl)),
        proxyGain: curve.map((p) => p.proxy - base.proxy),
        goldGain: curve.map((p) => p.gold - base.gold),
        base,
        peak,
        path,
        grid,
        proxyField,
        levels,
        accuracy: pairAccuracy(model.score, heldW, heldL),
        ceiling: pairAccuracy(gold, heldW, heldL),
      }
    },
    [proxy, comparisons, temperature, seed],
    { mode: 'release' },
  )
  const f = fit.value

  // The chosen n: its point on the curve and best-of-n's mean response, exactly.
  const at = useMemo(() => {
    const [p] = bestOfNCurve({ proxy: f.proxyScore, gold: f.goldScore, ns: [n] })
    const w = bestOfNWeights(POOL, n)
    let x1 = 0
    let x2 = 0
    f.order.forEach((i, rank) => {
      x1 += w[rank] * f.pool[2 * i]
      x2 += w[rank] * f.pool[2 * i + 1]
    })
    return {
      d: Math.sqrt(p.kl),
      kl: p.kl,
      proxy: p.proxy - f.base.proxy,
      gold: p.gold - f.base.gold,
      x: [x1, x2] as const,
    }
  }, [f, n])

  const goldField = useMemo(() => {
    const grid = f.grid
    return grid.map((b) => grid.map((a) => gold(Float64Array.of(a, b))))
  }, [f.grid])
  const shown = useMemo(
    () => ({
      x: Array.from({ length: SHOWN }, (_, i) => f.pool[2 * i]),
      y: Array.from({ length: SHOWN }, (_, i) => f.pool[2 * i + 1]),
    }),
    [f.pool],
  )
  const path = useMemo(() => ({ x: f.path.map((p) => p[0]), y: f.path.map((p) => p[1]) }), [f.path])

  const fx = useAxis({ label: 'feature x₁', range: [-SPAN, SPAN] })
  const fy = useAxis({ label: 'feature x₂', range: [-SPAN, SPAN], equal: fx })
  const cx = useAxis({ label: 'distance d = √KL bound (√nats)', range: [0, distance(MAX_N)] })
  const cy = useAxis({ label: 'reward gain over n = 1' })

  const handle = {
    kind: 'x' as const,
    at: at.d,
    onDrag: (d: number) => state.set('n', nearestN(d)),
    label: `n = ${n}`,
  }
  const peakN = NS[f.peak]

  return (
    <Figure
      title="Reward overoptimisation"
      state={state}
      defaultSize="L"
      caption={`Responses are points x = (x₁, x₂) drawn from a standard normal. The gold reward g(x) = x₁ + x₂ − 0.4 x₁x₂ − 0.25 (x₁² + x₂²) (left, shaded; darker is higher) plays the human and is largest at x₁ = x₂ ≈ 1.11 (the black cross). It labels N random pairs under the Bradley–Terry model at temperature T, and a proxy reward model is fitted to them by Bradley–Terry logistic regression. Best-of-n draws n of ${POOL} pool responses (grey dots, ${SHOWN} shown) and keeps the one the proxy scores highest. Left: the proxy's level lines (orange) at the pool's 50th, 90th, 99th and 99.9th percentiles, and best-of-n's mean response as n grows (black path; the dot is the chosen n). Right: the exact expected proxy and gold rewards of best-of-n, each minus its value at n = 1, against d = √(log n − (n − 1)/n), the square root of the KL bound. The proxy is multiplied by T, because Bradley–Terry estimates g / T. Drag the vertical line on the right to set n. With the linear proxy, best-of-n moves along the proxy's gradient and past the gold peak: the proxy keeps rising, while gold peaks near n = 13 and falls by about 1 by n = 1024, whatever N and T. The linear proxy's held-out pair accuracy, on ${HELD_OUT} fresh noisy pairs, is within about 0.01 of the gold reward's own accuracy on them, which is the ceiling the label noise sets, so held-out accuracy does not reveal the misspecification. The quadratic proxy can represent g. With 10,000 comparisons gold rises to its maximum and stays there; with 100 comparisons, or with noisy labels (T = 3), it often peaks and falls again, by less than with the linear proxy.`}
      readouts={
        <>
          <Readout label="n at the gold peak" value={String(peakN)} />
          <Readout label="gold gain at the peak" value={formatNumber(f.goldGain[f.peak])} />
          <Readout label={`KL bound at n = ${n} (nats)`} value={formatNumber(at.kl)} />
          <Readout label={`proxy gain at n = ${n}`} value={formatNumber(at.proxy)} />
          <Readout label={`gold gain at n = ${n}`} value={formatNumber(at.gold)} />
          <Readout label="held-out pair accuracy, proxy" value={formatNumber(f.accuracy)} />
          <Readout label="held-out pair accuracy, gold reward" value={formatNumber(f.ceiling)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <FixedHeight height={PANEL}>
          <Plot x={fx} y={fy}>
            <Raster
              name="gold reward"
              x={f.grid}
              y={f.grid}
              z={goldField}
              scale="sequential"
              fillOpacity={0.55}
              valueLabel="gold reward"
              range={[-4, 1.2]}
            />
            <Points name="pool" x={shown.x} y={shown.y} muted dense />
            <Contours
              name="proxy level lines"
              x={f.grid}
              y={f.grid}
              z={f.proxyField}
              levels={f.levels}
              labels={false}
              slot={1}
              stale={fit.stale}
            />
            <Curve name="best-of-n mean response" x={path.x} y={path.y} emphasis width={1.5} stale={fit.stale} />
            <Points name="gold maximum" x={[GOLD_OPTIMUM]} y={[GOLD_OPTIMUM]} emphasis shape={4} size={12} />
            <Points name={`mean response, n = ${n}`} x={[at.x[0]]} y={[at.x[1]]} emphasis size={9} live />
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={cx} y={cy}>
            <Curve name="proxy (× T)" x={f.d} y={f.proxyGain} slot={1} stale={fit.stale} />
            <Curve name="gold" x={f.d} y={f.goldGain} slot={0} stale={fit.stale} />
            <Points name="gold peak" x={[f.d[f.peak]]} y={[f.goldGain[f.peak]]} emphasis size={8} />
            <Handle {...handle} />
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
