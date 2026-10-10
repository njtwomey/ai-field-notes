import { useContext, useMemo, useState, type ReactNode } from 'react'
import {
  Bars,
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  FrameContext,
  int,
  Player,
  Plot,
  Points,
  Raster,
  Readout,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { pinwheel } from 'aifn-methods/data/synthetic'
import { child, integers, standardNormals, stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { assignNearest, kmeansPlusPlus, lloydUpdate } from 'aifn-compute/numerics/neighbours'

type P = [number, number]

const N = 400
const SPAN = 4
const PANEL = 300
const GRID = Array.from({ length: 61 }, (_, i) => -SPAN + (2 * SPAN * i) / 60)
const GRID_ROWS = GRID.flatMap((y) => GRID.map((x) => [x, y]))

/** Plots inside a Figure take the frame's height; this gives the plots inside it a fixed height instead. */
function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}

/** Fixed encoder outputs: the encoder is frozen, so only the codebook learns. */
const OUTPUTS: P[] = (() => {
  const x = toFlat(pinwheel(stream('vq-vae/collapse'), { n: N, arms: 4, noise: 0.05 }).x)
  return Array.from({ length: N }, (_, i) => [x[2 * i], x[2 * i + 1]])
})()

type Step = { codes: P[]; sizes: Float64Array; error: number; reset: number[] }

/**
 * Codebook learning with the encoder frozen. Each step assigns every encoder output to its nearest code and moves each
 * code a fraction `rate` of the way to the mean of its outputs, which is a gradient step on the codebook loss. A code
 * with no outputs has zero gradient and stays put. With resets, every `every` steps each such code is moved onto a
 * randomly chosen encoder output.
 */
function trace(o: {
  k: number
  init: string
  spread: number
  rate: number
  every: number
  steps: number
  seed: number
}) {
  const s = stream(`vq-vae/collapse/${o.seed}`)
  let codes: P[]
  if (o.init === 'kmeans++') {
    const c = toFlat(kmeansPlusPlus(child(s, 'init'), OUTPUTS, o.k).centroids)
    codes = Array.from({ length: o.k }, (_, j) => [c[2 * j], c[2 * j + 1]])
  } else {
    const z = standardNormals(child(s, 'init'), 2 * o.k)
    codes = Array.from({ length: o.k }, (_, j) => [o.spread * z[2 * j], o.spread * z[2 * j + 1]])
  }
  const out: Step[] = []
  let reset: number[] = []
  for (let t = 0; t <= o.steps; t++) {
    const a = assignNearest(OUTPUTS, codes)
    const sizes = Float64Array.from(toFlat(a.sizes))
    out.push({ codes, sizes, error: a.inertia / N, reset })
    if (t === o.steps) break
    const means = toFlat(lloydUpdate(OUTPUTS, a.labels, codes).centroids)
    codes = codes.map((e, j) => [e[0] + o.rate * (means[2 * j] - e[0]), e[1] + o.rate * (means[2 * j + 1] - e[1])])
    reset = []
    if (o.every > 0 && (t + 1) % o.every === 0) {
      codes = codes.map((e, j) => {
        if (sizes[j] > 0) return e
        reset.push(j)
        return OUTPUTS[integers(child(s, 'reset', t, j), N)]
      })
    }
  }
  return out
}

export function CodebookCollapse() {
  const state = useFigureState({
    k: choice([8, 16, 32], 16, { label: 'codebook size K' }),
    init: choice(
      [
        { value: 'random', label: 'random normal' },
        { value: 'kmeans++', label: 'k-means++ on the encoder outputs' },
      ],
      'random',
      { label: 'initial codebook' },
    ),
    spread: float(3, { ge: 0.05, le: 6, suggestions: [0.1, 1, 3], label: 'random spread (standard deviation)' }),
    every: int(0, { ge: 0, le: 50, suggestions: [0, 5, 20], label: 'reset unused codes every (0: never)' }),
    rate: float(0.3, { ge: 0.05, le: 1, suggestions: [0.1, 0.3, 1], label: 'step size' }),
    steps: int(40, { ge: 5, le: 200, suggestions: [40, 100], label: 'steps' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
  })
  const { k, init, spread, every, rate, steps, seed } = state

  const run = useComputed(
    () => trace({ k, init, spread, rate, every, steps, seed }),
    [k, init, spread, rate, every, steps, seed],
  )
  const path = run.value
  const key = `${k}|${init}|${spread}|${every}|${rate}|${steps}|${seed}`
  const [pos, setPos] = useState({ key, step: 0 })
  const step = pos.key === key ? Math.min(pos.step, path.length - 1) : 0
  const now = path[step]

  const cells = useMemo(() => {
    const labels = toFlat(assignNearest(GRID_ROWS, now.codes).labels)
    return GRID.map((_, r) => labels.slice(r * GRID.length, (r + 1) * GRID.length))
  }, [now.codes])
  const outputs = useMemo(() => ({ x: OUTPUTS.map((p) => p[0]), y: OUTPUTS.map((p) => p[1]) }), [])
  const split = useMemo(() => {
    const live = { x: [] as number[], y: [] as number[] }
    const dead = { x: [] as number[], y: [] as number[] }
    now.codes.forEach((e, j) => {
      const into = now.sizes[j] > 0 ? live : dead
      into.x.push(e[0])
      into.y.push(e[1])
    })
    return { live, dead }
  }, [now])
  const used = now.sizes.filter((c) => c > 0).length
  const perplexity = useMemo(() => {
    let h = 0
    for (const c of now.sizes) if (c > 0) h -= (c / N) * Math.log(c / N)
    return Math.exp(h)
  }, [now.sizes])
  const usage = useMemo(() => {
    const order = Array.from(now.sizes, (c, j) => ({ c, j })).sort((a, b) => b.c - a.c)
    return { x: order.map((_, i) => i + 1), y: order.map((o) => o.c / N) }
  }, [now.sizes])
  const usedCurve = useMemo(
    () => ({ x: path.map((_, t) => t), y: path.map((p) => p.sizes.filter((c) => c > 0).length) }),
    [path],
  )

  const x = useAxis({ label: 'latent z₁', range: [-SPAN, SPAN] })
  const y = useAxis({ label: 'latent z₂', range: [-SPAN, SPAN], equal: x })
  const bx = useAxis({ label: 'code, most used first', range: [0.5, k + 0.5] })
  const by = useAxis({ label: 'share of encoder outputs', range: [0, 0.5] })
  const ux = useAxis({ label: 'step', range: [0, steps] })
  const uy = useAxis({ label: 'codes in use', range: [0, k] })

  return (
    <Figure
      title="Dead codes and resets"
      state={state}
      defaultSize="L"
      caption={`Codebook learning with the encoder frozen: ${N} fixed encoder outputs (grey) and K codes, with the latent space shaded by the nearest code. Each step assigns every encoder output to its nearest code and moves each code a fraction of the way (the step size) to the mean of its outputs, which is a gradient step on the codebook loss. Black dots are codes in use; red dots are dead codes, which no encoder output chooses. A dead code's codebook loss is zero, so it never moves. Started from a wide random normal (standard deviation 3), many codes fall outside the data and stay dead: with K = 16 and seed 1, 10 codes are in use at the start and still 10 after 40 steps (6 to 10, depending on the seed). Resetting every unused code onto a random encoder output every 5 steps brings all 16 into use at the first reset and lowers the mean quantisation error ‖z_e − z_q‖² from 0.25 to 0.09. Starting from k-means++ on the encoder outputs uses every code from the first step and ends at 0.07. Top right: the share of encoder outputs per code, most used first. Bottom right: the number of codes in use over the steps.`}
      controls={
        <Player
          value={step}
          onChange={(s) => setPos({ key, step: s })}
          count={path.length}
          label="step"
          format={(s) => `step ${s} of ${path.length - 1}`}
        />
      }
      readouts={
        <>
          <Readout label="codes in use" value={`${used} of ${k}`} />
          <Readout label="perplexity of code use" value={formatNumber(perplexity)} />
          <Readout label="mean ‖z_e − z_q‖²" value={formatNumber(now.error)} />
          <Readout label="codes reset at this step" value={String(now.reset.length)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-[3fr_2fr]">
        <FixedHeight height={PANEL}>
          <Plot x={x} y={y}>
            <Raster name="code cells" x={GRID} y={GRID} z={cells} scale="categorical" fillOpacity={0.12} boundary />
            <Points name="encoder outputs" x={outputs.x} y={outputs.y} muted dense />
            <Points name="codes in use" x={split.live.x} y={split.live.y} emphasis size={9} live />
            <Points name="dead codes" x={split.dead.x} y={split.dead.y} tone="destructive" size={9} live />
          </Plot>
        </FixedHeight>
        <div className="grid grid-rows-2 gap-2">
          <FixedHeight height={PANEL / 2 - 4}>
            <Plot x={bx} y={by}>
              <Bars name="share" x={usage.x} y={usage.y} slot={0} />
              <Curve name="uniform 1/K" x={[0.5, k + 0.5]} y={[1 / k, 1 / k]} muted dashed />
            </Plot>
          </FixedHeight>
          <FixedHeight height={PANEL / 2 - 4}>
            <Plot x={ux} y={uy}>
              <Curve name="codes in use" x={usedCurve.x} y={usedCurve.y} slot={0} />
              <Points name="step played" x={[step]} y={[used]} emphasis size={7} live />
            </Plot>
          </FixedHeight>
        </div>
      </div>
    </Figure>
  )
}
