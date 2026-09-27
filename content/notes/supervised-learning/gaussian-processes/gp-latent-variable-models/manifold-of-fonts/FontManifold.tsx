import { useEffect, useMemo, useRef, useState } from 'react'
import {
  GlyphPlot,
  Heatmap,
  Interactive,
  ParamButton,
  ParamChoice,
  ParamSlider,
  Readout,
  formatNumber,
  useParam,
  type GlyphShape,
  type Handle,
  type HeatmapOverlay,
  type Vec2,
} from '@/components/viz'
import type { FontInfo, FontManifoldData, GlyphLayout } from '@/generated/contracts'
import { useFigure } from '@/lib/generated'
import { linspace } from '@/lib/math'
import {
  centre,
  gplvmFitter,
  pcaScores,
  predictor,
  type GplvmState,
  type Point,
  type Predictor,
} from '../_shared/gplvm'

const LIM = 3
const GRID = 41
const AXIS = linspace(-LIM, LIM, GRID)
const ITERATIONS = 300
/** Words in the interpolation strip, and their spacing in font units. */
const STRIP = 6
const STRIP_COL = 4400
const STRIP_ROW = 1000
/** Milliseconds of fitting per animation frame, so the page stays responsive while the model learns. */
const FRAME_BUDGET = 12

const CLASSES: FontInfo['cls'][] = ['sans', 'humanist', 'serif', 'didone', 'slab', 'mono']
const CLASS_NAMES = ['sans', 'humanist', 'serif', 'didone', 'slab', 'mono']

type Subset = 'all' | 'upright' | 'proportional'
const SUBSETS: { value: Subset; label: string }[] = [
  { value: 'all', label: 'all fonts' },
  { value: 'upright', label: 'no italics' },
  { value: 'proportional', label: 'no monospace' },
]
const inSubset = (f: FontInfo, s: Subset) =>
  s === 'upright' ? !f.italic : s === 'proportional' ? f.cls !== 'mono' : true

type Pair = 'weight' | 'style' | 'mono'
const PAIRS: { value: Pair; label: string }[] = [
  { value: 'weight', label: 'weight' },
  { value: 'style', label: 'sans–didone' },
  { value: 'mono', label: 'mono–italic' },
]

const fontName = (f: FontInfo) => `${f.family} ${f.style}`

/** The fitted manifold on one subset of fonts: latent points, the GP posterior and its standard deviation grid. */
type Model = {
  idx: number[]
  state: GplvmState
  p: Predictor
  mean: number[]
  scale: number
  sdGrid: number[][]
  /** Output vector (in font units) generated at latent point x. */
  generate: (x: Point) => number[]
}

type FitProgress = { subset: Subset; idx: number[]; X: Point[]; iteration: number; ms: number; model?: Model }

/**
 * Fits a 2-D GP-LVM to the chosen fonts in the browser, a few Adam steps per animation frame, and reports progress so
 * that the latent points can be drawn moving into place.
 */
function useManifoldFit(
  data: FontManifoldData | undefined,
  subset: Subset,
  onFitted: (model: Model) => void,
): FitProgress | undefined {
  const [progress, setProgress] = useState<FitProgress>()
  const fitted = useRef(onFitted)
  useEffect(() => {
    fitted.current = onFitted
  })
  useEffect(() => {
    if (!data) return
    const idx = data.fonts.flatMap((f, i) => (inSubset(f, subset) ? [i] : []))
    let ms = 0
    let frame = 0
    let t = performance.now()
    const { Y: centred, mean } = centre(idx.map((i) => data.vectors[i]))
    // One scale for every coordinate, so that the kernel variance is near 1 and distances keep their meaning.
    const scale = Math.sqrt(
      centred.reduce((s, r) => s + r.reduce((a, v) => a + v * v, 0), 0) / (idx.length * mean.length),
    )
    const Y = centred.map((r) => r.map((v) => v / scale))
    const fitter = gplvmFitter(Y, pcaScores(Y))
    ms += performance.now() - t
    let state = fitter.run(0)
    const tick = () => {
      t = performance.now()
      do state = fitter.run(5)
      while (fitter.iteration < ITERATIONS && performance.now() - t < FRAME_BUDGET)
      ms += performance.now() - t
      if (fitter.iteration < ITERATIONS) {
        setProgress({ subset, idx, X: state.X, iteration: fitter.iteration, ms })
        frame = requestAnimationFrame(tick)
        return
      }
      t = performance.now()
      const p = predictor(Y, state)
      const sdGrid = AXIS.map((b) => AXIS.map((a) => p.sd([a, b]) / state.sf))
      const generate = (x: Point) => p.mean(x).map((v, d) => mean[d] + scale * v)
      ms += performance.now() - t
      const model = { idx, state, p, mean, scale, sdGrid, generate }
      setProgress({ subset, idx, X: state.X, iteration: fitter.iteration, ms, model })
      fitted.current(model)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [data, subset])
  return progress?.subset === subset ? progress : undefined
}

/** Contours of one glyph from a font vector, shifted right by dx. */
function glyphContours(v: number[], g: GlyphLayout, dx: number, dy = 0): Vec2[][] {
  let o = g.offset
  return g.contours.map((n) => {
    const pts: Vec2[] = []
    for (let i = 0; i < n; i++) pts.push([v[o + 2 * i] + dx, v[o + 2 * i + 1] + dy])
    o += 2 * n
    return pts
  })
}

/** A line of text set from a font vector: glyph contours placed by the vector's own advance widths. */
function setLine(v: number[], glyphs: Map<string, GlyphLayout>, text: string, dy = 0) {
  const contours: Vec2[][] = []
  const starts: number[] = []
  let x = 0
  for (const ch of text) {
    const g = glyphs.get(ch)!
    starts.push(x)
    contours.push(...glyphContours(v, g, x, dy))
    x += v[g.advance]
  }
  return { contours, starts, width: x }
}

/** Latent point of the first font in `idx` that matches, in order of preference. */
function pick(fonts: FontInfo[], idx: number[], tests: ((f: FontInfo) => boolean)[]): number {
  for (const test of tests) {
    const i = idx.find((j) => test(fonts[j]))
    if (i !== undefined) return i
  }
  return idx[0]
}

function pairEnds(fonts: FontInfo[], idx: number[], pair: Pair): [number, number] {
  const is = (family: string, style: string) => (f: FontInfo) => f.family === family && f.style === style
  if (pair === 'weight') {
    const weights = idx.map((i) => fonts[i].weight)
    const [lo, hi] = [Math.min(...weights), Math.max(...weights)]
    return [
      pick(fonts, idx, [is('Source Sans 3', 'Light'), (f) => f.weight === lo]),
      pick(fonts, idx, [is('Source Sans 3', 'Black'), (f) => f.weight === hi]),
    ]
  }
  if (pair === 'style')
    return [
      pick(fonts, idx, [is('Roboto', 'Light'), (f) => f.cls === 'sans']),
      pick(fonts, idx, [is('Bodoni Moda', 'Black'), (f) => f.cls === 'didone']),
    ]
  return [
    pick(fonts, idx, [is('Courier Prime', 'Regular'), (f) => f.cls === 'mono', (f) => f.cls === 'slab']),
    pick(fonts, idx, [is('Libre Baskerville', 'Italic'), (f) => f.italic, (f) => f.cls === 'didone']),
  ]
}

/**
 * Moves the latent point so that the P's foot-serif tip goes to the target while the left edge of the P's stem stays
 * where it is: gradient descent on the summed squared distances, from the current point. This is Campbell and Kautz's
 * editing rule in miniature. The dragged vertices pull on the latent point, and every vertex of every glyph follows the
 * manifold.
 */
function project(m: Model, dims: { tip: number; stem: number }, start: Point, target: Vec2): Point {
  const now = m.generate(start)
  const goals: [number, number][] = [
    [dims.tip, target[0]],
    [dims.tip + 1, target[1]],
    [dims.stem, now[dims.stem]],
    [dims.stem + 1, now[dims.stem + 1]],
  ]
  const at = (x: Point) =>
    m.p
      .meanAt(
        x,
        goals.map(([d]) => d),
      )
      .map((v, k) => m.mean[goals[k][0]] + m.scale * v)
  const cost = (x: Point) => at(x).reduce((s, v, k) => s + (v - goals[k][1]) ** 2, 0)
  let x: Point = [...start]
  let c = cost(x)
  let eta = 1e-4
  for (let it = 0; it < 40; it++) {
    const g = at(x)
    const grad: Point = [0, 0]
    goals.forEach(([d, v], k) => {
      const gd = m.p.meanGrad(x, d)
      grad[0] += 2 * m.scale * (g[k] - v) * gd[0]
      grad[1] += 2 * m.scale * (g[k] - v) * gd[1]
    })
    let accepted = false
    while (eta > 1e-12) {
      const cand: Point = [
        Math.min(Math.max(x[0] - eta * grad[0], -LIM), LIM),
        Math.min(Math.max(x[1] - eta * grad[1], -LIM), LIM),
      ]
      const cc = cost(cand)
      if (cc < c) {
        x = cand
        c = cc
        eta *= 1.5
        accepted = true
        break
      }
      eta /= 2
    }
    if (!accepted) break
  }
  return x
}

/** A manifold of real fonts, learned in the browser: drag the latent point, or drag the foot serif of the "P". */
export function FontManifold() {
  const { data, error } = useFigure<FontManifoldData>('manifold-of-fonts/glyphs')
  const [subset, setSubset] = useState<Subset>('all')
  const [pair, setPair] = useState<Pair>('style')
  const x1 = useParam(0, { min: -LIM, max: LIM, step: 0.01 })
  const x2 = useParam(0, { min: -LIM, max: LIM, step: 0.01 })
  const here: Point = [x1.value, x2.value]
  // The font the latent point is nearest: after a refit, the point moves to where that font now sits.
  const follow = useRef<string>('Libre Baskerville Regular')
  const fit = useManifoldFit(data, subset, (m) => {
    if (!data) return
    const k = m.idx.findIndex((i) => fontName(data.fonts[i]) === follow.current)
    const at = m.state.X[k < 0 ? 0 : k]
    x1.set(at[0])
    x2.set(at[1])
  })

  const glyphs = useMemo(() => new Map((data?.glyphs ?? []).map((g) => [g.char, g])), [data])
  const others = useMemo(
    () => (data ? data.glyphs.map((g) => g.char).filter((c) => !data.display.includes(c)) : []).join(''),
    [data],
  )
  const model = fit?.model

  // The nearest training font to the latent point: shown while the model is still fitting, and named in the readout.
  const nearest = useMemo(() => {
    if (!fit) return undefined
    let best = { i: fit.idx[0], d: Infinity }
    fit.X.forEach((p, k) => {
      const d = Math.hypot(p[0] - x1.value, p[1] - x2.value)
      if (d < best.d) best = { i: fit.idx[k], d }
    })
    return best
  }, [fit, x1.value, x2.value])
  useEffect(() => {
    if (data && nearest && fit?.model) follow.current = fontName(data.fonts[nearest.i])
  }, [data, nearest, fit])

  const vector = useMemo(() => {
    if (model) return model.generate([x1.value, x2.value])
    return data && nearest ? data.vectors[nearest.i] : undefined
  }, [model, data, nearest, x1.value, x2.value])

  const word = useMemo(
    () => (vector && data ? setLine(vector, glyphs, data.display) : undefined),
    [vector, data, glyphs],
  )
  const rest = useMemo(() => (vector ? setLine(vector, glyphs, others, 0) : undefined), [vector, glyphs, others])

  const ends = useMemo(() => (data && fit ? pairEnds(data.fonts, fit.idx, pair) : undefined), [data, fit, pair])
  const strip = useMemo(() => {
    if (!model || !ends || !data) return undefined
    const a = model.state.X[model.idx.indexOf(ends[0])]
    const b = model.state.X[model.idx.indexOf(ends[1])]
    const shapes: GlyphShape[] = []
    const path: Point[] = []
    for (let s = 0; s < STRIP; s++) {
      const u = s / (STRIP - 1)
      const x: Point = [a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1])]
      path.push(x)
      // Three words to a row, read left to right and then down.
      const line = setLine(model.generate(x), glyphs, data.display, -Math.floor(s / 3) * STRIP_ROW)
      shapes.push({ contours: line.contours.map((c) => c.map(([px, py]): Vec2 => [px + (s % 3) * STRIP_COL, py])) })
    }
    return { shapes, path }
  }, [model, ends, data, glyphs])

  if (error) return <p className="text-sm text-destructive">{error.message}</p>
  if (!data || !fit || !word || !rest || !nearest) {
    return (
      <Interactive title="A manifold of real fonts, learned in your browser">
        <p className="text-sm text-muted-foreground">Loading the fonts…</p>
      </Interactive>
    )
  }

  const P = glyphs.get('P')!
  const pAt = data.display.indexOf('P')
  const dims = { tip: P.offset + 2 * data.handle, stem: P.offset + 2 * data.anchor }
  const pShift = word.starts[pAt]
  const tip: Vec2 = [vector![dims.tip] + pShift, vector![dims.tip + 1]]

  const classOf = fit.idx.map((i) => CLASSES.indexOf(data.fonts[i].cls))
  const overlay: HeatmapOverlay[] = [
    {
      name: 'training font',
      type: 'scatter',
      x: fit.X.map((p) => p[0]),
      y: fit.X.map((p) => p[1]),
      group: classOf,
      groupNames: CLASS_NAMES,
    },
    ...(strip
      ? [
          {
            name: 'path',
            type: 'line' as const,
            x: strip.path.map((p) => p[0]),
            y: strip.path.map((p) => p[1]),
            emphasis: true,
            showPoints: true,
          },
        ]
      : []),
  ]
  const latentHandles: Handle[] = [
    {
      kind: 'point',
      at: here,
      label: 'font',
      onDrag: ([a, b]) => {
        x1.set(a)
        x2.set(b)
      },
    },
  ]
  const glyphHandles: Handle[] | undefined = model
    ? [
        {
          kind: 'point',
          at: tip,
          label: 'serif',
          onDrag: ([a, b]) => {
            const next = project(model, dims, [x1.value, x2.value], [a - pShift, b])
            x1.set(next[0])
            x2.set(next[1])
          },
        },
      ]
    : undefined

  const tourTo = (i: number) => {
    const k = fit.idx.indexOf(i)
    if (k < 0) return
    x1.set(fit.X[k][0])
    x2.set(fit.X[k][1])
  }
  const tour = [
    ['Roboto', 'Light'],
    ['Libre Baskerville', 'Italic'],
    ['Bodoni Moda', 'Black'],
    ['Courier Prime', 'Regular'],
  ].map(([family, style]) => data.fonts.findIndex((f) => f.family === family && f.style === style))

  const sd = model ? model.p.sd(here) / model.state.sf : undefined
  const wordX: [number, number] = [-80, 4200]
  const D = data.vectors[0].length

  return (
    <Interactive
      title="A manifold of real fonts, learned in your browser"
      caption={
        <>
          {fit.idx.length} font instances from the Google Fonts repository, each a vector of {D.toLocaleString()}{' '}
          numbers: {data.glyphs.length} capitals in dense correspondence plus their advance widths. The 2-D GP-LVM is
          fitted here, in your browser, starting from PCA; the fit is redone when the set of fonts changes. Left: the
          latent space, with the training fonts by design class and the posterior standard deviation of the mapping as a
          fraction of σ_f (dark is uncertain). Drag the round handle to generate the font at that location. Right: the
          generated word and the other capitals it was trained on. Drag the right foot serif of the “P”: the latent
          point moves to where the manifold puts the serif tip under the pointer while the left edge of the stem stays
          put, and every other letter changes with it. Below: the word generated at six evenly spaced points on the
          straight line between two training fonts, drawn as the path of diamonds on the left.
        </>
      }
      controls={
        <>
          <ParamSlider label="latent x₁" param={x1} />
          <ParamSlider label="latent x₂" param={x2} />
          <ParamChoice label="learn from" value={subset} onChange={setSubset} options={SUBSETS} />
          <div className="flex flex-wrap items-end gap-2 sm:col-span-2">
            {tour.map((i) => (
              <ParamButton key={i} onClick={() => tourTo(i)} disabled={i < 0 || !fit.idx.includes(i)}>
                {i >= 0 ? fontName(data.fonts[i]) : '—'}
              </ParamButton>
            ))}
            <ParamButton
              onClick={() => {
                x1.set(LIM)
                x2.set(LIM)
              }}
            >
              Far from every font
            </ParamButton>
          </div>
          <ParamChoice label="interpolate" value={pair} onChange={setPair} options={PAIRS} />
        </>
      }
      readout={
        <>
          <Readout
            label="fit"
            value={
              model
                ? `${ITERATIONS} steps in ${Math.round(fit.ms)} ms`
                : `fitting… step ${fit.iteration} of ${ITERATIONS}`
            }
          />
          <Readout label="posterior sd ÷ σ_f" value={sd === undefined ? '—' : formatNumber(sd)} />
          <Readout
            label="nearest training font"
            value={`${fontName(data.fonts[nearest.i])} (distance ${formatNumber(nearest.d)})`}
          />
        </>
      }
    >
      <div className="grid items-center gap-4 md:grid-cols-[6fr_5fr]">
        <Heatmap
          x={AXIS}
          y={AXIS}
          z={model?.sdGrid ?? FLAT}
          range={[0, 1]}
          scale="sequential"
          xLabel="x₁"
          yLabel="x₂"
          valueLabel="posterior sd ÷ σ_f"
          overlay={overlay}
          handles={latentHandles}
          height={330}
        />
        <div className="flex flex-col gap-3">
          <GlyphPlot
            shapes={[{ contours: word.contours }]}
            xRange={wordX}
            yRange={[-60, 800]}
            guides={[0, data.cap_height]}
            handles={glyphHandles}
            ariaLabel={`The word ${data.display} in the generated font`}
          />
          <GlyphPlot
            shapes={[{ contours: rest.contours, tone: 'muted' }]}
            xRange={[-80, Math.max(rest.width, 1) * 1.02]}
            yRange={[-260, 780]}
            ariaLabel="The other training capitals in the generated font"
          />
        </div>
      </div>
      {strip && (
        <div className="w-full">
          <GlyphPlot
            shapes={strip.shapes}
            xRange={[-100, 3 * STRIP_COL - 300]}
            yRange={[-STRIP_ROW - 120, 800]}
            ariaLabel="The word generated along a straight line between two training fonts"
          />
        </div>
      )}
    </Interactive>
  )
}

const FLAT = AXIS.map(() => AXIS.map(() => 1))
