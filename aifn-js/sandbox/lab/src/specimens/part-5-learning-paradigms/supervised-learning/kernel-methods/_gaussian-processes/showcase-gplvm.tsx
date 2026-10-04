import { FONT_CLASSES, fonts, glyphContours, type FontDataset, type GlyphLayout } from 'aifn-methods/data/real/fonts'
import { gplvmFitSteps, gplvmModel, gplvmProblem, type GplvmState } from 'aifn-methods/learning/gaussian-processes'
import { fromData, toFlat } from 'aifn/foundation/tensor'
import { extend, now, trace, type Trace } from 'aifn/foundation/trace'
import { matern32, matern52, rbf } from 'aifn/learning/kernels'
import { useEffect, useMemo, useRef, useState } from 'react'
import { Button, Player } from '@lab/controls'
import { ControlRow, Dashboard, DashboardCell, DashboardRow, Figure } from '@lab/layout'
import { choice, row, slider, useFigureState, when, type FigureState } from '@lab/state'
import { Curve, Handle, Plot, Points, Raster, Readout, useAxis, type PlotPointer, type Vec2 } from '@lab/viz'
import { formatValue } from '@lab/views'

const f3 = (v: number) => formatValue(Number(v.toPrecision(3)))

const WORDS = ['GPLVM', 'A', 'G', 'R', 'HAMBURG', 'ABCDEFGHIJLMNOPRSTUVXYZ'] as const
type Word = (typeof WORDS)[number]
const KERNELS = {
  RBF: rbf,
  'Matérn 5/2': matern52,
  'Matérn 3/2': matern32,
} as const
type KernelName = keyof typeof KERNELS

/** L-BFGS steps per press of "fit". */
const FIT_STEPS = 150
/** Work per animation frame while fitting, so the page keeps painting. */
const FRAME_BUDGET_MS = 12
const RECORD = { logPosterior: (s: GplvmState) => s.logPosterior }
/** Cells per side of the predictive-sd background; it is redrawn only when the fit is not running (≈ 25 ms). */
const GRID = 36
const CLASS_NAMES = [...FONT_CLASSES]
const KERNEL_NAMES = Object.keys(KERNELS) as KernelName[]

// ── Drawing glyphs: an SVG path per contour, set along a line by the vector's own advance widths ─────────────────────

/** SVG path data of a word set from one font vector (font units, y up; the SVG flips it). */
function wordPath(row: ArrayLike<number>, glyphs: readonly GlyphLayout[]): { d: string; width: number } {
  let x0 = 0
  const parts: string[] = []
  for (const g of glyphs) {
    for (const c of glyphContours(row, g)) {
      let s = ''
      for (let k = 0; k < c.x.length; k++)
        s += `${k === 0 ? 'M' : 'L'}${(c.x[k] + x0).toFixed(1)} ${(-c.y[k]).toFixed(1)}`
      parts.push(s + 'Z')
    }
    x0 += g.advance >= 0 ? row[g.advance] : 700
  }
  return { d: parts.join(''), width: x0 }
}

function GlyphWord({
  row,
  glyphs,
  capHeight,
  label,
  strong,
  large = strong,
}: {
  row: ArrayLike<number>
  glyphs: readonly GlyphLayout[]
  capHeight: number
  label: string
  /** Ink rather than muted: a font the GPLVM draws, not a real one beside it. */
  strong?: boolean
  /** Twice the height (default: when strong). */
  large?: boolean
}) {
  const { d, width } = wordPath(row, glyphs)
  const pad = 0.25 * capHeight
  return (
    <div className="flex min-w-0 flex-col gap-1">
      <svg
        viewBox={`${-pad} ${-capHeight - pad} ${width + 2 * pad} ${capHeight + 2 * pad + 0.25 * capHeight}`}
        className={large ? 'h-24 w-full' : 'h-12 w-full'}
        preserveAspectRatio="xMinYMid meet"
        role="img"
        aria-label={label}
      >
        <path d={d} fillRule="evenodd" className={strong ? 'fill-foreground' : 'fill-muted-foreground'} />
      </svg>
      <div className="truncate text-xs text-muted-foreground">{label}</div>
    </div>
  )
}

// ── The figure ──────────────────────────────────────────────────────────────────────────────────────────────────────

const SCHEMA = {
  data: row('1 · data and kernel', {
    word: choice(WORDS, 'GPLVM', { label: 'characters' }),
    kernel: choice(KERNEL_NAMES, 'RBF', { label: 'kernel' }),
  }),
  explore: row('2 · explore the latent space', {
    mode: choice(['point', 'path'] as const, 'point', { label: 'show' }),
    count: slider(3, 12, 6, { label: 'fonts along the path', step: 1, when: when('mode', 'path') }),
  }),
}

/** One pinned latent point: at a real font's latent position (it moves with the fit) or wherever it was dropped. */
type Pin = { font: number } | { at: Vec2 }

export function GplvmFonts() {
  const figure = useFigureState(SCHEMA)
  const word = figure.data.word as Word
  const kernelName = figure.data.kernel as KernelName
  // A new dataset or kernel is a new problem: the body remounts, so the fit, the player and the latent point start over.
  return <GplvmBody key={`${word}:${kernelName}`} figure={figure} word={word} kernelName={kernelName} />
}

function GplvmBody({
  figure,
  word,
  kernelName,
}: {
  figure: FigureState<typeof SCHEMA>
  word: Word
  kernelName: KernelName
}) {
  const ds: FontDataset = useMemo(() => fonts({ chars: word, advances: true }), [word])
  const names = useMemo(() => ds.meta.fonts.map((f) => `${f.family} ${f.style}`), [ds])
  const problem = useMemo(() => gplvmProblem(ds.x, { kernel: KERNELS[kernelName] }), [ds, kernelName])
  const alg = useMemo(() => gplvmFitSteps(problem), [problem])

  // ── The fit: a trace that starts at PCA (step 0) and grows by `extend`, a few steps per animation frame ──
  const [tr, setTr] = useState<Trace<GplvmState>>(() => trace(alg, undefined, 0, { record: RECORD }))
  const [pos, setPos] = useState(0)
  const [fitting, setFitting] = useState(false)
  const run = useRef(0)
  // Unmounting stops a fit in progress.
  useEffect(
    () => () => {
      run.current++
    },
    [],
  )
  const fit = () => {
    const id = ++run.current
    const target = tr.meta.steps + FIT_STEPS
    let current = tr
    let chunk = 1
    setFitting(true)
    const frame = () => {
      if (run.current !== id) return
      const start = now()
      current = extend(current, alg, Math.min(chunk, target - current.meta.steps), { record: RECORD })
      const ms = now() - start
      // Size the next chunk to the frame budget from this one's cost per step.
      chunk = Math.max(1, Math.min(20, Math.round((chunk * FRAME_BUDGET_MS) / Math.max(ms, 0.5))))
      setTr(current)
      setPos(current.meta.steps)
      if (current.meta.steps < target && current.meta.stopped === 'limit') requestAnimationFrame(frame)
      else setFitting(false)
    }
    requestAnimationFrame(frame)
  }
  const at = Math.min(pos, tr.meta.steps)
  const state = tr.steps[at] ?? tr.final
  const model = useMemo(() => gplvmModel(problem, state), [problem, state])
  const latent = useMemo(() => {
    const z = toFlat(state.latent)
    return {
      x: Array.from({ length: problem.n }, (_, i) => z[2 * i]),
      y: Array.from({ length: problem.n }, (_, i) => z[2 * i + 1]),
    }
  }, [state, problem])

  // ── Latent box: held, growing to hold every step seen (a new dataset or kernel refits it) ──
  const extent = Math.max(...latent.x.map(Math.abs), ...latent.y.map(Math.abs))
  const need = Math.max(Math.ceil((extent * 1.15) / 0.5) * 0.5, 2)
  const [held, setHeld] = useState(need)
  // Grown during render (React's pattern for state derived from earlier renders); a new problem remounts the body.
  if (need > held) setHeld(need)
  const half = Math.max(held, need)

  // While fitting, the last background stays: the points and the curve move every frame, the raster waits.
  const fresh = useMemo(() => {
    if (fitting) return null
    const ticks = Array.from({ length: GRID }, (_, k) => -half + (2 * half * (k + 0.5)) / GRID)
    const pts = new Float64Array(GRID * GRID * 2)
    for (let i = 0; i < GRID; i++)
      for (let j = 0; j < GRID; j++) {
        pts[2 * (i * GRID + j)] = ticks[j]
        pts[2 * (i * GRID + j) + 1] = ticks[i]
      }
    const v = toFlat(model.variance(fromData(pts, [GRID * GRID, 2])))
    // As a fraction of the prior variance σ_f² (in data units): 0 at the fonts, 1 far from them.
    const prior = model.signalVariance * problem.scale * problem.scale
    const z = Array.from({ length: GRID }, (_, i) =>
      Array.from({ length: GRID }, (_, j) => Math.sqrt(v[i * GRID + j] / prior)),
    )
    return { x: ticks, y: ticks, z }
  }, [model, half, problem, fitting])
  const [kept, setKept] = useState(fresh)
  if (fresh && fresh !== kept) setKept(fresh)
  const background = fresh ?? kept ?? { x: [], y: [], z: [] }

  // ── The reader's latent point: pinned to a font until dragged ──
  const start = Math.max(0, names.indexOf('Libre Baskerville Regular'))
  const [pin, setPin] = useState<Pin>({ font: start })
  const place = (p: Pin): Vec2 => ('font' in p ? [latent.x[p.font], latent.y[p.font]] : p.at)
  const clampTo = ([a, b]: Vec2): Vec2 => [Math.max(-half, Math.min(half, a)), Math.max(-half, Math.min(half, b))]
  const point = place(pin)

  // ── The path mode: two pinned ends and the fonts the map draws at evenly spaced points on the segment between them ──
  const pathMode = figure.explore.mode === 'path'
  const count = Math.round(figure.explore.count)
  // The path starts at the reader's point font and ends, by default, at the font farthest from it in the PCA scores.
  const [ends, setEnds] = useState<[Pin, Pin]>(() => {
    let far = start
    latent.x.forEach((x, i) => {
      const d = (x - latent.x[start]) ** 2 + (latent.y[i] - latent.y[start]) ** 2
      const best = (latent.x[far] - latent.x[start]) ** 2 + (latent.y[far] - latent.y[start]) ** 2
      if (d > best) far = i
    })
    return [{ font: start }, { font: far }]
  })
  const a = place(ends[0])
  const b = place(ends[1])
  const path = useMemo(() => {
    if (!pathMode) return null
    const pts = new Float64Array(2 * count)
    for (let k = 0; k < count; k++) {
      const t = k / (count - 1)
      pts[2 * k] = a[0] + t * (b[0] - a[0])
      pts[2 * k + 1] = a[1] + t * (b[1] - a[1])
    }
    const at = fromData(pts, [count, 2])
    const rows = toFlat(model.project(at))
    const v = toFlat(model.variance(at))
    const prior = model.signalVariance * problem.scale ** 2
    return Array.from({ length: count }, (_, k) => ({
      t: k / (count - 1),
      x: pts[2 * k],
      y: pts[2 * k + 1],
      row: rows.slice(k * problem.d, (k + 1) * problem.d),
      sd: Math.sqrt(v[k] / prior),
    }))
  }, [pathMode, count, model, problem, a[0], a[1], b[0], b[1]]) // eslint-disable-line react-hooks/exhaustive-deps
  const nearestFont = ([x, y]: Vec2) =>
    latent.x.reduce(
      (best, xi, i) =>
        (xi - x) ** 2 + (latent.y[i] - y) ** 2 < (latent.x[best] - x) ** 2 + (latent.y[best] - y) ** 2 ? i : best,
      0,
    )
  const pathMarks = useMemo(() => (path ? { x: path.map((p) => p.x), y: path.map((p) => p.y) } : null), [path])
  const projected = useMemo(() => toFlat(model.project([point[0], point[1]])), [model, point[0], point[1]]) // eslint-disable-line react-hooks/exhaustive-deps
  const sdHere = useMemo(
    () => Math.sqrt(toFlat(model.variance([point[0], point[1]]))[0] / (model.signalVariance * problem.scale ** 2)),
    [model, problem, point[0], point[1]], // eslint-disable-line react-hooks/exhaustive-deps
  )
  const nearest = useMemo(
    () =>
      latent.x
        .map((x, i) => ({ i, d: Math.hypot(x - point[0], latent.y[i] - point[1]) }))
        .sort((a, b) => a.d - b.d)
        .slice(0, 3),
    [latent, point[0], point[1]], // eslint-disable-line react-hooks/exhaustive-deps
  )

  // ── Hover: the font under the pointer ──
  const [hover, setHover] = useState<number | null>(null)
  const onPointer = (e: PlotPointer) => {
    if (e.type !== 'move') return setHover(null)
    const r = (0.04 * 2 * half) ** 2
    let best = -1
    let bestD = r
    latent.x.forEach((x, i) => {
      const d = (x - e.point[0]) ** 2 + (latent.y[i] - e.point[1]) ** 2
      if (d < bestD) [best, bestD] = [i, d]
    })
    setHover(best < 0 ? null : best)
  }

  const classOf = useMemo(() => Array.from(toFlat(ds.y!)), [ds])
  const italic = useMemo(() => {
    const idx = ds.meta.fonts.flatMap((f, i) => (f.italic ? [i] : []))
    return {
      x: idx.map((i) => latent.x[i]),
      y: idx.map((i) => latent.y[i]),
      // Transparent fill with the ink outline: a ring around the font's own mark.
      colors: idx.map(() => 'rgba(0,0,0,0)'),
    }
  }, [ds, latent])

  // Relative to PCA (step 0): the absolute value is about −10⁵, whose ticks all round to the same label.
  const gain = useMemo(() => {
    const lp = toFlat(tr.series.logPosterior)
    return { x: Array.from(tr.index), y: Array.from(lp, (v) => v - lp[0]) }
  }, [tr])
  const shown = useMemo(() => ({ x: [at], y: [gain.y[at]] }), [at, gain])

  const z1 = useAxis({ label: 'latent x₁', range: [-half, half], nice: false })
  const z2 = useAxis({ label: 'latent x₂', range: [-half, half], nice: false, equal: z1 })
  const steps = useAxis({
    label: 'L-BFGS step',
    range: [0, Math.max(FIT_STEPS, Math.ceil(tr.meta.steps / FIT_STEPS) * FIT_STEPS)],
  })
  const gainAxis = useAxis({ label: 'log posterior − PCA’s', range: [0, undefined] })

  const glyphs = ds.meta.glyphs
  const flat = useMemo(() => Float64Array.from(toFlat(ds.x)), [ds])
  const rowOf = (i: number) => flat.subarray(i * problem.d, (i + 1) * problem.d)
  const cap = ds.meta.capHeight
  const hovered = hover ?? null
  return (
    <Figure
      title="A GPLVM manifold of fonts"
      purpose="A two-dimensional GPLVM maps every latent point to a whole font: dragging a point through the latent space draws fonts between the real ones, and the predictive sd shows where the map is guessing."
      state={figure}
      defaultSize="XL"
      controls={
        <>
          <ControlRow label="3 · fit (MAP, L-BFGS from PCA)">
            <div className="flex items-end gap-2">
              <Button
                variant="outline"
                size="sm"
                aria-label="Fit"
                onClick={fit}
                disabled={fitting || tr.meta.stopped !== 'limit'}
              >
                {fitting ? 'fitting…' : tr.meta.steps === 0 ? `fit ${FIT_STEPS} steps` : `${FIT_STEPS} more steps`}
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setTr(trace(alg, undefined, 0, { record: RECORD }))}
                disabled={fitting || tr.meta.steps === 0}
              >
                back to PCA
              </Button>
            </div>
            {tr.meta.steps > 0 && (
              <Player
                value={at}
                onChange={setPos}
                count={tr.meta.steps + 1}
                label="L-BFGS step"
                format={(p) => (p === 0 ? 'PCA' : `step ${p}`)}
              />
            )}
          </ControlRow>
        </>
      }
      readouts={
        <>
          {tr.meta.steps > 0 && (
            <Readout
              label="step"
              value={`${at} of ${tr.meta.steps}${tr.meta.stopped === 'done' ? ' (converged)' : ''}`}
            />
          )}
          {/* The gain over PCA, as the chart draws it: the absolute log posterior is about −10⁵ and reads as noise. */}
          <Readout label="log posterior − PCA’s" value={f3(gain.y[at] ?? 0)} />
          <Readout label="lengthscale ℓ" value={f3(state.lengthscale)} />
          <Readout label="noise σ² / σ_f²" value={f3(state.noiseVariance / state.signalVariance)} />
          {path ? (
            <>
              <Readout label="path" value={`(${f3(a[0])}, ${f3(a[1])}) → (${f3(b[0])}, ${f3(b[1])})`} />
              <Readout label="largest predictive sd / σ_f on it" value={f3(Math.max(...path.map((p) => p.sd)))} />
            </>
          ) : (
            <>
              <Readout label="latent point" value={`(${f3(point[0])}, ${f3(point[1])})`} />
              <Readout label="predictive sd / σ_f there" value={f3(sdHere)} />
            </>
          )}
        </>
      }
      caption={`${problem.n} fonts (Google Fonts, OFL and Apache 2.0), each the vector of ${problem.d.toLocaleString()} numbers that outline "${word}" in dense correspondence plus the advance widths. Press "fit" to run L-BFGS on the GPLVM's log posterior from the PCA scores; the curve (the log posterior's gain over PCA) grows as it runs and the player scrubs back through the steps. Left: the latent space, fonts coloured and shaped by design class, italics ringed, over the predictive sd of the map as a fraction of σ_f (0 at the fonts, 1 far from them). Drag the round handle, or press anywhere, to move the latent point: the large word is the GPLVM's mean font there, and below it are the three nearest real fonts. With "show: path", drag the start and end handles instead: the strip below the latent space sets the GPLVM's mean font at evenly spaced points on the segment between them, each with its predictive sd, so the morph is smooth where the path stays near real fonts and drifts towards the mean font where it crosses empty space. Hover a font to see its name and outline.`}
    >
      <Dashboard>
        <DashboardRow ratio={1.4} minHeight={380}>
          <DashboardCell ratio={1.1}>
            <Plot x={z1} y={z2} onPointer={onPointer}>
              <Raster
                x={background.x}
                y={background.y}
                z={background.z}
                range={[0, 1]}
                fillOpacity={0.7}
                valueLabel="predictive sd / σ_f"
              />
              <Points
                name="font (by design class)"
                x={latent.x}
                y={latent.y}
                group={classOf}
                groupNames={CLASS_NAMES}
              />
              <Points name="italic" x={italic.x} y={italic.y} colors={italic.colors} size={15} />
              {pathMarks ? (
                <>
                  <Curve name="path" x={[a[0], b[0]]} y={[a[1], b[1]]} dashed emphasis silent />
                  <Points name="fonts along the path" x={pathMarks.x} y={pathMarks.y} emphasis size={7} />
                  <Handle
                    kind="point"
                    at={a}
                    label="start"
                    onDrag={(p) => setEnds(([, e]) => [{ at: clampTo(p) }, e])}
                  />
                  <Handle kind="point" at={b} label="end" onDrag={(p) => setEnds(([s]) => [s, { at: clampTo(p) }])} />
                </>
              ) : (
                <Handle kind="point" at={point} label="font" onDrag={(p) => setPin({ at: clampTo(p) })} />
              )}
            </Plot>
          </DashboardCell>
          <DashboardCell ratio={1}>
            <div className="flex h-full flex-col justify-between gap-3 overflow-hidden">
              {path ? (
                // The path's two ends, large, each beside the real font nearest to it.
                ([0, count - 1] as const).map((k) => {
                  const near = nearestFont([path[k].x, path[k].y])
                  return (
                    <GlyphWord
                      key={k}
                      row={path[k].row}
                      glyphs={glyphs}
                      capHeight={cap}
                      label={`${k === 0 ? 'start' : 'end'} at (${f3(path[k].x)}, ${f3(path[k].y)}) · nearest real font ${names[near]}`}
                      strong
                    />
                  )
                })
              ) : (
                <>
                  <GlyphWord
                    row={projected}
                    glyphs={glyphs}
                    capHeight={cap}
                    label={`GPLVM mean at (${f3(point[0])}, ${f3(point[1])})`}
                    strong
                  />
                  <div className="grid grid-cols-3 gap-3">
                    {nearest.map(({ i }) => (
                      <GlyphWord key={i} row={rowOf(i)} glyphs={glyphs} capHeight={cap} label={names[i]} />
                    ))}
                  </div>
                </>
              )}
              <div className="min-h-20 rounded-md border border-dashed p-2">
                {hovered !== null ? (
                  <GlyphWord
                    row={rowOf(hovered)}
                    glyphs={glyphs}
                    capHeight={cap}
                    label={`${names[hovered]} · ${ds.meta.fonts[hovered].cls}${ds.meta.fonts[hovered].italic ? ' · italic' : ''}`}
                  />
                ) : (
                  <div className="text-xs text-muted-foreground">Hover a font in the latent space to see it here.</div>
                )}
              </div>
            </div>
          </DashboardCell>
        </DashboardRow>
        {path && (
          <DashboardRow ratio={0.3} minHeight={95 * Math.ceil(count / 6)}>
            <DashboardCell>
              <div className="grid h-full grid-cols-[repeat(auto-fill,minmax(9rem,1fr))] content-start gap-3 overflow-auto">
                {path.map((p, k) => (
                  <GlyphWord
                    key={k}
                    row={p.row}
                    glyphs={glyphs}
                    capHeight={cap}
                    label={`${k === 0 ? 'start' : k === count - 1 ? 'end' : `t = ${f3(p.t)}`} · sd ${f3(p.sd)}`}
                    strong
                    large={false}
                  />
                ))}
              </div>
            </DashboardCell>
          </DashboardRow>
        )}
        <DashboardRow ratio={0.5} minHeight={170}>
          <DashboardCell>
            {tr.meta.steps > 0 ? (
              <Plot x={steps} y={gainAxis} legend={false}>
                <Curve name="log posterior gain" x={gain.x} y={gain.y} slot={0} />
                <Points name="shown step" x={shown.x} y={shown.y} emphasis />
              </Plot>
            ) : (
              // Before a fit there is no path, only PCA's point: say what will be drawn here instead of an empty frame.
              <div className="flex h-full items-center justify-center rounded-md border border-dashed p-4 text-center text-sm text-muted-foreground">
                Press “fit” to run L-BFGS from the PCA scores: the gain in log posterior over PCA is drawn here, step by
                step.
              </div>
            )}
          </DashboardCell>
        </DashboardRow>
      </Dashboard>
    </Figure>
  )
}
