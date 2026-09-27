import { useMemo } from 'react'
import {
  Heatmap,
  Interactive,
  ParamButton,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type HeatmapOverlay,
  type XYSeries,
} from '@/components/viz'
import { linspace } from '@/lib/math'
import { centre, fitGplvm, pcaScores, predictor, type Point } from '../_shared/gplvm'
import { contours, KIND_NAMES, makeFonts, measuredContrast, measuredSerif, SERIF_TIP } from './glyphs'

const LIM = 3
const GRID = 33
const AXIS = linspace(-LIM, LIM, GRID)
const GLYPH_X: [number, number] = [-0.45, 1.2]
const GLYPH_Y: [number, number] = [-0.12, 1.02]

/** Fits the toy manifold once: 20 fonts, 152 numbers each, standardised, 2-D GP-LVM from a PCA start. */
function buildManifold() {
  const fonts = makeFonts()
  const { Y: centred, mean } = centre(fonts.map((f) => f.vector))
  const scale = Math.sqrt(centred.flat().reduce((s, v) => s + v * v, 0) / centred.flat().length)
  const Y = centred.map((r) => r.map((v) => v / scale))
  const fit = fitGplvm(Y, pcaScores(Y), { iterations: 300, snapshotEvery: 300 })
  const state = fit.snapshots[fit.snapshots.length - 1]
  const p = predictor(Y, state)
  const glyph = (x: Point) => p.mean(x).map((v, d) => mean[d] + scale * v)
  const sdGrid = AXIS.map((b) => AXIS.map((a) => p.sd([a, b]) / state.sf))
  return { fonts, mean, scale, state, p, glyph, sdGrid }
}

/**
 * Moves the latent point so that the generated serif tip lands on the target while the stem edge under the serif stays
 * where it is, by gradient descent on the summed squared distances, starting from the current point. This is the
 * editing rule of Campbell and Kautz in miniature: the edited vertices pull on the latent point, and every other vertex
 * of every glyph follows the manifold.
 */
function project(m: ReturnType<typeof buildManifold>, start: Point, target: Point): Point {
  const tip = 2 * SERIF_TIP
  const stem = 2 * (SERIF_TIP - 2)
  const g0 = m.glyph(start)
  // Output index and desired value of each constrained coordinate.
  const goals: [number, number][] = [
    [tip, target[0]],
    [tip + 1, target[1]],
    [stem, g0[stem]],
    [stem + 1, g0[stem + 1]],
  ]
  const cost = (x: Point) => {
    const g = m.glyph(x)
    return goals.reduce((s, [d, v]) => s + (g[d] - v) ** 2, 0)
  }
  let x: Point = [...start]
  let c = cost(x)
  let eta = 1
  for (let it = 0; it < 40; it++) {
    const g = m.glyph(x)
    const grad: Point = [0, 0]
    for (const [d, v] of goals) {
      const gd = m.p.meanGrad(x, d)
      grad[0] += 2 * m.scale * (g[d] - v) * gd[0]
      grad[1] += 2 * m.scale * (g[d] - v) * gd[1]
    }
    let accepted = false
    while (eta > 1e-6) {
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

/** A toy font manifold: drag the latent point, or drag the serif tip of the "I" and let the manifold follow. */
export function FontManifold() {
  const m = useMemo(() => buildManifold(), [])
  const x1 = useParam(m.state.X[0][0], { min: -LIM, max: LIM, step: 0.01 })
  const x2 = useParam(m.state.X[0][1], { min: -LIM, max: LIM, step: 0.01 })
  const here: Point = [x1.value, x2.value]
  const glyph = useMemo(() => m.glyph([x1.value, x2.value]), [m, x1.value, x2.value])
  const sd = m.p.sd(here) / m.state.sf

  const nearest = useMemo(() => {
    let best = { i: 0, d: Infinity }
    m.state.X.forEach((p, i) => {
      const d = Math.hypot(p[0] - x1.value, p[1] - x2.value)
      if (d < best.d) best = { i, d }
    })
    return best
  }, [m, x1.value, x2.value])

  const overlay = useMemo(
    (): HeatmapOverlay[] => [
      {
        name: 'training font',
        type: 'scatter',
        x: m.state.X.map((p) => p[0]),
        y: m.state.X.map((p) => p[1]),
        group: m.fonts.map((f) => f.kind),
        groupNames: KIND_NAMES,
      },
    ],
    [m],
  )
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

  const series: XYSeries[] = contours(glyph).map((c) => ({
    name: 'glyph outline',
    type: 'line',
    x: c.map((p) => p[0]),
    y: c.map((p) => p[1]),
    slot: 0,
  }))
  const tip: Point = [glyph[2 * SERIF_TIP], glyph[2 * SERIF_TIP + 1]]
  const glyphHandles: Handle[] = [
    {
      kind: 'point',
      at: tip,
      label: 'serif tip',
      onDrag: (target) => {
        const next = project(m, [x1.value, x2.value], target)
        x1.set(next[0])
        x2.set(next[1])
      },
    },
  ]

  const tour = (i: number) => {
    x1.set(m.state.X[i][0])
    x2.set(m.state.X[i][1])
  }

  return (
    <Interactive
      title="A toy manifold of glyphs"
      caption="A stand-in for the paper's 46 real fonts: 20 synthetic fonts, each an “o” and an “I” drawn as 76 corresponding vertices (152 numbers), with a 2-D GP-LVM fitted in the browser. Left: the latent space, with the training fonts by style and the posterior standard deviation of the mapping as a fraction of σ_f (dark is uncertain). Drag the black point to generate the font at that location. Between fonts the glyphs interpolate smoothly; far from every font the mean returns to the average font and the standard deviation approaches 1. Right: drag the serif tip of the “I”. The latent point moves to where the manifold puts the tip under the pointer while the stem under the serif stays put, and the “o” changes with it, because in the training fonts long serifs come with high stroke contrast."
      controls={
        <>
          <ParamSlider label="latent x₁" param={x1} />
          <ParamSlider label="latent x₂" param={x2} />
          <div className="flex flex-wrap items-end gap-2">
            <ParamButton onClick={() => tour(0)}>A sans font</ParamButton>
            <ParamButton onClick={() => tour(12)}>A serif font</ParamButton>
            <ParamButton onClick={() => tour(4)}>An italic</ParamButton>
            <ParamButton
              onClick={() => {
                x1.set(LIM)
                x2.set(LIM)
              }}
            >
              Far corner
            </ParamButton>
          </div>
        </>
      }
      readout={
        <>
          <Readout label="posterior sd ÷ σ_f" value={formatNumber(sd)} />
          <Readout
            label="nearest training font"
            value={`${m.fonts[nearest.i].name} (distance ${formatNumber(nearest.d)})`}
          />
          <Readout label="serif length" value={formatNumber(measuredSerif(glyph) < 1e-3 ? 0 : measuredSerif(glyph))} />
          <Readout label="stroke contrast of the o" value={formatNumber(measuredContrast(glyph))} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Heatmap
          x={AXIS}
          y={AXIS}
          z={m.sdGrid}
          range={[0, 1]}
          scale="sequential"
          xLabel="x₁"
          yLabel="x₂"
          valueLabel="posterior sd ÷ σ_f"
          overlay={overlay}
          handles={latentHandles}
          height={360}
        />
        <XYChart series={series} handles={glyphHandles} xRange={GLYPH_X} yRange={GLYPH_Y} equalAspect />
      </div>
    </Interactive>
  )
}
