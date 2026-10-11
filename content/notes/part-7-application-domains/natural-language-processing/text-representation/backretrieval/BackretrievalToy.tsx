import { useMemo, useState } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Readout,
  type Segment,
  Segments,
  slider,
  useAxis,
  useFigureState,
  type Vec2,
  Vectors,
} from 'aifn-render'
import { normal, stream } from 'aifn-compute/foundation/random'

const RANGE: [number, number] = [-3.5, 3.5]
const UNIT: [number, number] = [0, 1]
const Y_FIT: [number | undefined, number | undefined] = [0, undefined]
// The swept "models": every combination of a rotation and a noise level between the two languages' embeddings.
const SWEEP_ANGLES = [0, 5, 10, 15, 20, 30, 45, 60, 90]
const SWEEP_NOISE = [0, 0.15, 0.3, 0.6]

type World = {
  /** Latent meanings of the source-language items and of the (different) target-language items. */
  zS: Vec2[]
  zT: Vec2[]
  /** Standard-normal draws, scaled by the noise sliders, so that every setting reuses the same randomness. */
  eTarget: Vec2[]
  eTranslation: Vec2[]
  eImageS: Vec2[]
  eImageT: Vec2[]
}

type Embedded = {
  textS: Vec2[]
  textT: Vec2[]
  /** Language-B translations of the source texts: the parallel data that backretrieval does not need. */
  translation: Vec2[]
  imageS: Vec2[]
  imageT: Vec2[]
}

const d2 = (u: Vec2, v: Vec2) => (u[0] - v[0]) ** 2 + (u[1] - v[1]) ** 2

function makeWorld(n: number, seed: number): World {
  const r = stream(seed)
  const draw = (): Vec2[] => Array.from({ length: n }, () => [normal(r), normal(r)])
  return { zS: draw(), zT: draw(), eTarget: draw(), eTranslation: draw(), eImageS: draw(), eImageT: draw() }
}

function embed(w: World, angleDeg: number, textNoise: number, imageNoise: number): Embedded {
  const t = (angleDeg * Math.PI) / 180
  const [c, s] = [Math.cos(t), Math.sin(t)]
  const lang = (z: Vec2, e: Vec2): Vec2 => [
    c * z[0] - s * z[1] + textNoise * e[0],
    s * z[0] + c * z[1] + textNoise * e[1],
  ]
  const view = (z: Vec2, e: Vec2): Vec2 => [z[0] + imageNoise * e[0], z[1] + imageNoise * e[1]]
  return {
    textS: w.zS,
    textT: w.zT.map((z, j) => lang(z, w.eTarget[j])),
    translation: w.zS.map((z, i) => lang(z, w.eTranslation[i])),
    imageS: w.zS.map((z, i) => view(z, w.eImageS[i])),
    imageT: w.zT.map((z, j) => view(z, w.eImageT[j])),
  }
}

/** Rank (1 = most similar) of `pool[target]` among `pool` by closeness to `from`. */
function rankOf(from: Vec2, pool: Vec2[], target: number): number {
  const d = d2(from, pool[target])
  let rank = 1
  for (const p of pool) if (d2(from, p) < d) rank++
  return rank
}

function nearest(from: Vec2, pool: Vec2[]): number {
  let best = 0
  for (let j = 1; j < pool.length; j++) if (d2(from, pool[j]) < d2(from, pool[best])) best = j
  return best
}

/** Ground-truth Recall@K with the true translations, and Backretrieval@K through the images. */
function metrics(e: Embedded, k: number) {
  let xlr = 0
  let bkr = 0
  e.textS.forEach((q, i) => {
    if (rankOf(q, e.translation, i) <= k) xlr++
    if (rankOf(e.imageT[nearest(q, e.textT)], e.imageS, i) <= k) bkr++
  })
  const n = e.textS.length
  return { xlr: xlr / n, bkr: bkr / n }
}

function pearson(x: number[], y: number[]): number {
  const mx = x.reduce((a, b) => a + b, 0) / x.length
  const my = y.reduce((a, b) => a + b, 0) / y.length
  let sxy = 0
  let sxx = 0
  let syy = 0
  x.forEach((xi, i) => {
    sxy += (xi - mx) * (y[i] - my)
    sxx += (xi - mx) ** 2
    syy += (y[i] - my) ** 2
  })
  return sxx === 0 || syy === 0 ? NaN : sxy / Math.sqrt(sxx * syy)
}

/** Ranks with ties given their average rank, as Spearman's coefficient requires. */
function ranks(x: number[]): number[] {
  const order = x.map((v, i) => [v, i] as const).sort((a, b) => a[0] - b[0])
  const out = new Array<number>(x.length)
  for (let i = 0; i < order.length;) {
    let j = i
    while (j + 1 < order.length && order[j + 1][0] === order[i][0]) j++
    for (let m = i; m <= j; m++) out[order[m][1]] = (i + j) / 2 + 1
    i = j + 1
  }
  return out
}

const spearman = (x: number[], y: number[]) => pearson(ranks(x), ranks(y))

/**
 * A toy world of image–text pairs in two languages. Each item has a latent meaning in the plane. Source texts sit at
 * their meanings; target-language texts are the meanings rotated and perturbed (the misalignment of the embedding model
 * under evaluation); images are the meanings perturbed by image-feature noise. The source and target collections
 * describe different items, as in a non-parallel corpus.
 */
export function BackretrievalToy() {
  const state = useFigureState({
    angle: slider(0, 90, 10, { step: 1, label: 'rotation between languages (°)' }),
    textNoise: float(0.2, { min: 0, max: 1, step: 0.05, label: 'cross-lingual noise' }),
    imageNoise: float(0.3, { min: 0, max: 1.2, step: 0.05, label: 'image-feature noise' }),
    n: int(120, { min: 20, max: 200, step: 10, label: 'items per language N' }),
    k: int(5, { min: 1, max: 20, step: 1, label: 'cut-off K' }),
    seed: int(1, { min: 1, max: 30, step: 1, label: 'seed' }),
  })
  const [query, setQuery] = useState(0)

  const world = useMemo(() => makeWorld(state.n, state.seed), [state.n, state.seed])
  const e = useMemo(
    () => embed(world, state.angle, state.textNoise, state.imageNoise),
    [world, state.angle, state.textNoise, state.imageNoise],
  )
  const current = useMemo(() => metrics(e, state.k), [e, state.k])

  // The sweep does not depend on the misalignment sliders, so dragging them only moves the marked point.
  const sweep = useMemo(() => {
    const points = SWEEP_NOISE.flatMap((noise, g) =>
      SWEEP_ANGLES.map((a) => ({ ...metrics(embed(world, a, noise, state.imageNoise), state.k), g })),
    )
    const x = points.map((p) => p.xlr)
    const y = points.map((p) => p.bkr)
    return { points, pearson: pearson(x, y), spearman: spearman(x, y) }
  }, [world, state.imageNoise, state.k])

  const q = Math.min(query, state.n - 1)
  const path = useMemo(() => {
    const retrieved = nearest(e.textS[q], e.textT)
    const from = e.imageT[retrieved]
    const byImage = e.imageS.map((p, i) => [d2(from, p), i] as const).sort((a, b) => a[0] - b[0])
    const topK = byImage.slice(0, state.k).map(([, i]) => i)
    return { retrieved, from, topK, rank: rankOf(from, e.imageS, q) }
  }, [e, q, state.k])

  const textSeries = useMemo(
    () =>
      [
        {
          name: 'source texts (language A)',
          x: e.textS.map((p) => p[0]),
          y: e.textS.map((p) => p[1]),
          slot: 0,
        },
        {
          name: 'target texts (language B)',
          x: e.textT.map((p) => p[0]),
          y: e.textT.map((p) => p[1]),
          slot: 1,
        },
        {
          name: 'true translation of the query (unused)',
          x: [e.translation[q][0]],
          y: [e.translation[q][1]],
          slot: 2,
        },
        { name: 'query', x: [e.textS[q][0]], y: [e.textS[q][1]], emphasis: true },
      ] as const,
    [e, q],
  )
  const textArrow: Segment[] = useMemo(() => [{ from: e.textS[q], to: e.textT[path.retrieved] }], [e, q, path])

  const imageSeries = useMemo(
    () =>
      [
        { name: 'source images', x: e.imageS.map((p) => p[0]), y: e.imageS.map((p) => p[1]), slot: 0 },
        { name: 'target images', x: e.imageT.map((p) => p[0]), y: e.imageT.map((p) => p[1]), slot: 1 },
        { name: "query's image", x: [e.imageS[q][0]], y: [e.imageS[q][1]], emphasis: true },
      ] as const,
    [e, q],
  )
  const topKSegments: Segment[] = useMemo(() => path.topK.map((i) => ({ from: path.from, to: e.imageS[i] })), [path, e])
  const imageArrow: Segment[] = useMemo(() => [{ from: path.from, to: e.imageS[q] }], [path, e, q])

  const sweepSeries = useMemo(
    () =>
      [
        {
          name: 'random embedding: K / N',
          x: [0, 1],
          y: [state.k / state.n, state.k / state.n],
          muted: true,
          dashed: true,
        },
        {
          name: 'swept models',
          x: sweep.points.map((p) => p.xlr),
          y: sweep.points.map((p) => p.bkr),
          group: sweep.points.map((p) => p.g),
          groupNames: SWEEP_NOISE.map((s) => `noise ${s}`),
        },
        { name: 'current setting', x: [current.xlr], y: [current.bkr], emphasis: true },
      ] as const,
    [sweep, current, state.k, state.n],
  )

  const xAxis = useAxis({ range: RANGE })
  const yAxis = useAxis({ range: RANGE, equal: xAxis })
  const xAxis2 = useAxis({ range: RANGE })
  const yAxis2 = useAxis({ range: RANGE, equal: xAxis2 })
  const xAxis3 = useAxis({ label: 'ground truth Recall@K (true translations)', range: UNIT })
  const yAxis3 = useAxis({ label: 'Backretrieval@K', range: Y_FIT })
  return (
    <Figure
      title="Backretrieval in a toy world of captioned images"
      state={state}
      caption="Each item has a latent meaning in the plane. Source texts (language A) sit at their meanings. Target texts (language B) describe different items, and the embedding model under test rotates and perturbs them. Images are the meanings plus image-feature noise; similarity is negative squared distance. Drag the query (or press anywhere in the left panel) to pick the nearest source text. Left: the arrow goes from the query to its nearest target text. Right: from that text's image, thin lines reach the K most similar source images, and the arrow reaches the query's own image; the query scores a hit when its image is among those K. Bottom: each point is one model from a grid of rotations and noise levels, with the current setting in ink. Backretrieval never sees the true translation, yet it ranks the models almost as the ground truth does."

      readouts={
        <>
          <Readout label="ground truth Recall@K" value={formatNumber(current.xlr)} />
          <Readout label="Backretrieval@K" value={formatNumber(current.bkr)} />
          <Readout label="random K / N" value={formatNumber(state.k / state.n)} />
          <Readout label="query image rank" value={path.rank} />
          <Readout label="Pearson over models" value={formatNumber(sweep.pearson)} />
          <Readout label="Spearman over models" value={formatNumber(sweep.spearman)} />
        </>
      }
    >
      <div className="flex flex-col gap-4">
        <div className="grid gap-4 md:grid-cols-2">
          <div>
            <div className="text-xs text-muted-foreground">Text embeddings f (steps 1–2)</div>
            <Plot
              x={xAxis}
              y={yAxis}
              ariaLabel={'Source and target text embeddings with the query and its retrieved target text'}
            >
              <Points {...textSeries[0]} />
              <Points {...textSeries[1]} />
              <Points {...textSeries[2]} />
              <Points {...textSeries[3]} />
              <Vectors vectors={textArrow} />
              <Handle kind="point" at={e.textS[q]} label="query" onDrag={(p) => setQuery(nearest(p, e.textS))} />
            </Plot>
          </div>
          <div>
            <div className="text-xs text-muted-foreground">Image features g (steps 3–4)</div>
            <Plot
              x={xAxis2}
              y={yAxis2}
              ariaLabel={"Image features, the K source images nearest the retrieved image, and the query's image"}
            >
              <Points {...imageSeries[0]} />
              <Points {...imageSeries[1]} />
              <Points {...imageSeries[2]} />
              <Segments segments={topKSegments} />
              <Vectors vectors={imageArrow} />
            </Plot>
          </div>
        </div>
        <Plot
          x={xAxis3}
          y={yAxis3}
          height={300}
          ariaLabel={'Backretrieval against ground-truth recall for a grid of models'}
        >
          <Curve {...sweepSeries[0]} />
          <Points {...sweepSeries[1]} />
          <Points {...sweepSeries[2]} />
        </Plot>
      </div>
    </Figure>
  )
}
