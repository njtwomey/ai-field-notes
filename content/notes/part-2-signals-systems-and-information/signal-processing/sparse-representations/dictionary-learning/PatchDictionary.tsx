import { useMemo, useState } from 'react'
import { child, stream } from 'aifn-compute/foundation/random'
import { toArray } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { dictionaryLearningSteps, sparseCode } from 'aifn-compute/signal/sparse'
import {
  choice,
  Curve,
  Figure,
  formatNumber,
  Handle,
  int,
  Player,
  Plot,
  Points,
  Raster,
  Readout,
  Segments,
  slider,
  useAxis,
  useComputed,
  useElementSize,
  useFigureState,
} from 'aifn-render'
import { FixedHeight } from '../_shared/FixedHeight'
import {
  dctDictionary,
  fromTiles,
  imageRows,
  makeImage,
  mosaic,
  mosaicColumns,
  P,
  tiles,
  trainingPatches,
  W,
} from './patches'

const TRAIN = 300
const T = W / P
const IMAGE = makeImage()
const TILES = tiles(IMAGE)
const DCT = dctDictionary()
const PIXELS = Array.from({ length: W }, (_, i) => i + 0.5)

type Box = { from: [number, number]; to: [number, number] }

/** The four sides of the rectangle from (x0, y0) to (x1, y1). */
const box = (x0: number, y0: number, x1: number, y1: number): Box[] => [
  { from: [x0, y0], to: [x1, y0] },
  { from: [x1, y0], to: [x1, y1] },
  { from: [x1, y1], to: [x0, y1] },
  { from: [x0, y1], to: [x0, y0] },
]

/** Charts inside take this height rather than the figure frame's. */
const rmse = (img: Float64Array) => Math.sqrt(img.reduce((s, v, i) => s + (v - IMAGE[i]) ** 2, 0) / (W * W))

/** Code every tile over D with s atoms and rebuild the image. */
function reconstruct(D: number[][], s: number) {
  const X = toArray(sparseCode(D, TILES.Y, { method: 'omp', sparsity: s }).X) as number[][]
  const img = fromTiles(D, X, TILES.means)
  return { X, img, rows: imageRows(img), error: rmse(img) }
}

/** The boxes around the atoms tile i uses, in a mosaic laid out in `order`. */
function usedBoxes(X: number[][], i: number, order: readonly number[]): Box[] {
  const cols = mosaicColumns(order.length)
  return order.flatMap((j, n) => {
    if (X[j][i] === 0) return []
    const x0 = (n % cols) * (P + 1) - 1
    const y0 = Math.floor(n / cols) * (P + 1) - 1
    return box(x0, y0, x0 + P + 1, y0 + P + 1)
  })
}

/** Equal-unit axes for a mosaic of k atoms, rows downwards. */
function useMosaicAxes(k: number) {
  const cols = mosaicColumns(k)
  const w = cols * (P + 1) - 1
  const h = Math.ceil(k / cols) * (P + 1) - 1
  const x = useAxis({ range: [-1.5, w + 0.5], zoom: false, format: () => '', key: k })
  const y = useAxis({ range: [-1.5, h + 0.5], zoom: false, format: () => '', inverse: true, equal: x, key: k })
  return { x, y, cx: Array.from({ length: w }, (_, j) => j), cy: Array.from({ length: h }, (_, i) => i) }
}

export function PatchDictionary() {
  const state = useFigureState({
    k: int(24, { min: 8, max: 48, label: 'atoms k', suggestions: [16, 24, 36, 48] }),
    s: int(2, { min: 1, max: 4, label: 'atoms per patch s' }),
    update: choice(
      [
        { value: 'ksvd', label: 'K-SVD' },
        { value: 'mod', label: 'method of optimal directions' },
      ],
      'ksvd',
      { label: 'dictionary update' },
    ),
    rounds: int(12, { min: 1, max: 40, label: 'rounds', suggestions: [5, 12, 25] }),
    start: int(1, { min: 1, max: 9999, label: 'start seed' }),
    cx: slider(3, W - 3, 21, { step: P, onChart: true }),
    cy: slider(3, W - 3, 27, { step: P, onChart: true }),
  })
  const { k, s, update, rounds, start, cx, cy } = state

  const run = useComputed(() => {
    const root = stream(start)
    const { Y } = trainingPatches(IMAGE, TRAIN, child(root, 'patches'))
    const alg = dictionaryLearningSteps(Y, { atoms: k, sparsity: s, update })
    const tr = trace(alg, undefined, rounds, { stream: root })
    const states = tr.steps.map((st) => ({ D: toArray(st.D) as number[][], objective: st.objective }))
    // Show the atoms in order of use at the end, most used first; K-SVD and MOD keep each atom's index across rounds.
    const X = toArray(tr.final.X) as number[][]
    const use = X.map((row) => row.filter((v) => v !== 0).length)
    const order = Array.from({ length: k }, (_, j) => j).sort((a, b) => use[b] - use[a] || a - b)
    return { states, order }
  }, [k, s, update, rounds, start])
  const { states, order } = run.value

  const key = `${k}|${s}|${update}|${rounds}|${start}`
  const [pos, setPos] = useState({ key, step: 0 })
  const step = pos.key === key ? Math.min(pos.step, states.length - 1) : 0
  const current = states[step]

  const learnt = useMemo(() => reconstruct(current.D, s), [current, s])
  const dct = useMemo(() => reconstruct(DCT, s), [s])
  const learntMosaic = useMemo(() => mosaic(current.D, order), [current, order])
  const dctOrder = useMemo(() => DCT[0].map((_, j) => j), [])
  const dctMosaic = useMemo(() => mosaic(DCT, dctOrder), [dctOrder])

  const tile = Math.min(T - 1, Math.floor(cy / P)) * T + Math.min(T - 1, Math.floor(cx / P))
  const tx = (tile % T) * P
  const ty = Math.floor(tile / T) * P
  const tileBox = useMemo(() => box(tx, ty, tx + P, ty + P), [tx, ty])
  const learntUsed = useMemo(() => usedBoxes(learnt.X, tile, order), [learnt, tile, order])
  const dctUsed = useMemo(() => usedBoxes(dct.X, tile, dctOrder), [dct, tile, dctOrder])
  const curve = useMemo(() => ({ x: states.map((_, t) => t), y: states.map((st) => st.objective) }), [states])

  const ix = useAxis({ label: 'column', range: [0, W], zoom: false })
  const iy = useAxis({ label: 'row', range: [0, W], inverse: true, equal: ix, zoom: false })
  const lm = useMosaicAxes(k)
  const dm = useMosaicAxes(DCT[0].length)
  const ox = useAxis({ label: 'round', integer: true, range: [0, undefined] })
  const oy = useAxis({ label: '½‖Y − DX‖²', log: true })

  const [cell, cellSize] = useElementSize<HTMLDivElement>()

  const image = (rows: number[][], name: string, title: string, handle: boolean) => (
    <Plot x={ix} y={iy} title={title} fitHeight>
      <Raster x={PIXELS} y={PIXELS} z={rows} scale="sequential" range={[0, 1]} colorBar={false} valueLabel={name} />
      <Segments name="chosen tile" segments={tileBox} emphasis width={2} live />
      {handle && <Handle {...state.handle(['cx', 'cy'], { label: 'tile' })} />}
    </Plot>
  )

  return (
    <Figure
      title="A dictionary learnt from image patches"
      state={state}
      defaultSize="XL"
      caption={`Top: a 60 × 60 test image cut into 100 tiles of 6 × 6 pixels; each tile, less its mean, is coded with s atoms by orthogonal matching pursuit and rebuilt, over the learnt dictionary and over the 35 non-constant atoms of the 2-D discrete cosine transform. Bottom: the atoms and the training objective by round. Each atom is scaled so that its entry of largest magnitude shows as +1 (red positive, blue negative), the learnt ones in order of use, most used first; boxes mark the atoms the chosen tile uses. The dictionary is trained on ${TRAIN} patches at random positions (flat patches skipped), starting from ${k} of them; play the rounds to watch noisy patches turn into edges at the angles of the image's shapes, stripes at the stripes' angle and period, and corners. Drag the box on the left image to choose a tile. With the same s, the learnt dictionary rebuilds the image better than the cosine basis even when it has fewer atoms, because its atoms are cut to this image; the cosines are not, so a sharp edge spreads over many of them.`}
      controls={
        <Player
          value={step}
          onChange={(t) => setPos({ key, step: t })}
          count={states.length}
          label="round"
          format={(t) => (t === 0 ? 'initial: random patches' : `round ${t} of ${states.length - 1}`)}
        />
      }
      readouts={
        <>
          <Readout label="training objective" value={formatNumber(current.objective)} />
          <Readout label="RMS error, learnt" value={formatNumber(learnt.error)} />
          <Readout label="RMS error, cosines" value={formatNumber(dct.error)} />
          <Readout label="atoms used by the tile" value={String(learntUsed.length / 4)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
        <div ref={cell}>{image(imageRows(IMAGE), 'image', 'image', true)}</div>
        {image(learnt.rows, 'learnt', `learnt, s = ${s}`, false)}
        {image(dct.rows, 'cosines', `cosines, s = ${s}`, false)}
        <Plot x={lm.x} y={lm.y} title={`learnt atoms (${k})`} fitHeight>
          <Raster
            x={lm.cx}
            y={lm.cy}
            z={learntMosaic.z}
            scale="diverging"
            range={[-1, 1]}
            colorBar={false}
            valueLabel="learnt atom"
            stale={run.stale}
          />
          <Segments name="used by the tile" segments={learntUsed} emphasis width={2} />
        </Plot>
        <Plot x={dm.x} y={dm.y} title="cosine atoms (35)" fitHeight>
          <Raster
            x={dm.cx}
            y={dm.cy}
            z={dctMosaic.z}
            scale="diverging"
            range={[-1, 1]}
            colorBar={false}
            valueLabel="cosine atom"
          />
          <Segments name="used by the tile" segments={dctUsed} emphasis width={2} />
        </Plot>
        <FixedHeight height={Math.max(200, Math.round(cellSize.width))}>
          <Plot x={ox} y={oy} title="training objective" legend={false}>
            <Curve name="objective" x={curve.x} y={curve.y} slot={0} showPoints />
            <Points name="this round" x={[step]} y={[curve.y[step]]} emphasis size={10} live />
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
