/**
 * Showcase: contrastive alignment of two views (a tiny CLIP). Coloured shapes are seen twice, as a small image and as
 * a noisy caption of their attributes (`aifn-methods/data` `pairedShapes`); two MLP encoders map the views onto the
 * unit circle (or sphere) and are trained in the worker by CLIP's symmetric InfoNCE with a learnable temperature
 * (`aifn-methods/neural/contrastive`). Everything drawn is computed by aifn: the embeddings at a checkpoint, the batch
 * similarities, zero-shot classification by the class captions, retrieval, and Wang & Isola's alignment and uniformity.
 */
import { useMemo, useState, type ReactNode } from 'react'
import {
  PAIRED_COLOURS,
  PAIRED_SHAPES,
  PAIRED_SIZES,
  pairedShapes,
  type PairedViews,
} from 'aifn-methods/data/synthetic'
import {
  embed,
  retrieve,
  similarities,
  TwoTower,
  zeroShot,
  type ContrastiveAblationRun,
  type ContrastiveSnapshot,
  type TemperatureSetting,
} from 'aifn-methods/neural/contrastive'
import { pca } from 'aifn-methods/unsupervised'
import { stream } from 'aifn/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn/foundation/tensor'
import { Player, StatusText } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { cn } from '@lab/lib/utils'
import { call, choice, int, row, slider, toggle, useFigureState, useStreamed, type Task } from '@lab/state'
import { Button } from '@lab/ui/button'
import { Curve, Handle, Plot, Plots, Points, Raster, Readout, Segments, useAxis } from '@lab/viz'
import { formatValue } from '@lab/views'

const f3 = (v: number) => (Number.isFinite(v) ? formatValue(Number(v.toPrecision(3))) : '—')
const pct = (v: number) => (Number.isFinite(v) ? `${Math.round(100 * v)}%` : '—')

/** Test pairs scored at each checkpoint and drawn on the circle. */
const TEST_N = 300
/** Pairs drawn on the circle. */
const SHOWN = 160
/** The batch shown as a similarity matrix and in the retrieval gallery. */
const BATCH = 12
const TOP_K = 4
const ATTRIBUTES = { shape: PAIRED_SHAPES, colour: PAIRED_COLOURS, size: PAIRED_SIZES } as const
type AttributeName = keyof typeof ATTRIBUTES
const TEMPERATURES = ['learned', '0.02', '0.05', '0.1', '0.3', '1'] as const

const temperatureSetting = (t: string): TemperatureSetting => (t === 'learned' ? 'learned' : Number(t))

/** The worker task building paired shapes from a named stream. */
const dataTask = (seed: string, knobs: Record<string, unknown>) =>
  call<PairedViews>('data/synthetic/pairedShapes', call('foundation/random/stream', seed), knobs)

// ── Drawing one image of the A view ──────────────────────────────────────────────────────────────────────────────────

/** A shape image (one row of view A, channel-major) as an SVG of coloured cells: the data's own RGB values. */
function ShapeImage({ row: x, size, label }: { row: ArrayLike<number>; size: number; label: string }) {
  const P = size * size
  const cells: ReactNode[] = []
  const c = (v: number) => Math.round(255 * Math.max(0, Math.min(1, v)))
  for (let r = 0; r < size; r++)
    for (let k = 0; k < size; k++) {
      const i = r * size + k
      cells.push(
        <rect
          key={i}
          x={k}
          y={r}
          width={1.02}
          height={1.02}
          fill={`rgb(${c(x[i])},${c(x[P + i])},${c(x[2 * P + i])})`}
        />,
      )
    }
  return (
    <svg
      viewBox={`0 0 ${size} ${size}`}
      className="size-full"
      role="img"
      aria-label={label}
      shapeRendering="crispEdges"
    >
      {cells}
    </svg>
  )
}

/** One row of a [n, d] tensor's values. */
const rowOf = (v: ArrayLike<number>, d: number, i: number) =>
  Array.prototype.slice.call(v, i * d, (i + 1) * d) as number[]

// ── The training figure ──────────────────────────────────────────────────────────────────────────────────────────────

const SCHEMA = {
  data: row('1 · data: shapes seen as an image and as a caption', {
    n: int(1500, { label: 'training pairs', ge: 100, le: 5000, suggestions: [500, 1500, 3000] }),
    heldOut: int(2, { label: 'held-out shape–colour pairs', ge: 0, le: 6 }),
    captionNoise: slider(0, 0.6, 0.15, { label: 'caption noise sd', step: 0.05 }),
    jitter: slider(0, 1, 0.3, { label: 'position and rotation jitter', step: 0.05 }),
  }),
  model: row('2 · encoders and loss', {
    dim: choice([2, 3, 8], 2, { label: 'embedding dimension d' }),
    batchSize: int(32, { ge: 2, le: 512, suggestions: [4, 8, 16, 32, 64, 128], label: 'batch size (B − 1 negatives)' }),
    temperature: choice(TEMPERATURES, 'learned', { label: 'temperature τ' }),
    steps: int(1000, { label: 'Adam steps', ge: 10, le: 5000, suggestions: [300, 1000, 2000] }),
    seed: int(1, { label: 'seed', ge: 0, le: 9999 }),
  }),
  view: row('3 · view', {
    colourBy: choice(['shape', 'colour', 'size'] as const, 'shape', { label: 'colour points by' }),
    pairs: toggle(true, 'lines joining pairs'),
    prototypes: toggle(false, 'class captions'),
  }),
}

type Setup = {
  knobs: { n: number; heldOut: number; captionNoise: number; jitter: number }
  options: { dim: number; batchSize: number; temperature: TemperatureSetting; steps: number; seed: number }
}

const keyOf = (s: Setup) => JSON.stringify(s)

export function ClipShowcase() {
  const state = useFigureState(SCHEMA)
  const current: Setup = {
    knobs: {
      n: state.data.n,
      heldOut: state.data.heldOut,
      captionNoise: state.data.captionNoise,
      jitter: state.data.jitter,
    },
    options: {
      dim: state.model.dim,
      batchSize: state.model.batchSize,
      temperature: temperatureSetting(state.model.temperature),
      steps: state.model.steps,
      seed: state.model.seed,
    },
  }
  // Nothing trains until Train is pressed; the run shown keeps the setup it was trained with.
  const [trained, setTrained] = useState<Setup | null>(null)
  const stale = trained !== null && keyOf(trained) !== keyOf(current)
  const shown = trained ?? current
  const task = useMemo((): Task<ContrastiveSnapshot> | null => {
    if (!trained) return null
    const { knobs, options } = trained
    return call<ContrastiveSnapshot>(
      'neural/contrastive/contrastiveTrainingRun',
      dataTask(`clip-train-${options.seed}`, knobs),
      dataTask(`clip-test-${options.seed}`, { ...knobs, n: TEST_N, include: 'all' }),
      {
        ...options,
        initialTemperature: options.temperature === 'learned' ? 0.1 : options.temperature,
        every: Math.max(1, Math.round(options.steps / 50)),
        seed: options.seed,
        evaluate: TEST_N,
      },
    )
  }, [trained])
  const run = useStreamed(task)
  const snap = run.value

  // The test pairs, rebuilt here from the same stream as the worker's.
  const test = useMemo(
    () => pairedShapes(stream(`clip-test-${shown.options.seed}`), { ...shown.knobs, n: TEST_N, include: 'all' }),
    [shown.options.seed, shown.knobs.n, shown.knobs.heldOut, shown.knobs.captionNoise, shown.knobs.jitter], // eslint-disable-line react-hooks/exhaustive-deps
  )
  const [pickedStep, setPicked] = useState<{ task: typeof task; index: number } | null>(null)
  const count = snap?.checkpoints.length ?? 0
  // The player opens at step 0 and stays where the reader puts it; a new run starts at 0 again.
  const index = Math.min(pickedStep?.task === task ? pickedStep.index : 0, Math.max(0, count - 1))
  const setIndex = (i: number) => setPicked({ task, index: i })
  const cp = snap?.checkpoints[index] ?? null

  const model = useMemo(
    () =>
      snap
        ? TwoTower({ inA: test.a.shape[1], inB: test.b.shape[1], hidden: snap.config.hidden, dim: snap.config.dim })
        : null,
    [snap?.config.dim, snap?.config.hidden, test], // eslint-disable-line react-hooks/exhaustive-deps
  )
  const d = snap?.config.dim ?? shown.options.dim

  // Embeddings at the checkpoint: the test pairs, and the class captions (prototypes).
  const emb = useMemo(() => {
    if (!model || !cp) return null
    return {
      za: embed(model, cp.params, test.a, 'a'),
      zb: embed(model, cp.params, test.b, 'b'),
      zp: embed(model, cp.params, test.prototypes, 'b'),
    }
  }, [model, cp, test])

  // A 2-D view: the embedding itself for d = 2, else the top two principal axes of the last checkpoint's embeddings.
  const lastParams = snap?.checkpoints.at(-1)?.params
  const projector = useMemo(() => {
    if (!model || !lastParams || d === 2) return null
    const za = embed(model, lastParams, test.a, 'a')
    const zb = embed(model, lastParams, test.b, 'b')
    const both = fromData(Float64Array.from([...toFlat(za), ...toFlat(zb)]), [2 * TEST_N, d])
    return pca({ components: 2 }).fit({ kind: 'dataset', x: both })
  }, [model, lastParams, test, d])
  const view = (z: Tensor) => {
    const v = toFlat(projector ? projector.transform(z) : z)
    const n = z.shape[0]
    return { x: Array.from({ length: n }, (_, i) => v[2 * i]), y: Array.from({ length: n }, (_, i) => v[2 * i + 1]) }
  }

  const attr = state.view.colourBy as AttributeName
  const groups = useMemo(() => Array.from(toFlat(test.truth.attributes[attr])), [test, attr])
  const shownIdx = useMemo(() => Array.from({ length: SHOWN }, (_, i) => i), [])
  const picture = useMemo(() => {
    if (!emb) return null
    const pa = view(emb.za)
    const pb = view(emb.zb)
    const pp = view(emb.zp)
    return {
      a: { x: shownIdx.map((i) => pa.x[i]), y: shownIdx.map((i) => pa.y[i]) },
      b: { x: shownIdx.map((i) => pb.x[i]), y: shownIdx.map((i) => pb.y[i]) },
      p: pp,
      segments: shownIdx.map((i) => ({ from: [pa.x[i], pa.y[i]] as const, to: [pb.x[i], pb.y[i]] as const })),
    }
  }, [emb, projector]) // eslint-disable-line react-hooks/exhaustive-deps
  const protoGroups = useMemo(() => {
    const j = attr === 'shape' ? 0 : attr === 'size' ? 1 : 2
    return test.meta.combinationCodes.map((c) => c[j])
  }, [test, attr])

  // The batch of the similarity matrix and the gallery: the first BATCH test pairs.
  const batch = useMemo(() => {
    if (!emb) return null
    const ids = Array.from({ length: BATCH }, (_, i) => i)
    const za = fromData(Float64Array.from(ids.flatMap((i) => rowOf(toFlat(emb.za), d, i))), [BATCH, d])
    const zb = fromData(Float64Array.from(ids.flatMap((i) => rowOf(toFlat(emb.zb), d, i))), [BATCH, d])
    const s = similarities(za, zb)
    return { za, zb, z: ids.map((i) => ids.map((j) => s[i * BATCH + j])) }
  }, [emb, d])
  const cells = useMemo(() => Array.from({ length: BATCH }, (_, i) => i + 1), [])

  // Retrieval: an image or a caption of the batch, picked in the gallery, on the circle or on the matrix.
  const [pick, setPick] = useState<{ view: 'a' | 'b'; i: number }>({ view: 'a', i: 0 })
  const matches = useMemo(() => {
    if (!batch) return null
    const [from, to] = pick.view === 'a' ? [batch.za, batch.zb] : [batch.zb, batch.za]
    return retrieve(rowOf(toFlat(from), d, pick.i), to, TOP_K)
  }, [batch, pick, d])
  const zeroShotPick = useMemo(() => {
    if (!emb || pick.view !== 'a') return null
    const row1 = fromData(Float64Array.from(rowOf(toFlat(emb.za), d, pick.i)), [1, d])
    return zeroShot(row1, emb.zp)[0]
  }, [emb, pick, d])

  // Axes: the circle with equal units; the curves against step.
  const cx = useAxis({ label: d === 2 ? 'z₁' : 'principal axis 1', range: [-1.25, 1.25], nice: false })
  const cy = useAxis({ label: d === 2 ? 'z₂' : 'principal axis 2', range: [-1.25, 1.25], nice: false, equal: cx })
  const mi = useAxis({ label: 'caption j', range: [0.5, BATCH + 0.5], nice: false, integer: true })
  // Image 1 at the top: row i of the matrix sits at y = BATCH + 1 − i.
  const mj = useAxis({
    label: 'image i',
    range: [0.5, BATCH + 0.5],
    nice: false,
    integer: true,
    format: (v) => String(BATCH + 1 - v),
  })
  const stepAxis = useAxis({ label: 'step', range: [0, shown.options.steps], key: shown.options.steps, integer: true })
  const lossAxis = useAxis({ label: 'minibatch loss', hold: 'union', key: task })
  const tauAxis = useAxis({ label: 'τ', hold: 'union', key: task, log: true })
  const accAxis = useAxis({ label: 'accuracy', range: [0, 1] })
  const wiAxis = useAxis({ label: 'alignment, uniformity', hold: 'union', key: task })

  const curves = useMemo(() => {
    if (!snap) return null
    const steps = snap.losses.map((_, t) => t)
    const cs = snap.checkpoints
    const at = cs.map((c) => c.step)
    return {
      steps,
      losses: [...snap.losses],
      temperatures: [...snap.temperatures],
      at,
      seen: cs.map((c) => c.zeroShotSeen),
      held: cs.map((c) => c.zeroShotHeldOut),
      heldOnly: cs.map((c) => c.zeroShotHeldOutOnly),
      top1: cs.map((c) => c.retrievalTop1),
      alignment: cs.map((c) => c.alignment),
      uniformity: cs.map((c) => c.uniformity),
    }
  }, [snap])
  const pickStep = (v: number) => {
    if (!snap || run.running) return
    const cs = snap.checkpoints
    let best = 0
    for (let i = 1; i < cs.length; i++) if (Math.abs(cs[i].step - v) < Math.abs(cs[best].step - v)) best = i
    setIndex(best)
  }
  const marker = cp && <Handle kind="x" at={cp.step} label="checkpoint" onDrag={pickStep} />
  const circle = useMemo(() => {
    const t = Array.from({ length: 121 }, (_, k) => (2 * Math.PI * k) / 120)
    return { x: t.map(Math.cos), y: t.map(Math.sin) }
  }, [])
  const diagonal = cells
  const antiDiagonal = useMemo(() => cells.map((c) => BATCH + 1 - c), [cells])
  const levelNames = ATTRIBUTES[attr]
  const heldNames = test.meta.heldOutCombinations.map((k) => test.meta.combinationNames[k])
  const progress = snap ? snap.step / snap.steps : 0
  const caption = (i: number) => test.meta.combinationNames[toFlat(test.truth.combination)[i]]
  const imageRow = (i: number) => rowOf(toFlat(test.a), test.a.shape[1], i)
  const imageSize = test.meta.views[0].image![1]

  return (
    <Figure
      title="Train a tiny CLIP, then watch the two views align"
      purpose="Two encoders, one for images and one for captions, are trained so that each object's two embeddings meet on the unit circle while the batch's other pairs are pushed apart; then a caption alone classifies images it was never trained to label."
      state={state}
      defaultSize="XL"
      controls={
        <ControlRow label="4 · train">
          <div className="flex flex-wrap items-center gap-3">
            {run.running ? (
              <Button size="sm" variant="destructive" aria-label="Stop" onClick={run.stop}>
                Stop
              </Button>
            ) : (
              <Button
                size="sm"
                variant={!trained || stale ? 'default' : 'outline'}
                aria-label="Train"
                onClick={() => setTrained(JSON.parse(JSON.stringify(current)) as Setup)}
              >
                {trained ? 'Retrain' : 'Train'}
              </Button>
            )}
            <div className="h-1.5 w-40 overflow-hidden rounded bg-muted" aria-busy={run.running}>
              <div className="h-full bg-primary" style={{ width: `${100 * Math.min(1, progress)}%` }} />
            </div>
            <StatusText tone={trained && run.error ? 'error' : !trained || stale ? 'attention' : 'muted'}>
              {!trained
                ? 'Not trained yet: choose the settings, then press Train.'
                : run.error
                  ? `failed: ${run.error}`
                  : stale
                    ? 'Settings changed since this run: press Retrain to train with them.'
                    : run.stopped
                      ? `stopped at step ${snap?.step ?? 0}`
                      : `step ${snap?.step ?? 0} / ${shown.options.steps}${run.running ? '…' : ''}`}
            </StatusText>
          </div>
          <Player
            label="checkpoint"
            value={index}
            onChange={setIndex}
            count={Math.max(1, count)}
            format={(p) => `step ${snap?.checkpoints[p]?.step ?? 0}`}
          />
        </ControlRow>
      }
      readouts={{
        [`at step ${cp?.step ?? 0}`]: (
          <>
            <Readout label="temperature τ" value={cp ? f3(cp.temperature) : '—'} />
            <Readout label="test InfoNCE (one batch of 300)" value={cp ? f3(cp.loss) : '—'} />
            <Readout label="alignment (lower: pairs closer)" value={cp ? f3(cp.alignment) : '—'} />
            <Readout label="uniformity (lower: more spread)" value={cp ? f3(cp.uniformity) : '—'} />
          </>
        ),
        'zero-shot (24 class captions)': (
          <>
            <Readout label="seen classes" value={cp ? pct(cp.zeroShotSeen) : '—'} />
            <Readout label="held-out classes, all 24 candidates" value={cp ? pct(cp.zeroShotHeldOut) : '—'} />
            <Readout label="held-out, among held-out classes" value={cp ? pct(cp.zeroShotHeldOutOnly) : '—'} />
            <Readout label="retrieval top-1 (same class)" value={cp ? pct(cp.retrievalTop1) : '—'} />
          </>
        ),
      }}
      caption={
        <>
          Data: aifn <code>pairedShapes</code> (seeded): {shown.knobs.n} training pairs of a {imageSize} × {imageSize}{' '}
          RGB image and a 9-entry caption (shape, size and colour one-hots plus noise);{' '}
          {heldNames.length ? `${heldNames.join(', ')} never appear in training` : 'no combination is held out'}. Model:
          two MLPs (input → 32 → d, ReLU), outputs normalised onto the unit {d === 2 ? 'circle' : 'sphere'}
          {d === 2 ? '' : ', drawn by its two principal axes at the last checkpoint'}; loss: aifn <code>infoNce</code>{' '}
          (symmetric, in-batch negatives) with τ{' '}
          {shown.options.temperature === 'learned'
            ? 'learned as log(1/τ) (aifn learnedTemperature), from 0.1'
            : `fixed at ${shown.options.temperature}`}
          ; Adam, step size 0.01, in the worker. Circles are images, squares captions; a line joins each pair. Drag the
          checkpoint marker on any curve, or use the player. Click a point on the circle, a row of the matrix, or an
          item of the gallery to retrieve its nearest matches. The matrix holds cosine similarities of a batch of{' '}
          {BATCH}: training raises the diagonal (each image with its own caption) and lowers the rest, but two objects
          of the same class have near-identical captions, so their off-diagonal cells stay bright.
        </>
      }
    >
      <Plots cols={2} widths={[1, 1]} scale={0.62}>
        <Plot
          x={cx}
          y={cy}
          title={d === 2 ? 'embeddings on the unit circle' : `d = ${d}: principal-axis view`}
          onPlotClick={([x, y]) => {
            if (!picture) return
            let best = 0
            let bd = Infinity
            for (let i = 0; i < BATCH; i++) {
              const dd = (picture.a.x[i] - x) ** 2 + (picture.a.y[i] - y) ** 2
              if (dd < bd) [best, bd] = [i, dd]
            }
            setPick({ view: 'a', i: best })
          }}
        >
          <Curve name="unit circle" x={circle.x} y={circle.y} muted silent />
          {picture && state.view.pairs && <Segments segments={picture.segments} />}
          {picture && (
            <Points
              name="images"
              x={picture.a.x}
              y={picture.a.y}
              group={groups.slice(0, SHOWN)}
              groupNames={levelNames}
              shape={0}
              thin
            />
          )}
          {picture && (
            <Points name="captions" x={picture.b.x} y={picture.b.y} group={groups.slice(0, SHOWN)} shape={1} thin />
          )}
          {picture && state.view.prototypes && (
            <Points name="class captions" x={picture.p.x} y={picture.p.y} group={protoGroups} shape={3} size={13} />
          )}
          {picture && (
            <Points
              name="picked"
              x={[pick.view === 'a' ? picture.a.x[pick.i] : picture.b.x[pick.i]]}
              y={[pick.view === 'a' ? picture.a.y[pick.i] : picture.b.y[pick.i]]}
              emphasis
              thin
            />
          )}
        </Plot>
        <Plot
          x={mi}
          y={mj}
          title={`similarities in a batch of ${BATCH} (diagonal: true pairs)`}
          legend={false}
          onPlotClick={([, y]) => setPick({ view: 'a', i: Math.max(0, Math.min(BATCH - 1, BATCH - Math.round(y))) })}
        >
          {batch && (
            <Raster
              x={cells}
              y={cells}
              z={[...batch.z].reverse()}
              scale="diverging"
              range={[-1, 1]}
              valueLabel="cos(zᵃᵢ, zᵇⱼ)"
            />
          )}
          <Points name="true pairs" x={diagonal} y={antiDiagonal} emphasis shape={1} size={7} />
        </Plot>
      </Plots>
      <Plots cols={2} rows={2} scale={0.55}>
        <Plot x={stepAxis} y={lossAxis} title="minibatch InfoNCE" legend={false}>
          {curves && <Curve name="loss" x={curves.steps} y={curves.losses} slot={0} thin />}
          {marker}
        </Plot>
        <Plot x={stepAxis} y={tauAxis} title="temperature τ" legend={false}>
          {curves && <Curve name="τ" x={curves.steps} y={curves.temperatures} slot={1} />}
          {marker}
        </Plot>
        <Plot x={stepAxis} y={accAxis} title="zero-shot and retrieval (test)">
          {curves && <Curve name="seen classes" x={curves.at} y={curves.seen} slot={0} />}
          {curves && <Curve name="held-out (of 24)" x={curves.at} y={curves.held} slot={1} />}
          {curves && <Curve name="held-out (of held-out)" x={curves.at} y={curves.heldOnly} slot={1} dashed />}
          {curves && <Curve name="retrieval top-1" x={curves.at} y={curves.top1} slot={2} />}
          {marker}
        </Plot>
        <Plot x={stepAxis} y={wiAxis} title="alignment and uniformity (test)">
          {curves && <Curve name="alignment" x={curves.at} y={curves.alignment} slot={3} />}
          {curves && <Curve name="uniformity" x={curves.at} y={curves.uniformity} slot={4} />}
          {marker}
        </Plot>
      </Plots>
      <Gallery
        count={BATCH}
        pick={pick}
        setPick={setPick}
        matches={matches}
        imageRow={imageRow}
        imageSize={imageSize}
        caption={caption}
        zeroShot={zeroShotPick === null ? null : test.meta.combinationNames[zeroShotPick]}
        heldOut={(i) => toFlat(test.truth.heldOut)[i] === 1}
      />
    </Figure>
  )
}

/** The batch as a gallery of images and captions; picking one lists its nearest matches in the other view. */
function Gallery({
  count,
  pick,
  setPick,
  matches,
  imageRow,
  imageSize,
  caption,
  zeroShot,
  heldOut,
}: {
  count: number
  pick: { view: 'a' | 'b'; i: number }
  setPick: (p: { view: 'a' | 'b'; i: number }) => void
  matches: { index: number[]; similarity: number[] } | null
  imageRow: (i: number) => ArrayLike<number>
  imageSize: number
  caption: (i: number) => string
  zeroShot: string | null
  heldOut: (i: number) => boolean
}) {
  const ids = Array.from({ length: count }, (_, i) => i)
  const item = (view: 'a' | 'b', i: number, small = false) =>
    view === 'a' ? (
      <span className={cn('block overflow-hidden rounded-sm border border-border', small ? 'size-8' : 'size-11')}>
        <ShapeImage row={imageRow(i)} size={imageSize} label={`image ${i + 1}: ${caption(i)}`} />
      </span>
    ) : (
      <span className="inline-flex items-center gap-1.5 text-xs">{caption(i)}</span>
    )
  return (
    <div className="flex flex-col gap-3 text-sm">
      <div className="text-xs font-medium text-muted-foreground">
        retrieval in the batch: pick an image or a caption (italic: a held-out class)
      </div>
      <div className="flex flex-wrap gap-1.5" role="group" aria-label="images">
        {ids.map((i) => (
          <button
            key={i}
            type="button"
            aria-pressed={pick.view === 'a' && pick.i === i}
            onClick={() => setPick({ view: 'a', i })}
            className={cn(
              'rounded-md p-0.5 ring-offset-background',
              pick.view === 'a' && pick.i === i ? 'ring-2 ring-primary' : 'hover:bg-muted',
            )}
            title={caption(i)}
          >
            {item('a', i)}
          </button>
        ))}
      </div>
      <div className="flex flex-wrap gap-1.5" role="group" aria-label="captions">
        {ids.map((i) => (
          <button
            key={i}
            type="button"
            aria-pressed={pick.view === 'b' && pick.i === i}
            onClick={() => setPick({ view: 'b', i })}
            className={cn(
              'rounded-md border px-2 py-1',
              heldOut(i) && 'italic',
              pick.view === 'b' && pick.i === i ? 'border-primary bg-primary/10' : 'border-border hover:bg-muted',
            )}
          >
            {item('b', i)}
          </button>
        ))}
      </div>
      <div className="flex flex-wrap items-center gap-3 rounded-md border border-border p-2">
        <span className="text-xs text-muted-foreground">
          {pick.view === 'a' ? 'image' : 'caption'} {pick.i + 1}
        </span>
        {item(pick.view, pick.i, true)}
        <span className="text-xs text-muted-foreground">→ nearest {pick.view === 'a' ? 'captions' : 'images'}:</span>
        {matches ? (
          matches.index.map((j, r) => (
            <span
              key={j}
              className={cn(
                'inline-flex items-center gap-1.5 rounded px-1 tabular-nums',
                j === pick.i && 'bg-success/15 text-success',
              )}
              title={j === pick.i ? 'its own pair' : undefined}
            >
              {item(pick.view === 'a' ? 'b' : 'a', j, true)}
              <span className="text-xs">
                {r + 1}. {f3(matches.similarity[r])}
                {j === pick.i ? ' ✓' : ''}
              </span>
            </span>
          ))
        ) : (
          <span className="text-xs text-muted-foreground">press Train</span>
        )}
        {zeroShot && (
          <span className="text-xs">
            zero-shot class: <b>{zeroShot}</b> {zeroShot === caption(pick.i) ? '✓' : `(true: ${caption(pick.i)})`}
          </span>
        )}
      </div>
    </div>
  )
}

// ── The ablation figure ──────────────────────────────────────────────────────────────────────────────────────────────

const ABLATION_BATCHES = [4, 16, 64] as const
const ABLATION_TEMPERATURES: readonly TemperatureSetting[] = [0.02, 0.1, 0.5, 'learned']
const temperatureName = (t: TemperatureSetting) => (t === 'learned' ? 'τ learned' : `τ = ${t}`)

const ABLATION_SCHEMA = {
  run: row('1 · runs', {
    steps: int(400, { label: 'Adam steps per run', ge: 20, le: 3000, suggestions: [200, 400, 1000] }),
    dim: choice([2, 8], 2, { label: 'embedding dimension d' }),
  }),
}

export function ClipAblation() {
  const state = useFigureState(ABLATION_SCHEMA)
  const current = { steps: state.run.steps, dim: state.run.dim }
  const [trained, setTrained] = useState<typeof current | null>(null)
  const stale = trained !== null && JSON.stringify(trained) !== JSON.stringify(current)
  const task = useMemo(
    () =>
      trained
        ? call<readonly ContrastiveAblationRun[]>(
            'neural/contrastive/contrastiveAblation',
            dataTask('clip-train-1', { n: 1500 }),
            dataTask('clip-test-1', { n: TEST_N, include: 'all' }),
            {
              steps: trained.steps,
              dim: trained.dim,
              batchSizes: ABLATION_BATCHES,
              temperatures: ABLATION_TEMPERATURES,
            },
          )
        : null,
    [trained],
  )
  const run = useStreamed(task)
  const runs = run.value ?? []
  const total = ABLATION_BATCHES.length * ABLATION_TEMPERATURES.length
  const tIndex = (t: TemperatureSetting) => ABLATION_TEMPERATURES.indexOf(t)
  const bIndex = (b: number) => (ABLATION_BATCHES as readonly number[]).indexOf(b)
  const ax = useAxis({ label: 'alignment (lower: pairs closer)', range: [0, undefined] })
  const ay = useAxis({ label: 'uniformity (lower: more spread)', range: [undefined, 0] })
  const bx = useAxis({ label: 'batch size', categories: ABLATION_BATCHES.map(String) })
  const by = useAxis({ label: 'zero-shot accuracy, seen classes', range: [0, 1] })
  const byTemperature = ABLATION_TEMPERATURES.map((t) => runs.filter((r) => r.temperature === t))
  const best = runs.length ? runs.reduce((a, b) => (b.zeroShotSeen > a.zeroShotSeen ? b : a)) : null
  return (
    <Figure
      title="Ablation: batch size and temperature trade alignment against uniformity"
      purpose="More negatives per batch spread the embeddings out (uniformity falls at every temperature) and lift zero-shot accuracy; a very low temperature keeps pairs tightest but spreads them least."
      state={state}
      defaultSize="L"
      controls={
        <ControlRow label="2 · run">
          <div className="flex flex-wrap items-center gap-3">
            {run.running ? (
              <Button size="sm" variant="destructive" aria-label="Stop" onClick={run.stop}>
                Stop
              </Button>
            ) : (
              <Button
                size="sm"
                variant={!trained || stale ? 'default' : 'outline'}
                aria-label="Train"
                onClick={() => setTrained({ ...current })}
              >
                {trained ? 'Rerun' : 'Train'}
              </Button>
            )}
            <StatusText tone={trained && run.error ? 'error' : 'muted'}>
              {!trained
                ? `Not run yet: ${total} runs of ${current.steps} steps each, one after another.`
                : run.error
                  ? `failed: ${run.error}`
                  : stale
                    ? 'Settings changed: press Rerun.'
                    : `${runs.length} / ${total} runs${run.running ? '…' : ''}`}
            </StatusText>
          </div>
        </ControlRow>
      }
      readouts={
        <>
          <Readout label="runs" value={`${runs.length} / ${total}`} />
          <Readout
            label="best zero-shot (seen)"
            value={
              best ? `${pct(best.zeroShotSeen)} at B = ${best.batchSize}, ${temperatureName(best.temperature)}` : '—'
            }
          />
        </>
      }
      caption={
        <>
          Each point is one tiny CLIP trained from the same initialisation on the same 1500 pairs (aifn{' '}
          <code>contrastiveAblation</code>, in the worker), scored on 300 test pairs by Wang & Isola's alignment E‖zᵃ −
          zᵇ‖² and uniformity log E exp(−2‖z − z′‖²) (aifn <code>alignment</code>, <code>uniformity</code>). Colour is
          the temperature, marker the batch size (circle 4, square 16, triangle 64). The best corner is the bottom left:
          pairs together, everything else spread out.
        </>
      }
    >
      <Plots cols={2} scale={0.6}>
        <Plot x={ax} y={ay} title="alignment against uniformity">
          {runs.length > 0 && (
            <Points
              name="runs"
              x={runs.map((r) => r.alignment)}
              y={runs.map((r) => r.uniformity)}
              group={runs.map((r) => tIndex(r.temperature))}
              groupNames={ABLATION_TEMPERATURES.map(temperatureName)}
              shape={runs.map((r) => bIndex(r.batchSize))}
            />
          )}
        </Plot>
        <Plot x={bx} y={by} title="zero-shot accuracy by batch size">
          {byTemperature.map(
            (rs, k) =>
              rs.length > 0 && (
                <Curve
                  key={k}
                  name={temperatureName(ABLATION_TEMPERATURES[k])}
                  x={rs.map((r) => bIndex(r.batchSize))}
                  y={rs.map((r) => r.zeroShotSeen)}
                  slot={k}
                  showPoints
                />
              ),
          )}
        </Plot>
      </Plots>
    </Figure>
  )
}
