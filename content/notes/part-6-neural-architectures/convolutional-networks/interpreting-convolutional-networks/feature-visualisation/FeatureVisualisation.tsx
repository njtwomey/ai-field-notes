import { useMemo, useState } from 'react'
import {
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Player,
  Plot,
  Readout,
  row,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { stream } from 'aifn-compute/foundation/random'
import { get, reshape, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { activationMaximisation } from 'aifn-compute/learning/explain'
import { boxOutline, FixedHeight, ImageTile, useTrainedCnn, type TileOutline } from '../_shared/figure-helpers'
import {
  C1,
  C2,
  conv2Field,
  logits,
  SHAPE_CLASSES,
  shapesSetup,
  SIZE,
  smallCnn,
  smallCnnTrunk,
  trainingSet,
  imageAt,
  type CnnParams,
} from '../_shared/small-cnn'

const UNITS = [
  ...SHAPE_CLASSES.map((s, c) => ({ value: `class-${c}`, label: `class score: ${s}` })),
  ...Array.from({ length: C2 }, (_, k) => ({ value: `channel-${k}`, label: `conv2 channel ${k}, centre unit` })),
]
const TOP = 6
const CENTRE = 4

const net = smallCnn(SHAPE_CLASSES.length)
const trunk = smallCnnTrunk(4)

/** The unit's value as a differentiable function of one 16 x 16 image. */
function unitFunction(params: CnnParams, unit: string) {
  const [kind, index] = unit.split('-')
  const k = Number(index)
  if (kind === 'class') return (x: Tensor) => get(net.apply(params, reshape(x, [1, 1, SIZE, SIZE])), 0, k)
  const head = params.slice(0, 4)
  return (x: Tensor) => get(trunk.apply(head, reshape(x, [1, 1, SIZE, SIZE])), 0, k, CENTRE, CENTRE)
}

/**
 * Activation maximisation on a small trained network: gradient ascent on the input, with an L2 penalty, jitter and
 * blur, scrubbed by a Player; beside it the first-layer filters and the training images that excite the unit most.
 */
export function FeatureVisualisation() {
  const state = useFigureState({
    training: row('1 · train the network', {
      steps: int(120, { ge: 10, le: 600, suggestions: [60, 120, 240], label: 'Adam steps' }),
      rate: float(0.02, { gt: 0, le: 0.5, suggestions: [0.005, 0.01, 0.02, 0.05], label: 'learning rate' }),
    }),
    unit: choice(UNITS, 'class-2', { label: '2 · unit to maximise' }),
    ascent: row('3 · gradient ascent on the input', {
      iterations: int(150, { ge: 10, le: 500, suggestions: [50, 150, 300], label: 'steps' }),
      stepSize: float(0.05, { gt: 0, le: 1, suggestions: [0.01, 0.05, 0.1], label: 'step size η' }),
      l2: float(0.05, { ge: 0, le: 5, suggestions: [0, 0.05, 0.2, 1], label: 'L2 weight λ' }),
      jitter: int(1, { ge: 0, le: 4, suggestions: [0, 1, 2], label: 'jitter (pixels)' }),
      blur: float(0.5, { ge: 0, le: 2, suggestions: [0, 0.5, 1], label: 'blur σ (pixels)' }),
    }),
  })
  const { steps, rate } = state.training
  const { iterations, stepSize, l2, jitter, blur } = state.ascent
  const setup = useMemo(() => shapesSetup(steps, rate), [steps, rate])
  const progress = useTrainedCnn(setup)
  const params = progress.params

  const am = useComputed(
    () => {
      if (!params) return null
      const alg = activationMaximisation(unitFunction(params, state.unit), [SIZE, SIZE], {
        stepSize,
        l2,
        jitter,
        blur,
        blurEvery: 4,
        clip: [0, 1],
        init: [0.1, 0.05],
      })
      const tr = trace(alg, undefined, iterations, {
        record: { value: (s) => s.value },
        keep: 'all',
        stream: stream(3),
      })
      return {
        images: tr.steps.map((s) => s.x),
        values: Array.from(toFlat(tr.series.value)),
        index: Array.from(tr.index),
      }
    },
    [params, state.unit, iterations, stepSize, l2, jitter, blur],
    { mode: 'release' },
  )
  const run = am.value
  const runKey = `${state.unit}|${iterations}|${stepSize}|${l2}|${jitter}|${blur}|${progress.done}`
  const [picked, setPicked] = useState({ key: runKey, at: 0 })
  const count = run ? run.images.length : 1
  const at = picked.key === runKey ? Math.min(picked.at, count - 1) : 0
  const setAt = (v: number) => setPicked({ key: runKey, at: Math.max(0, Math.min(count - 1, Math.round(v))) })

  // The training images, and each one's response for the chosen unit: the class score, or the channel's largest
  // value over the 8 x 8 map with the place it occurs.
  const data = useMemo(() => trainingSet(setup), [setup])
  const top = useMemo(() => {
    if (!params) return []
    const [kind, index] = state.unit.split('-')
    const k = Number(index)
    const n = data.shape.length
    let scored: { i: number; v: number; r: number; c: number }[]
    if (kind === 'class') {
      scored = logits(net, params, data.x).map((z, i) => ({ i, v: z[k], r: -1, c: -1 }))
    } else {
      const maps = toFlat(trunk.apply(params.slice(0, 4), data.x) as Tensor)
      const hw = (SIZE / 2) ** 2
      scored = Array.from({ length: n }, (_, i) => {
        let best = -Infinity
        let at = 0
        for (let p = 0; p < hw; p++) {
          const v = maps[(i * C2 + k) * hw + p]
          if (v > best) [best, at] = [v, p]
        }
        return { i, v: best, r: Math.floor(at / (SIZE / 2)), c: at % (SIZE / 2) }
      })
    }
    return scored.sort((a, b) => b.v - a.v).slice(0, TOP)
  }, [params, state.unit, data])

  const filters = useMemo(() => {
    if (!params) return null
    const w = toFlat((params[0] as { weight: Tensor }).weight)
    const m = Math.max(...Array.from(w, Math.abs))
    return { w, m }
  }, [params])

  const stepAxis = useAxis({ label: 'gradient-ascent step', integer: true })
  const valueAxis = useAxis({ label: 'unit value f(x)' })
  const image = run ? run.images[at] : new Float64Array(SIZE * SIZE)
  const isChannel = state.unit.startsWith('channel')
  const [f0, f1] = conv2Field(CENTRE)
  const centreField: TileOutline[] = isChannel
    ? [{ name: 'receptive field', ...boxOutline(f0, f1, f0, f1, SIZE), emphasis: true }]
    : []
  return (
    <Figure
      title="Activation maximisation on a small trained network"
      state={state}
      caption={
        'A network of two 3 × 3 convolutions, global average pooling and a linear layer is trained in the page on 16 × 16 images of discs, squares and crosses. Top left: the image found by gradient ascent on the input for the chosen unit, at the step the player shows; for a conv2 unit the box marks the 8 × 8 pixels it can see. Top right: the unit value at every step; drag the vertical line to scrub. Bottom: the six first-layer filters (diverging colour, pale at zero) and the six training images that excite the unit most (for a channel, the box marks where in the image its largest response sits). Set λ, jitter and blur to 0 to see unregularised ascent; a class score is an average over all positions, so its image repeats a motif across the frame.'
      }
      readouts={
        <>
          <Readout
            label="training"
            value={progress.done ? `${steps} steps done` : `step ${progress.losses.length} of ${steps}`}
          />
          <Readout
            label="final training loss"
            value={progress.losses.length ? formatNumber(progress.losses[progress.losses.length - 1]) : '—'}
          />
          <Readout label="unit value at this step" value={run ? formatNumber(run.values[at]) : '—'} />
          <Readout label="best training image" value={top.length ? formatNumber(top[0].v) : '—'} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-[1fr_2fr]">
        <ImageTile
          values={image}
          width={SIZE}
          height={SIZE}
          range={[0, 1]}
          title={`step ${run ? run.index[at] : 0}`}
          outlines={centreField}
          ariaLabel="Image found by activation maximisation"
        />
        <FixedHeight height={220}>
          <Plot x={stepAxis} y={valueAxis} ariaLabel="Unit value against gradient-ascent step">
            {run && <Curve name="f(x)" x={run.index} y={run.values} stale={am.stale} />}
            {run && <Handle kind="x" at={run.index[at]} onDrag={(v) => setAt(v)} label="step" />}
          </Plot>
        </FixedHeight>
      </div>
      <Player
        value={at}
        onChange={setAt}
        count={count}
        label="ascent step"
        format={(p) => `${run ? run.index[p] : 0}`}
      />
      <div className="mt-3 grid grid-cols-1 gap-4 md:grid-cols-2">
        <div>
          <div className="mb-1 text-xs text-muted-foreground">first-layer filters (3 × 3)</div>
          <div className="grid grid-cols-6 gap-2">
            {Array.from({ length: C1 }, (_, k) => (
              <ImageTile
                key={k}
                values={filters ? filters.w.slice(9 * k, 9 * k + 9) : new Float64Array(9)}
                width={3}
                height={3}
                scale="diverging"
                range={filters ? [-filters.m, filters.m] : [-1, 1]}
                title={`${k}`}
                ariaLabel={`First-layer filter ${k}`}
              />
            ))}
          </div>
        </div>
        <div>
          <div className="mb-1 text-xs text-muted-foreground">training images that excite the unit most</div>
          <div className="grid grid-cols-6 gap-2">
            {Array.from({ length: TOP }, (_, j) => {
              const t = top[j]
              const outline: TileOutline[] =
                t && t.r >= 0
                  ? [
                      {
                        name: 'receptive field',
                        ...boxOutline(...conv2Field(t.r), ...conv2Field(t.c), SIZE),
                        emphasis: true,
                      },
                    ]
                  : []
              return (
                <ImageTile
                  key={j}
                  values={t ? imageAt(data.x, t.i) : new Float64Array(SIZE * SIZE)}
                  width={SIZE}
                  height={SIZE}
                  range={[0, 1]}
                  title={t ? formatNumber(t.v) : ''}
                  outlines={outline}
                  ariaLabel={`Top-activating training image ${j + 1}`}
                />
              )
            })}
          </div>
        </div>
      </div>
    </Figure>
  )
}
