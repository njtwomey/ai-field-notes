import { useMemo, useState } from 'react'
import {
  choice,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Pixels,
  Player,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { stream, uniform as drawUniform } from 'aifn/foundation/random'
import { mosaic } from '../_shared/spins'
import { barsAndStripes, hiddenProbs, sample, trainRbm, visibleProbs, type Method, type Rbm } from './rbm'

const SIDE = 4
const EPOCHS = 400
const CHAINS = 10
/** Images per row in every mosaic. */
const COLS = 5
const EXTENT = COLS * SIDE + COLS - 1
const DATA = barsAndStripes(SIDE)
const OPTIMUM = -Math.log(DATA.length)
/** Ten of the 30 patterns, spread over the set, for the reconstructions. */
const SHOWN = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28].map((i) => DATA[i % DATA.length])

const METHODS: { value: Method; label: string; slot: number }[] = [
  { value: 'cd1', label: 'CD-1', slot: 0 },
  { value: 'cd10', label: 'CD-10', slot: 1 },
  { value: 'pcd', label: 'PCD', slot: 2 },
]

/** Seeded uniform draws. */
function draws(seed: number) {
  const g = stream(seed)
  return () => drawUniform(g)
}

/** A mosaic of small images: row 0 at the top, square pixels, `extent` columns wide. */
function Mosaic({
  image,
  extent = image.width,
  scale,
  range,
  ariaLabel,
}: {
  image: { width: number; height: number; values: ArrayLike<number> }
  extent?: number
  scale?: 'sequential' | 'diverging'
  range: [number, number]
  ariaLabel: string
}) {
  const x = useAxis({ range: [-0.5, extent - 0.5], nice: false })
  const y = useAxis({ range: [-0.5, image.height - 0.5], nice: false, inverse: true, equal: x })
  return (
    <Plot x={x} y={y} bare height={140} ariaLabel={ariaLabel}>
      <Pixels width={image.width} height={image.height} values={image.values} scale={scale} range={range} />
    </Plot>
  )
}

const UNIT_RANGE: [number, number] = [0, 1]

/**
 * An RBM with 16 visible units learns the 30 bars-and-stripes patterns on a 4 × 4 grid, trained three ways at once.
 * The exact log-likelihood is computed by summing over the hidden states.
 */
export function RbmTrainer() {
  const state = useFigureState({
    H: choice<number>(
      [4, 8, 10].map((v) => ({ value: v, label: String(v) })),
      8,
      { label: 'hidden units' },
    ),
    rate: float(0.2, { gt: 0, max: 0.5, scale: 'log10', suggestions: [0.05, 0.1, 0.2, 0.5], label: 'learning rate' }),
    seed: int(1, { ge: 0, label: 'seed' }),
    method: choice<Method>(
      METHODS.map(({ value, label }) => ({ value, label })),
      'cd10',
      { label: 'shown method' },
    ),
    steps: int(20, { min: 0, max: 100, step: 1, label: 'Gibbs steps' }),
    temperature: float(1, { min: 0.2, max: 3, step: 0.05, label: 'sampling temperature T' }),
  })
  const H = state.H

  const runs = useMemo(
    () =>
      Object.fromEntries(
        METHODS.map(({ value }) => [
          value,
          trainRbm(DATA, H, {
            epochs: EPOCHS,
            rate: state.rate,
            method: value,
            batch: 5,
            uniform: draws(state.seed),
            every: 5,
          }),
        ]),
      ) as Record<Method, ReturnType<typeof trainRbm>>,
    [H, state.rate, state.seed],
  )
  // The walk through the epochs restarts at 0 for new training runs.
  const [position, setPosition] = useState({ runs, epoch: 0 })
  const epoch = position.runs === runs ? position.epoch : 0
  const go = (v: number) => setPosition({ runs, epoch: Math.max(0, Math.min(EPOCHS, Math.round(v))) })
  const snap = runs[state.method][epoch]
  const model: Rbm = useMemo(() => ({ V: SIDE * SIDE, H, W: snap.W, b: snap.b, c: snap.c }), [snap, H])

  // Filters: column j of W, the weights from hidden unit j to the 16 pixels, on a symmetric scale.
  const filters = useMemo(() => {
    const images = Array.from({ length: H }, (_, j) =>
      Array.from({ length: SIDE * SIDE }, (__, i) => snap.W[i * H + j]),
    )
    const bound = Math.max(0.5, ...Array.from(snap.W, Math.abs))
    return { ...mosaic(images, SIDE, SIDE, COLS, 1, 0), range: [-bound, bound] as [number, number] }
  }, [snap, H])

  // Gibbs chains from random visible states, `steps` block-Gibbs steps at temperature T, drawn from a fixed seed.
  const chains = useMemo(() => {
    const uniform = draws(state.seed + 77)
    const out: Float64Array[] = []
    for (let k = 0; k < CHAINS; k++) {
      let v = sample(new Float64Array(SIDE * SIDE).fill(0.5), uniform)
      for (let s = 0; s < state.steps; s++)
        v = sample(
          visibleProbs(model, sample(hiddenProbs(model, v, state.temperature), uniform), state.temperature),
          uniform,
        )
      out.push(v)
    }
    return mosaic(out, SIDE, SIDE, COLS, 1, 0.5)
  }, [model, state.steps, state.temperature, state.seed])

  // Reconstructions: data (top row) and p(v | h) with h drawn from p(h | v) (bottom row).
  const reconstructions = useMemo(() => {
    const uniform = draws(state.seed + 99)
    const recon = SHOWN.map((v) => visibleProbs(model, sample(hiddenProbs(model, v), uniform)))
    // Rows alternate: five data patterns, their reconstructions, the next five, theirs.
    const rows = [...SHOWN.slice(0, COLS), ...recon.slice(0, COLS), ...SHOWN.slice(COLS), ...recon.slice(COLS)]
    return mosaic(rows, SIDE, SIDE, COLS, 1, 0.5)
  }, [model, state.seed])

  const epochs = useMemo(() => runs.cd1.map((s) => s.epoch), [runs])
  const series = useMemo((): SeriesSpec[] => {
    const lines: SeriesSpec[] = METHODS.map(({ value, label, slot }) => ({
      name: label,
      type: 'line',
      x: epochs,
      y: runs[value].map((s) => s.logLik),
      slot,
    }))
    lines.push({ name: 'optimum −log 30', type: 'line', x: [0, EPOCHS], y: [OPTIMUM, OPTIMUM], dashed: true, slot: 3 })
    lines.push({ name: 'shown', type: 'scatter', x: [epoch], y: [runs[state.method][epoch].logLik], emphasis: true })
    return lines
  }, [epochs, runs, epoch, state.method])
  const xAxis = useAxis({ label: 'epoch', hold: 'union' })
  const yAxis = useAxis({ label: 'mean log p(v)', hold: 'union' })
  return (
    <Figure
      title="Training an RBM on bars and stripes"
      state={state}
      caption={
        <>
          The data are the 30 bars-and-stripes patterns on a 4 × 4 grid: any set of full rows, or any set of full
          columns. The same RBM is trained from the same start with CD-1, CD-10 and persistent CD (PCD), in mini-batches
          of 5; the chart shows the exact mean log-likelihood of each, against the best possible value −log 30. Pick a
          method and step through its epochs to see its filters (the weights of each hidden unit over the 16 pixels),
          ten Gibbs chains started from random pixels, and one-step reconstructions of data patterns. Raising the
          sampling temperature flattens every conditional and the chains lose the patterns; lowering it freezes them.
          Drag the epoch line on the chart.
        </>
      }
      controls={<Player value={epoch} onChange={go} count={EPOCHS + 1} label="epoch" format={(v) => `epoch ${v}`} />}
      readouts={
        <>
          <Readout label="epoch" value={epoch} />
          {METHODS.map(({ value, label }) => (
            <Readout key={value} label={`${label} log-likelihood`} value={formatNumber(runs[value][epoch].logLik)} />
          ))}
          <Readout label="optimum" value={formatNumber(OPTIMUM)} />
        </>
      }
    >
      <div className="grid gap-4 lg:grid-cols-[1fr_1fr]">
        <Plot
          x={xAxis}
          y={yAxis}
          height={300}
          ariaLabel={'Exact log-likelihood of the RBM during training by three methods'}
        >
          {seriesLayers(series)}
          <Handle kind="x" at={epoch} label="epoch" onDrag={go} />
        </Plot>
        <div className="grid grid-cols-3 content-start gap-3">
          <div className="flex flex-col gap-1">
            <span className="text-xs text-muted-foreground">filters: weights of each hidden unit (red +, blue −)</span>
            <Mosaic
              image={filters}
              extent={EXTENT}
              scale="diverging"
              range={filters.range}
              ariaLabel="Weights of each hidden unit drawn as 4 by 4 images"
            />
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-xs text-muted-foreground">
              Gibbs chains after {state.steps} steps at T = {formatNumber(state.temperature)}
            </span>
            <Mosaic image={chains} range={UNIT_RANGE} ariaLabel="Visible states of ten Gibbs chains" />
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-xs text-muted-foreground">
              data (rows 1, 3) and reconstructions p(v | h) (rows 2, 4)
            </span>
            <Mosaic image={reconstructions} range={UNIT_RANGE} ariaLabel="Data patterns and their reconstructions" />
          </div>
        </div>
      </div>
    </Figure>
  )
}
