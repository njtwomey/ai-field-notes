import { useMemo, useState } from 'react'
import {
  ImagePlot,
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  type Handle,
  type XYSeries,
} from 'aifn-render'
import { rng } from '@/lib/math'
import { PlayButton } from '../_shared/Playback'
import { usePlayLoop } from '../_shared/usePlayLoop'
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

/**
 * An RBM with 16 visible units learns the 30 bars-and-stripes patterns on a 4 × 4 grid, trained three ways at once.
 * The exact log-likelihood is computed by summing over the hidden states.
 */
export function RbmTrainer() {
  const [H, setH] = useState(8)
  const [rate, setRate] = useState(0.2)
  const [seed, setSeed] = useState(1)
  const [method, setMethod] = useState<Method>('cd10')
  const [epoch, setEpoch] = useState(EPOCHS)
  const [steps, setSteps] = useState(20)
  const [temperature, setTemperature] = useState(1)
  const [playing, setPlaying] = useState(false)

  const runs = useMemo(
    () =>
      Object.fromEntries(
        METHODS.map(({ value }) => [
          value,
          trainRbm(DATA, H, { epochs: EPOCHS, rate, method: value, batch: 5, uniform: rng(seed).uniform, every: 5 }),
        ]),
      ) as Record<Method, ReturnType<typeof trainRbm>>,
    [H, rate, seed],
  )
  const snap = runs[method][epoch]
  const model: Rbm = useMemo(() => ({ V: SIDE * SIDE, H, W: snap.W, b: snap.b, c: snap.c }), [snap, H])

  usePlayLoop(playing, 40, (k) => {
    const next = Math.min(EPOCHS, epoch + k)
    setEpoch(next)
    if (next >= EPOCHS) setPlaying(false)
    return next < EPOCHS
  })

  // Filters: column j of W, the weights from hidden unit j to the 16 pixels, on a symmetric scale.
  const filters = useMemo(() => {
    const images = Array.from({ length: H }, (_, j) =>
      Array.from({ length: SIDE * SIDE }, (__, i) => snap.W[i * H + j]),
    )
    const bound = Math.max(0.5, ...Array.from(snap.W, Math.abs))
    return { ...mosaic(images, SIDE, SIDE, COLS, 1, 0), bound }
  }, [snap, H])

  // Gibbs chains from random visible states, `steps` block-Gibbs steps at temperature T, drawn from a fixed seed.
  const chains = useMemo(() => {
    const { uniform } = rng(seed + 77)
    const out: Float64Array[] = []
    for (let k = 0; k < CHAINS; k++) {
      let v = sample(new Float64Array(SIDE * SIDE).fill(0.5), uniform)
      for (let s = 0; s < steps; s++)
        v = sample(visibleProbs(model, sample(hiddenProbs(model, v, temperature), uniform), temperature), uniform)
      out.push(v)
    }
    return mosaic(out, SIDE, SIDE, COLS, 1, 0.5)
  }, [model, steps, temperature, seed])

  // Reconstructions: data (top row) and p(v | h) with h drawn from p(h | v) (bottom row).
  const reconstructions = useMemo(() => {
    const { uniform } = rng(seed + 99)
    const recon = SHOWN.map((v) => visibleProbs(model, sample(hiddenProbs(model, v), uniform)))
    // Rows alternate: five data patterns, their reconstructions, the next five, theirs.
    const rows = [...SHOWN.slice(0, COLS), ...recon.slice(0, COLS), ...SHOWN.slice(COLS), ...recon.slice(COLS)]
    return mosaic(rows, SIDE, SIDE, COLS, 1, 0.5)
  }, [model, seed])

  const epochs = useMemo(() => runs.cd1.map((s) => s.epoch), [runs])
  const series = useMemo((): XYSeries[] => {
    const lines: XYSeries[] = METHODS.map(({ value, label, slot }) => ({
      name: label,
      type: 'line',
      x: epochs,
      y: runs[value].map((s) => s.logLik),
      slot,
    }))
    lines.push({ name: 'optimum −log 30', type: 'line', x: [0, EPOCHS], y: [OPTIMUM, OPTIMUM], dashed: true, slot: 3 })
    lines.push({ name: 'shown', type: 'scatter', x: [epoch], y: [runs[method][epoch].logLik], emphasis: true })
    return lines
  }, [epochs, runs, epoch, method])
  const handles = useMemo(
    (): Handle[] => [
      {
        kind: 'x',
        at: epoch,
        label: 'epoch',
        onDrag: (x) => {
          setPlaying(false)
          setEpoch(Math.max(0, Math.min(EPOCHS, Math.round(x))))
        },
      },
    ],
    [epoch],
  )

  return (
    <Interactive
      title="Training an RBM on bars and stripes"
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
      controls={
        <>
          <ParamChoice
            label="hidden units"
            value={String(H)}
            onChange={(v) => setH(Number(v))}
            options={['4', '8', '10'].map((v) => ({ value: v, label: v }))}
          />
          <ParamSlider label="learning rate" value={rate} onChange={setRate} min={0.05} max={0.5} step={0.05} />
          <ParamSlider label="seed" value={seed} onChange={setSeed} min={1} max={10} step={1} />
          <ParamChoice
            label="shown method"
            value={method}
            onChange={setMethod}
            options={METHODS.map(({ value, label }) => ({ value, label }))}
          />
          <ParamSlider
            label="epoch"
            value={epoch}
            onChange={(v) => {
              setPlaying(false)
              setEpoch(v)
            }}
            min={0}
            max={EPOCHS}
            step={1}
            withArrows
          />
          <ParamSlider label="Gibbs steps" value={steps} onChange={setSteps} min={0} max={100} step={1} withArrows />
          <ParamSlider
            label="sampling temperature T"
            value={temperature}
            onChange={setTemperature}
            min={0.2}
            max={3}
            step={0.05}
          />
          <div className="flex gap-2 self-end">
            <PlayButton
              playing={playing}
              onToggle={() => {
                if (!playing && epoch >= EPOCHS) setEpoch(0)
                setPlaying((p) => !p)
              }}
            />
          </div>
        </>
      }
      readout={
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
        <XYChart
          height={300}
          xLabel="epoch"
          yLabel="mean log p(v)"
          series={series}
          handles={handles}
          ariaLabel="Exact log-likelihood of the RBM during training by three methods"
        />
        <div className="grid grid-cols-3 content-start gap-3">
          <div className="flex flex-col gap-1">
            <span className="text-xs text-muted-foreground">filters: weights of each hidden unit (red +, blue −)</span>
            <ImagePlot
              width={filters.width}
              height={filters.height}
              xExtent={EXTENT}
              values={filters.values}
              scale="diverging"
              range={[-filters.bound, filters.bound]}
              ariaLabel="Weights of each hidden unit drawn as 4 by 4 images"
            />
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-xs text-muted-foreground">
              Gibbs chains after {steps} steps at T = {formatNumber(temperature)}
            </span>
            <ImagePlot
              width={chains.width}
              height={chains.height}
              values={chains.values}
              range={[0, 1]}
              ariaLabel="Visible states of ten Gibbs chains"
            />
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-xs text-muted-foreground">
              data (rows 1, 3) and reconstructions p(v | h) (rows 2, 4)
            </span>
            <ImagePlot
              width={reconstructions.width}
              height={reconstructions.height}
              values={reconstructions.values}
              range={[0, 1]}
              ariaLabel="Data patterns and their reconstructions"
            />
          </div>
        </div>
      </div>
    </Interactive>
  )
}
