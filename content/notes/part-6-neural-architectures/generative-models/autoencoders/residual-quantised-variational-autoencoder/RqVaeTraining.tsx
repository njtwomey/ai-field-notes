import { useContext, useMemo, useState, type ReactNode } from 'react'
import {
  call,
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
  Readout,
  Shapes,
  useAxis,
  useComputed,
  useFigureState,
  useScaleColor,
  type FilledShape,
} from 'aifn-render'
import type { AutoencoderRun, AutoencoderRunOptions } from 'aifn-methods/generative/autoencoders'
import { moons, pinwheel } from 'aifn-methods/data/synthetic'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'

const N = 400
const CHECKPOINTS = 30
const PANEL = 280

/** Plots inside a Figure take the frame's height; this gives the plots inside it a fixed height instead. */
function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}

const DATA = {
  pinwheel: { label: 'pinwheel (4 arms)', options: { n: N, arms: 4, noise: 0.05 } },
  moons: { label: 'two moons', options: { n: N, noise: 0.08 } },
} as const
type DataKind = keyof typeof DATA

/** Group 2-d rows by their stage-1 codeword, one series per codeword, so each cluster keeps one colour throughout. */
function byFirstCode(xy: ArrayLike<number>, first: ArrayLike<number>, k: number, rows: number) {
  return Array.from({ length: k }, (_, c) => {
    const x: number[] = []
    const y: number[] = []
    for (let i = 0; i < rows; i++)
      if (first[i] === c) {
        x.push(xy[2 * i])
        y.push(xy[2 * i + 1])
      }
    return { x, y }
  })
}

export function RqVaeTraining() {
  const state = useFigureState({
    data: choice(
      (Object.keys(DATA) as DataKind[]).map((value) => ({ value, label: DATA[value].label })),
      'pinwheel',
      { label: 'data' },
    ),
    codes: choice([2, 3, 4, 8], 4, { label: 'codewords per stage K' }),
    depth: int(3, { ge: 1, le: 4, label: 'stages D' }),
    shownStages: int(3, { ge: 1, le: 4, label: 'stages used d (at most D)' }),
    beta: float(0.25, { ge: 0, le: 2, suggestions: [0.1, 0.25, 1], label: 'commitment weight β' }),
    steps: int(600, { ge: 100, le: 4000, suggestions: [300, 600, 1500], label: 'Adam steps' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
  })
  const { data, codes, depth, shownStages, beta, steps, seed } = state

  // The data, made from its seed both here (to draw it) and in the worker (to train on), so no tensor crosses over.
  const points = useMemo(() => {
    const s = stream(`rq-vae/${data}`)
    const made = data === 'moons' ? moons(s, DATA.moons.options) : pinwheel(s, DATA.pinwheel.options)
    const flat = toFlat(made.x)
    return { flat, x: flat.filter((_, i) => i % 2 === 0), y: flat.filter((_, i) => i % 2 === 1) }
  }, [data])

  const options: AutoencoderRunOptions = useMemo(
    () => ({
      kind: 'rqvae',
      latent: 2,
      hidden: [32, 32],
      codes,
      depth,
      beta,
      observationSd: 1,
      steps,
      stepSize: 3e-3,
      batchSize: 64,
      checkpoints: CHECKPOINTS,
      shown: N,
      samples: 0,
      seed,
    }),
    [codes, depth, beta, steps, seed],
  )
  const key = `${data}|${codes}|${depth}|${beta}|${steps}|${seed}`
  // The run streams from the compute worker: each partial answer holds every checkpoint so far.
  const run = useComputed(
    () =>
      call<AutoencoderRun>(
        'generative/autoencoders/autoencoderRun',
        call(
          `data/synthetic/${data}`,
          call('foundation/random/stream', `rq-vae/${data}`),
          DATA[data as DataKind].options,
        ),
        options,
      ),
    [data, options],
    {
      mode: 'worker',
      initial: null as { run: AutoencoderRun; key: string } | null,
      then: (r) => ({ run: r, key }),
      cancelAfter: 400,
    },
  )
  const current = run.value?.key === key ? run.value.run : null
  const count = current?.checkpoints.length ?? 0
  const [pos, setPos] = useState({ key, at: 0 })
  const at = pos.key === key ? Math.min(pos.at, Math.max(0, count - 1)) : 0
  const stage = Math.min(shownStages, depth) - 1
  const cp = current?.checkpoints[at] ?? null

  const view = useMemo(() => {
    if (!cp?.codebook || !cp.stageCodes || !cp.codeTree || !cp.partialReconstructions) return null
    const L = 2
    const codesFlat = cp.stageCodes
    const book = cp.codebook
    const first = Array.from({ length: N }, (_, i) => codesFlat[i * depth])
    // The quantised latent after `stage + 1` stages: the sum of the chosen codewords of the first stages. Stage d's
    // codebook is rows dK to (d + 1)K − 1 of the stacked codebooks.
    const partial = new Float64Array(N * L)
    for (let i = 0; i < N; i++)
      for (let d = 0; d <= stage; d++) {
        const row = d * codes + codesFlat[i * depth + d]
        for (let j = 0; j < L; j++) partial[i * L + j] += book[row * L + j]
      }
    const tree = cp.codeTree.filter((n) => n.depth >= 1)
    return {
      first,
      encoder: { x: cp.codes.filter((_, i) => i % 2 === 0), y: cp.codes.filter((_, i) => i % 2 === 1) },
      latent: byFirstCode(partial, first, codes, N),
      recon: byFirstCode(cp.partialReconstructions[stage], first, codes, N),
      tree,
      prefixes: tree.filter((n) => n.depth === stage + 1).length,
    }
  }, [cp, stage, codes, depth])

  const history = current?.history
  const stageColour = useScaleColor('sequential')
  const stageCurves = useMemo(() => {
    if (!history) return []
    return Array.from({ length: depth }, (_, d) => ({
      x: history.step,
      y: history.stageError.map((e) => e[d]),
      color: stageColour(depth === 1 ? 1 : 0.35 + (0.65 * d) / (depth - 1)),
    }))
  }, [history, depth, stageColour])

  // The code tree as an icicle: one row per stage, each occupied prefix a bar spanning its rows, coloured by its
  // stage-1 codeword. Rows are ordered by code, so a node's rows are [offset, offset + count).
  const icicle: FilledShape[] = useMemo(
    () =>
      (view?.tree ?? []).map((n) => {
        const y0 = n.depth - 1 - 0.42
        const y1 = n.depth - 1 + 0.42
        const x0 = n.offset + 0.15
        const x1 = n.offset + n.count - 0.15
        return {
          contours: [
            [
              [x0, y0],
              [x1, y0],
              [x1, y1],
              [x0, y1],
            ],
          ],
          tone: n.prefix[0],
          opacity: n.depth === stage + 1 ? 0.9 : 0.4,
        }
      }),
    [view, stage],
  )

  const lx = useAxis({ label: 'latent z₁' })
  const ly = useAxis({ label: 'latent z₂', equal: lx })
  const dx = useAxis({ label: 'x₁', range: [-3, 3] })
  const dy = useAxis({ label: 'x₂', range: [-3, 3], equal: dx })
  const sx = useAxis({ label: 'Adam step', range: [0, steps] })
  const sy = useAxis({ label: 'mean ‖z_e − ẑ⁽ᵈ⁾‖²', log: true })
  const tx = useAxis({ label: 'training points, ordered by code', range: [0, N] })
  const ty = useAxis({
    label: 'stage',
    categories: Array.from({ length: depth }, (_, d) => `stage ${d + 1}`),
    inverse: true,
  })

  const latest = history && history.stageError.length > 0 ? history.stageError.at(-1) : undefined

  return (
    <Figure
      title="Training an RQ-VAE"
      state={state}
      defaultSize="XL"
      caption={`An RQ-VAE with a two-dimensional latent, trained by Adam on ${N} points: encoder and decoder are multilayer perceptrons with two hidden layers of 32 units, each of the D stages has its own codebook of K codewords, started from residual k-means of the encoder outputs, and the training runs in the browser and fills in as it goes. Top left: the encoder outputs z_e (grey) and the quantised latents after the stage chosen, coloured by stage-1 codeword. Top right: the data (grey) and its reconstructions decoded from the first d stages. Bottom left: the mean squared distance between z_e and its quantisation after each stage over training, one line per stage, darker for later stages; each stage removes a share of what the stages before it left. Bottom right: the code tree, stage 1 at the top. Each bar is one occupied code prefix and spans the training points that share it, so wide bars are popular prefixes and the bottom row holds the distinct whole codes. Step the stages used d from 1 to D to watch clusters split inside clusters. Coding is greedy: the error averaged over the data falls at every stage, but one point's error can rise.`}
      controls={
        <Player
          value={at}
          onChange={(s) => setPos({ key, at: s })}
          count={Math.max(1, count)}
          label="checkpoint"
          format={(s) => `step ${current?.checkpoints[s]?.step ?? 0}`}
        />
      }
      readouts={
        <>
          <Readout label="step" value={cp ? String(cp.step) : '…'} />
          <Readout label="distinct prefixes" value={view ? `${view.prefixes} of ≤ ${codes ** (stage + 1)}` : '…'} />
          <Readout label="distinct whole codes" value={cp ? String(cp.uniqueCodes) : '…'} />
          <Readout label="codewords used by stage" value={cp?.usedByStage ? cp.usedByStage.join(', ') : '…'} />
          <Readout
            label="final error by stage"
            value={latest ? Array.from(latest, (e) => formatNumber(e)).join(', ') : '…'}
          />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <FixedHeight height={PANEL}>
          <Plot x={lx} y={ly} title="latent space">
            {view && <Points name="encoder outputs z_e" x={view.encoder.x} y={view.encoder.y} muted dense />}
            {view?.latent.map((g, k) => (
              <Points key={k} name={`stage-1 codeword ${k + 1}`} x={g.x} y={g.y} slot={k} size={7} />
            ))}
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={dx} y={dy} title="data space">
            <Points name="data" x={points.x} y={points.y} muted dense />
            {view?.recon.map((g, k) => (
              <Points key={k} name={`stage-1 codeword ${k + 1}`} x={g.x} y={g.y} slot={k} size={7} />
            ))}
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={sx} y={sy} title="error after each stage">
            {stageCurves.map((c, d) => (
              <Curve key={d} name={`after stage ${d + 1}`} x={c.x} y={c.y} color={c.color} />
            ))}
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={tx} y={ty} title="code tree">
            <Shapes name="code prefixes" shapes={icicle} />
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
