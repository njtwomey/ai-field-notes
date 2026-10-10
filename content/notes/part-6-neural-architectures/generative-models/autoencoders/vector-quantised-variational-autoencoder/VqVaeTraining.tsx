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
  Raster,
  Readout,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import type { AutoencoderRun, AutoencoderRunOptions } from 'aifn-methods/generative/autoencoders'
import { moons, pinwheel } from 'aifn-methods/data/synthetic'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { assignNearest } from 'aifn-compute/numerics/neighbours'

const N = 400
const CHECKPOINTS = 40
const PANEL = 280
const CELLS = 61

/** Plots inside a Figure take the frame's height; this gives the plots inside it a fixed height instead. */
function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}

const rows = (flat: ArrayLike<number>) =>
  Array.from({ length: flat.length / 2 }, (_, i) => [flat[2 * i], flat[2 * i + 1]])
const columns = (flat: ArrayLike<number>, n = flat.length / 2) => ({
  x: Array.from({ length: n }, (_, i) => flat[2 * i]),
  y: Array.from({ length: n }, (_, i) => flat[2 * i + 1]),
})

export function VqVaeTraining() {
  const state = useFigureState({
    data: choice(
      [
        { value: 'pinwheel', label: 'pinwheel (4 arms)' },
        { value: 'moons', label: 'two moons' },
      ],
      'pinwheel',
      { label: 'data' },
    ),
    codes: choice([4, 8, 16], 8, { label: 'codebook size K' }),
    beta: float(0.25, { ge: 0, le: 4, suggestions: [0, 0.1, 0.25, 1, 2], label: 'commitment weight β' }),
    steps: int(1500, { ge: 100, le: 6000, suggestions: [500, 1500, 3000], label: 'Adam steps' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
  })
  const { data, codes, beta, steps, seed } = state

  // The data, made from its seed both here (to draw it) and in the worker (to train on), so no tensor crosses over.
  const spec = useMemo(
    () =>
      data === 'moons'
        ? { make: 'moons' as const, options: { n: N, noise: 0.08 } }
        : { make: 'pinwheel' as const, options: { n: N, arms: 4, noise: 0.05 } },
    [data],
  )
  const dataset = useMemo(() => {
    const s = stream(`vq-vae/${data}`)
    return spec.make === 'moons'
      ? moons(s, spec.options)
      : pinwheel(s, spec.options as { n: number; arms: number; noise: number })
  }, [data, spec])

  const options: AutoencoderRunOptions = useMemo(
    () => ({
      kind: 'vqvae',
      latent: 2,
      hidden: [32, 32],
      codes,
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
    [codes, beta, steps, seed],
  )
  const key = `${data}|${codes}|${beta}|${steps}|${seed}`
  // The run streams from the compute worker: each partial answer holds every checkpoint so far.
  const run = useComputed(
    () =>
      call<AutoencoderRun>(
        'generative/autoencoders/autoencoderRun',
        call(`data/synthetic/${spec.make}`, call('foundation/random/stream', `vq-vae/${data}`), spec.options),
        options,
      ),
    [spec, data, options],
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
  const cp = current?.checkpoints[at] ?? null

  const view = useMemo(() => {
    if (!current || !cp || !cp.codebook) return null
    const z = rows(cp.codes)
    const book = rows(cp.codebook)
    const a = assignNearest(z, book)
    const sizes = toFlat(a.sizes)
    let entropy = 0
    for (const c of sizes) if (c > 0) entropy -= (c / z.length) * Math.log(c / z.length)
    // The latent panel fits the encoder outputs and the codes, square, with the code cells drawn over it.
    let lo = Infinity
    let hi = -Infinity
    for (const v of [...cp.codes, ...cp.codebook]) {
      lo = Math.min(lo, v)
      hi = Math.max(hi, v)
    }
    const pad = 0.08 * (hi - lo || 1)
    const range = [lo - pad, hi + pad] as const
    const grid = Array.from({ length: CELLS }, (_, i) => range[0] + ((range[1] - range[0]) * i) / (CELLS - 1))
    const cellLabels = toFlat(
      assignNearest(
        grid.flatMap((y) => grid.map((x) => [x, y])),
        book,
      ).labels,
    )
    const cells = grid.map((_, r) => cellLabels.slice(r * CELLS, (r + 1) * CELLS))
    return {
      latent: columns(cp.codes),
      codebook: columns(cp.codebook),
      recon: columns(cp.reconstructions),
      error: a.inertia / z.length,
      used: sizes.filter((c) => c > 0).length,
      perplexity: Math.exp(entropy),
      range,
      grid,
      cells,
    }
  }, [current, cp])

  const points = useMemo(() => columns(toFlat(dataset.x)), [dataset])
  const history = current?.history
  const marker = useMemo(() => {
    if (!history || !cp) return null
    const i = history.step.findIndex((s) => s >= cp.step)
    return i < 0 ? null : { step: history.step[i], rec: history.reconstruction[i], reg: history.regulariser[i] }
  }, [history, cp])

  const zx = useAxis({ label: 'latent z₁', range: view?.range })
  const zy = useAxis({ label: 'latent z₂', range: view?.range, equal: zx })
  const dx = useAxis({ label: 'x₁', range: [-3, 3] })
  const dy = useAxis({ label: 'x₂', range: [-3, 3], equal: dx })
  const sx = useAxis({ label: 'Adam step', range: [0, steps] })
  const sy = useAxis({ label: 'loss on the training set', log: true })

  return (
    <Figure
      title="Training a VQ-VAE"
      state={state}
      defaultSize="XL"
      caption={`A VQ-VAE with a two-dimensional latent, trained by Adam on ${N} points. The encoder and decoder are multilayer perceptrons with two hidden layers of 32 units, the reconstruction term is the squared error divided by 2, and the codebook starts as K draws from a standard normal. The training runs in the browser and fills in as it goes. Left: the encoder outputs z_e of the training points (grey) and the codes (black), with the latent space shaded by the nearest code. Middle: the data (grey) and its reconstructions (black). The decoder only ever sees the K codes, so every reconstruction is one of at most K points. Right: the reconstruction term and the sum of the codebook and commitment terms, on a log scale; the dot marks the checkpoint played. At β = 0 nothing holds the encoder outputs near the codes: they drift outwards without limit, the codes chase them, and the vector-quantisation terms grow by orders of magnitude. Between β = 0.1 and 0.25 the encoder outputs gather around the codes and the reconstructions spread along the data. At β = 2 the encoder outputs are pulled so tightly onto the codes that some codes lose all their points.`}
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
          <Readout label="reconstruction term" value={marker ? formatNumber(marker.rec) : '…'} />
          <Readout label="mean ‖z_e − z_q‖²" value={view ? formatNumber(view.error) : '…'} />
          <Readout label="codes in use" value={view ? `${view.used} of ${codes}` : '…'} />
          <Readout label="perplexity of code use" value={view ? formatNumber(view.perplexity) : '…'} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <FixedHeight height={PANEL}>
          <Plot x={zx} y={zy} title="latent space">
            {view && (
              <Raster
                name="code cells"
                x={view.grid}
                y={view.grid}
                z={view.cells}
                scale="categorical"
                fillOpacity={0.14}
                boundary
                stale={run.stale}
              />
            )}
            {view && <Points name="encoder outputs z_e" x={view.latent.x} y={view.latent.y} muted dense />}
            {view && <Points name="codes" x={view.codebook.x} y={view.codebook.y} emphasis size={9} />}
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={dx} y={dy} title="data space">
            <Points name="data" x={points.x} y={points.y} muted dense />
            {view && <Points name="reconstructions" x={view.recon.x} y={view.recon.y} emphasis size={8} />}
          </Plot>
        </FixedHeight>
        <div className="md:col-span-2">
          <FixedHeight height={PANEL * 0.75}>
            <Plot x={sx} y={sy} title="loss">
              {history && <Curve name="reconstruction" x={history.step} y={history.reconstruction} slot={0} />}
              {history && <Curve name="codebook + commitment" x={history.step} y={history.regulariser} slot={1} />}
              {marker && (
                <Points
                  name="checkpoint"
                  x={[marker.step, marker.step]}
                  y={[marker.rec, marker.reg]}
                  emphasis
                  size={7}
                  live
                />
              )}
            </Plot>
          </FixedHeight>
        </div>
      </div>
    </Figure>
  )
}
