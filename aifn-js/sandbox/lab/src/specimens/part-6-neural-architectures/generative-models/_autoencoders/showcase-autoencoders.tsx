/**
 * Autoencoders trained in the worker (`aifn-methods/generative/autoencoders`' `autoencoderRun`) on 2-d point clouds or
 * 5 × 7 digits: reconstructions, prior samples and the decoded code grid against the data, the codes of the data, and
 * the loss parts, played over checkpoints.
 */
import { useMemo, useState } from 'react'
import type { AutoencoderCheckpoint, AutoencoderRun, AutoencoderRunOptions } from 'aifn-methods/generative/autoencoders'
import { Player } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { call, choice, float, int, row, setting, useFigureState, type Task } from '@lab/state'
import { TrainControls, useTrainedRun } from '@lab/views'
import { Curve, formatNumber, Handle, Plot, Plots, Points, Raster, Readout, useAxis } from '@lab/viz'

const fmt = (v: number, digits = 3) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(digits))) : '—')

const DATASETS = {
  moons: { label: 'two moons', key: 'moons', knobs: { n: 600, noise: 0.06, spacing: 'random' } },
  spirals: { label: 'two spirals', key: 'spirals', knobs: { n: 600, arms: 2, noise: 0.03 } },
  ring: { label: '8 Gaussians on a ring', key: 'gaussianRing', knobs: { n: 600 } },
  digits: { label: '5 × 7 digits', key: 'digits', knobs: { perClass: 40, flip: 0.03, noise: 0 } },
} as const
type DataKey = keyof typeof DATASETS
const KINDS = [
  { value: 'autoencoder', label: 'autoencoder' },
  { value: 'vae', label: 'VAE' },
  { value: 'cvae', label: 'conditional VAE' },
  { value: 'vqvae', label: 'VQ-VAE' },
] as const
type Kind = (typeof KINDS)[number]['value']

type Settings = { data: DataKey; seed: number; options: AutoencoderRunOptions }

const task = ({ data, seed, options }: Settings): Task<AutoencoderRun> => {
  const d = DATASETS[data]
  return call<AutoencoderRun>(
    'applied/generative/autoencoders/autoencoderRun',
    call(`applied/data/synthetic/${d.key}`, call('foundation/random/stream', seed), d.knobs),
    options,
  )
}

const columns = (a: ArrayLike<number>, d: number, n = a.length / d) => ({
  x: Array.from({ length: n }, (_, i) => a[d * i]),
  y: Array.from({ length: n }, (_, i) => a[d * i + 1]),
})

/** Images (rows of 35 = 7 × 5 pixels) laid out in a grid of `cols` columns with a 1-pixel gap, as raster rows. */
function tiles(rows: ArrayLike<number>, count: number, cols: number): number[][] {
  const r = Math.ceil(count / cols)
  const H = r * 8 - 1
  const W = cols * 6 - 1
  const out = Array.from({ length: H }, () => new Array<number>(W).fill(NaN))
  for (let k = 0; k < count; k++) {
    const R = Math.floor(k / cols)
    const C = k % cols
    for (let i = 0; i < 7; i++)
      for (let j = 0; j < 5; j++)
        out[H - 1 - (R * 8 + i)][C * 6 + j] = Math.min(1, Math.max(0, rows[k * 35 + i * 5 + j]))
  }
  return out
}

export function AutoencoderShowcase() {
  const state = useFigureState({
    setup: row('1 · data and model', {
      data: choice(
        (Object.keys(DATASETS) as DataKey[]).map((k) => ({ value: k, label: DATASETS[k].label })),
        'moons',
        { label: 'data' },
      ),
      kind: choice(KINDS, 'vae', { label: 'model' }),
      latent: int(2, { ge: 1, le: 8, suggestions: [1, 2, 4], label: 'code dimension' }),
      beta: float(1, { ge: 0, suggestions: [0.1, 0.25, 1, 4], label: 'β (KL or commitment weight)' }),
      codes: int(16, {
        ge: 2,
        le: 128,
        suggestions: [4, 8, 16, 32],
        label: 'codebook size',
        when: (v) => v.kind === 'vqvae',
      }),
    }),
    run: row('2 · training run', {
      steps: int(1500, { ge: 1, suggestions: [500, 1500, 3000], label: 'Adam steps' }),
      rate: float(2e-3, { gt: 0, scale: 'log10', suggestions: [5e-4, 1e-3, 2e-3, 5e-3], label: 'step size' }),
      seed: int(1, { ge: 0, le: 9999, label: 'seed' }),
    }),
    show: row('3 · show', {
      grid: setting(true, 'decoded code grid (2-d code)'),
      samples: setting(true, 'samples from the prior'),
    }),
  })
  const { data, kind, latent, beta, codes } = state.setup
  const images = data === 'digits'
  const settings: Settings = {
    data: data as DataKey,
    seed: state.run.seed,
    options: {
      kind: kind as Kind,
      latent,
      beta: kind === 'vqvae' && beta === 1 ? 0.25 : beta,
      codes,
      steps: state.run.steps,
      stepSize: state.run.rate,
      likelihood: images ? 'bernoulli' : 'gaussian',
      seed: state.run.seed,
      shown: images ? 24 : 600,
      samples: images ? 30 : 400,
      grid: images ? 7 : 15,
    },
  }
  const trained = useTrainedRun(settings, task)
  const run = trained.run.value
  const shots = useMemo(() => run?.checkpoints ?? [], [run])
  const [picked, setPicked] = useState<{ run: Settings | null; index: number } | null>(null)
  const index = Math.min(picked && picked.run === trained.trained ? picked.index : 0, Math.max(0, shots.length - 1))
  const shot: AutoencoderCheckpoint | undefined = shots[index]
  const pick = (i: number) => setPicked({ run: trained.trained, index: i })
  const pickStep = (step: number) => {
    let best = 0
    shots.forEach((c, i) => {
      if (Math.abs(c.step - step) < Math.abs(shots[best].step - step)) best = i
    })
    pick(best)
  }
  const trainedImages = trained.trained?.data === 'digits'
  const box = 2.5
  const x1 = useAxis({ label: 'x₁', range: [-box, box] })
  const x2 = useAxis({ label: 'x₂', range: [-box, box], equal: x1 })
  const z1 = useAxis({ label: 'z₁', range: [-4, 4] })
  const z2 = useAxis({
    label: run && run.latent === 1 ? 'label' : 'z₂',
    range: [-4, 4],
    equal: run && run.latent === 1 ? undefined : z1,
  })
  const stepAxis = useAxis({ label: 'Adam step', range: [0, run?.steps ?? 1], key: run?.steps })
  const lossAxis = useAxis({
    label: 'loss (nats per point)',
    log: true,
    hold: 'union',
    key: JSON.stringify(trained.trained),
  })
  const imgX = useAxis({ label: '', range: [0, 1] })
  const imgY = useAxis({ label: '', range: [0, 1] })
  const D = run?.dimension ?? 2
  const L = run?.latent ?? 2
  const codesXY = useMemo(() => {
    if (!shot) return null
    const n = shot.codes.length / L
    return {
      x: Array.from({ length: n }, (_, i) => shot.codes[i * L]),
      y: Array.from({ length: n }, (_, i) => (L > 1 ? shot.codes[i * L + 1] : 0)),
    }
  }, [shot, L])
  const labels = useMemo(() => Array.from(run?.labels ?? []), [run])
  const marker = shot ? <Handle kind="x" at={shot.step} onDrag={pickStep} label={`step ${shot.step}`} /> : null
  const empty = !trained.trained ? 'press Train to start' : !shot ? 'training…' : undefined
  const done = run?.done ?? 0
  const total = trained.trained?.options.steps ?? settings.options.steps ?? 1
  const hist = run?.history
  // The loss parts at the shown checkpoint's step.
  const at = hist && shot ? hist.step.findIndex((t) => t >= shot.step) : -1
  return (
    <Figure
      title="Autoencoders, VAEs and VQ-VAEs"
      purpose="An autoencoder squeezes data through a small code and back; a VAE makes the code a distribution pulled towards N(0, I), so that decoding prior draws gives new data, with β trading reconstruction for a tidier code; a conditional VAE also feeds the label to both networks; a VQ-VAE snaps codes to a learnt codebook."
      state={state}
      defaultSize="XL"
      controls={
        <>
          <TrainControls run={trained as never} progress={done / total} progressText={`${done} / ${total} steps`} />
          <ControlRow label="4 · checkpoints">
            <Player
              className="col-span-full"
              value={index}
              onChange={pick}
              count={Math.max(1, shots.length)}
              label="checkpoint"
              format={(i) => `step ${shots[i]?.step ?? 0}`}
            />
          </ControlRow>
        </>
      }
      readouts={
        <>
          <Readout label="step" value={shot ? shot.step : '—'} />
          <Readout label="reconstruction" value={hist && at >= 0 ? fmt(hist.reconstruction[at]) : '—'} />
          <Readout
            label={run?.kind === 'vqvae' ? 'codebook + commitment' : 'KL'}
            value={hist && at >= 0 ? fmt(hist.regulariser[at]) : '—'}
          />
          {run?.kind === 'vqvae' && <Readout label="codes used" value={shot ? `${shot.used} / ${codes}` : '—'} />}
        </>
      }
      caption="aifn autoencoderRun: MLP encoder and decoder (64 → 64, ReLU) trained by Adam on minibatches of 64; Gaussian reconstruction with σ = 0.1 for points, Bernoulli for the binary digits. Points: data (grey), reconstructions of every point, samples decoded from N(0, I) (from a Gaussian fitted to the codes for the plain autoencoder, from random codebook entries for the VQ-VAE; for the conditional VAE, from each label in turn), and with a 2-d code the decoder over a 15 × 15 grid of codes on [−3, 3]², whose image is the learnt manifold. Digits: originals above reconstructions, then samples (one row per label for the conditional VAE). Middle: the codes of the data, coloured by label, with the codebook in ink for the VQ-VAE. Right: the loss parts. Play the checkpoints or drag the step marker."
    >
      <Plots cols={3}>
        {trainedImages && shot ? (
          <Plot x={imgX} y={imgY} title="reconstructions (top) and samples (bottom)">
            <Raster
              x={Array.from({ length: 6 * 12 - 1 }, (_, j) => (j + 0.5) / (6 * 12 - 1))}
              y={Array.from({ length: 8 * 5 - 1 }, (_, i) => (i + 0.5) / (8 * 5 - 1))}
              z={tiles(
                Float64Array.from([...shot.reconstructions.slice(0, 24 * 35), ...shot.samples.slice(0, 30 * 35)]),
                54,
                12,
              )}
              scale="sequential"
              range={[0, 1]}
              valueLabel="pixel"
            />
          </Plot>
        ) : (
          <Plot x={x1} y={x2} title={empty ?? 'data, reconstructions and samples'}>
            {run && <Points name="data" {...columns(run.data, D)} muted thin />}
            {shot && state.show.samples && <Points name="samples" {...columns(shot.samples, D)} slot={6} thin />}
            {shot && <Points name="reconstructions" {...columns(shot.reconstructions, D)} slot={4} thin />}
            {shot && state.show.grid && shot.grid && (
              <Points name="decoded grid" {...columns(shot.grid, D)} emphasis size={4} />
            )}
          </Plot>
        )}
        <Plot x={z1} y={z2} title="codes of the data">
          {codesXY && <Points name="codes" x={codesXY.x} y={codesXY.y} group={labels} thin />}
          {shot?.codebook && <Points name="codebook" {...columns(shot.codebook, L)} emphasis size={7} />}
        </Plot>
        <Plot x={stepAxis} y={lossAxis} title="loss parts">
          {hist && (
            <Curve name="reconstruction" slot={2} x={hist.step} y={hist.reconstruction.map((v) => Math.max(1e-3, v))} />
          )}
          {hist && (
            <Curve
              name={run?.kind === 'vqvae' ? 'VQ terms' : 'KL'}
              slot={3}
              x={hist.step}
              y={hist.regulariser.map((v) => Math.max(1e-3, v))}
            />
          )}
          {marker}
        </Plot>
      </Plots>
    </Figure>
  )
}
