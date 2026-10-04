import { useMemo, useState } from 'react'
import {
  ControlRow,
  Figure,
  Player,
  Plot,
  Plots,
  Points,
  Readout,
  Select,
  formatNumber,
  useAxis,
} from 'aifn-render'
import {
  autoencoderRun,
  type AutoencoderCheckpoint,
  type AutoencoderKind,
  type AutoencoderRun,
} from 'aifn-applied/generative/autoencoders'
import { moons, pinwheel } from 'aifn-applied/data/synthetic'
import { stream } from 'aifn/foundation/random'

const DATASETS = [
  { value: 'moons', label: 'Two moons dataset' },
  { value: 'pinwheel', label: 'Pinwheel dataset' },
]

const MODELS = [
  { value: 'vae', label: 'Standard VAE (β = 1.0)' },
  { value: 'beta-vae', label: 'β-VAE (β = 4.0, stronger latent pressure)' },
  { value: 'autoencoder', label: 'Deterministic Autoencoder (no KL prior)' },
]

export function VaeExplorer() {
  const [dataChoice, setDataChoice] = useState<'moons' | 'pinwheel'>('moons')
  const [modelChoice, setModelChoice] = useState<string>('vae')
  const [steps, setSteps] = useState(250)
  const [checkpointIndex, setCheckpointIndex] = useState(0)

  // Generate synthetic dataset
  const dataset = useMemo(() => {
    const s = stream('vae-data-stream')
    if (dataChoice === 'moons') {
      const data = moons(s, { n: 400, noise: 0.08 })
      return { x: data.x, y: data.y }
    } else {
      const data = pinwheel(s, { n: 400, arms: 4, noise: 0.05 })
      return { x: data.x, y: data.y }
    }
  }, [dataChoice])

  // Run autoencoder training
  const run: AutoencoderRun | null = useMemo(() => {
    let beta = 1.0
    let kind: AutoencoderKind = 'vae'
    if (modelChoice === 'beta-vae') {
      beta = 4.0
    } else if (modelChoice === 'autoencoder') {
      kind = 'autoencoder'
      beta = 0.0
    }

    const gen = autoencoderRun(dataset, {
      kind,
      beta,
      latent: 2,
      hidden: [32, 32],
      steps,
      stepSize: 0.003,
      batchSize: 64,
      checkpoints: 20,
      shown: 150,
      samples: 150,
      seed: 42,
    })

    let last: AutoencoderRun | null = null
    for (const snapshot of gen) {
      last = snapshot
    }
    return last
  }, [dataset, modelChoice, steps])

  const checkpoints = run?.checkpoints ?? []
  const numCheckpoints = checkpoints.length
  const currentIdx = Math.min(checkpointIndex, Math.max(0, numCheckpoints - 1))
  const currentCheckpoint: AutoencoderCheckpoint | null = checkpoints[currentIdx] ?? null

  // Real data points
  const dataPoints = useMemo(() => {
    if (!run) return { x: [], y: [] }
    const flat = run.data
    const n = Math.min(200, flat.length / 2)
    const xs: number[] = []
    const ys: number[] = []
    for (let i = 0; i < n; i++) {
      xs.push(flat[2 * i])
      ys.push(flat[2 * i + 1])
    }
    return { x: xs, y: ys }
  }, [run])

  // Reconstructions at current checkpoint
  const reconPoints = useMemo(() => {
    if (!currentCheckpoint) return { x: [], y: [] }
    const flat = currentCheckpoint.reconstructions
    const n = flat.length / 2
    const xs: number[] = []
    const ys: number[] = []
    for (let i = 0; i < n; i++) {
      xs.push(flat[2 * i])
      ys.push(flat[2 * i + 1])
    }
    return { x: xs, y: ys }
  }, [currentCheckpoint])

  // Latent codes at current checkpoint
  const latentCodes = useMemo(() => {
    if (!currentCheckpoint) return { x: [], y: [] }
    const flat = currentCheckpoint.codes
    const n = Math.min(200, flat.length / 2)
    const xs: number[] = []
    const ys: number[] = []
    for (let i = 0; i < n; i++) {
      xs.push(flat[2 * i])
      ys.push(flat[2 * i + 1])
    }
    return { x: xs, y: ys }
  }, [currentCheckpoint])

  // Prior samples decoded from z ~ N(0, I)
  const priorSamples = useMemo(() => {
    if (!currentCheckpoint) return { x: [], y: [] }
    const flat = currentCheckpoint.samples
    const n = flat.length / 2
    const xs: number[] = []
    const ys: number[] = []
    for (let i = 0; i < n; i++) {
      xs.push(flat[2 * i])
      ys.push(flat[2 * i + 1])
    }
    return { x: xs, y: ys }
  }, [currentCheckpoint])

  const dataAxisX = useAxis({ label: 'x₁', range: [-3, 3] })
  const dataAxisY = useAxis({ label: 'x₂', range: [-3, 3] })

  const latentAxisX = useAxis({ label: 'Latent z₁', range: [-3.5, 3.5] })
  const latentAxisY = useAxis({ label: 'Latent z₂', range: [-3.5, 3.5] })

  const lossSteps = run?.history.step ?? []
  const reconLoss = run?.history.reconstruction ?? []
  const klLoss = run?.history.regulariser ?? []

  const lossIdx = currentCheckpoint
    ? lossSteps.findIndex((s) => s >= currentCheckpoint.step)
    : -1
  const curRecon = lossIdx >= 0 ? reconLoss[lossIdx] : reconLoss[reconLoss.length - 1]
  const curKl = lossIdx >= 0 ? klLoss[lossIdx] : klLoss[klLoss.length - 1]

  return (
    <Figure
      title="Variational Autoencoder (VAE) Latent Manifold and Reconstruction"
      purpose="Explore generative reconstructions, latent Gaussian space clustering, and β-VAE disentanglement tradeoffs."
      caption={
        'Interactive Variational Autoencoder training (Kingma & Welling, 2014; Higgins et al., 2017). Left: data space showing training points (grey), decoder reconstructions (blue), and prior samples decoded from z ~ N(0, I) (coloured). Right: latent space (z₁, z₂) where the encoder maps points into a continuous Gaussian manifold. In a deterministic autoencoder, the latent space has arbitrary holes, causing prior samples to fail; in VAE, the KL term pulls the latent codes into a standard normal ball.'
      }
    >
      <ControlRow label="Model configuration">
        <Select
          label="Dataset"
          value={dataChoice}
          options={DATASETS}
          onChange={(v) => setDataChoice(v as 'moons' | 'pinwheel')}
        />
        <Select
          label="Architecture"
          value={modelChoice}
          options={MODELS}
          onChange={setModelChoice}
        />
        <Select
          label="Training updates"
          value={String(steps)}
          options={[
            { value: '100', label: '100 updates' },
            { value: '250', label: '250 updates' },
            { value: '500', label: '500 updates' },
          ]}
          onChange={(v) => setSteps(Number(v))}
        />
      </ControlRow>

      <ControlRow label="Training progression">
        <Player
          count={Math.max(1, numCheckpoints)}
          value={currentIdx}
          onChange={setCheckpointIndex}
        />
      </ControlRow>

      <Plots>
        <Plot x={dataAxisX} y={dataAxisY} title="Data space: real, reconstructions & prior samples">
          <Points
            x={dataPoints.x}
            y={dataPoints.y}
            slot={0}
            size={4}
            thin={true}
          />
          <Points
            x={reconPoints.x}
            y={reconPoints.y}
            slot={1}
            size={4}
            thin={true}
          />
          <Points
            x={priorSamples.x}
            y={priorSamples.y}
            slot={2}
            size={5}
          />
        </Plot>

        <Plot x={latentAxisX} y={latentAxisY} title="Latent space representation (z₁, z₂)">
          <Points
            x={latentCodes.x}
            y={latentCodes.y}
            slot={0}
            size={4}
          />
        </Plot>
      </Plots>

      <ControlRow label="Diagnostics">
        <Readout
          label="Step"
          value={currentCheckpoint ? currentCheckpoint.step : '—'}
        />
        <Readout
          label="Reconstruction error (MSE)"
          value={curRecon !== undefined ? formatNumber(Number(curRecon.toFixed(3))) : '—'}
        />
        <Readout
          label="KL divergence (nats)"
          value={curKl !== undefined ? formatNumber(Number(curKl.toFixed(3))) : '—'}
        />
        <Readout
          label="Latent dimension"
          value="2"
        />
      </ControlRow>
    </Figure>
  )
}
