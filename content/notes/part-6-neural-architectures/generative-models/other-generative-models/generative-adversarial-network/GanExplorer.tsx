import { useMemo, useState } from 'react'
import {
  ControlRow,
  Curve,
  Figure,
  Player,
  Plot,
  Plots,
  Points,
  Raster,
  Readout,
  Select,
  formatNumber,
  useAxis,
} from 'aifn-render'
import { stream } from 'aifn/foundation/random'
import {
  gaussianRing,
  moons,
  pinwheel,
  spirals,
} from 'aifn-applied/data/synthetic'
import {
  ganRun,
  type GanCheckpoint,
  type GanRun,
} from 'aifn-applied/generative/gan'
import type { AdversarialGame } from 'aifn/learning/losses'

const DATASETS = [
  { value: 'ring', label: '8 Gaussians on a ring (mode collapse test)' },
  { value: 'pinwheel', label: 'Pinwheel (5 spiral arms)' },
  { value: 'moons', label: 'Two moons' },
  { value: 'spirals', label: 'Two spirals' },
]

const GAMES = [
  { value: 'non-saturating', label: 'Non-saturating (standard GAN: max log D(G(z)))' },
  { value: 'minimax', label: 'Minimax (original: min log(1 - D(G(z))))' },
  { value: 'wasserstein', label: 'Wasserstein + GP (WGAN-GP)' },
  { value: 'hinge', label: 'Hinge GAN' },
]

export function GanExplorer() {
  const [dataChoice, setDataChoice] = useState<'ring' | 'pinwheel' | 'moons' | 'spirals'>('ring')
  const [game, setGame] = useState<AdversarialGame>('non-saturating')
  const [criticSteps, setCriticSteps] = useState(2)
  const [steps, setSteps] = useState(200)
  const [checkpointIndex, setCheckpointIndex] = useState(0)

  // Run the training simulation
  const runResult = useMemo(() => {
    const s = stream(`gan-${dataChoice}`)
    let rawData
    if (dataChoice === 'ring') {
      rawData = gaussianRing(s, { n: 400 })
    } else if (dataChoice === 'pinwheel') {
      rawData = pinwheel(s, { n: 400 })
    } else if (dataChoice === 'moons') {
      rawData = moons(s, { n: 400, noise: 0.08 })
    } else {
      rawData = spirals(s, { n: 400, arms: 2, noise: 0.04 })
    }

    const gen = ganRun(rawData, {
      game,
      criticSteps,
      hidden: [24, 24],
      steps,
      critic: { name: 'adam', stepSize: 0.002 },
      generator: { name: 'adam', stepSize: 0.001 },
      samples: 250,
      grid: 32,
      checkpoints: 10,
      seed: 42,
    })

    let last: GanRun | null = null
    for (const step of gen) {
      last = step
    }
    return last
  }, [dataChoice, game, criticSteps, steps])

  const run = runResult
  const checkpoints = run?.checkpoints ?? []
  const numCheckpoints = checkpoints.length
  const currentIdx = Math.min(checkpointIndex, Math.max(0, numCheckpoints - 1))
  const currentCheckpoint: GanCheckpoint | null = checkpoints[currentIdx] ?? null

  const box = run?.box ?? 3.5
  const gridAxis = useMemo(() => (run ? Array.from(run.gridX) : []), [run])

  // Real points
  const realPoints = useMemo(() => {
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

  // Generated fake samples at current checkpoint
  const fakePoints = useMemo(() => {
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

  // Discriminator / Critic surface on grid
  const fieldMatrix = useMemo(() => {
    if (!currentCheckpoint || !run) return []
    const flat = currentCheckpoint.field
    const G = 32
    const m: number[][] = []
    for (let i = 0; i < G; i++) {
      const row: number[] = []
      for (let j = 0; j < G; j++) {
        row.push(flat[i * G + j])
      }
      m.push(row)
    }
    return m
  }, [currentCheckpoint, run])

  const planeAxisX = useAxis({ label: 'x₁', range: [-box, box] })
  const planeAxisY = useAxis({ label: 'x₂', range: [-box, box] })

  const lossSteps = useMemo(() => {
    if (!run) return []
    return Array.from({ length: run.generatorLoss.length }, (_, i) => i)
  }, [run])

  const genLoss = useMemo(() => {
    if (!run) return []
    return Array.from(run.generatorLoss)
  }, [run])

  const critLoss = useMemo(() => {
    if (!run) return []
    return Array.from(run.criticLoss)
  }, [run])

  const lossX = useAxis({ label: 'Generator step', range: [0, Math.max(1, lossSteps.length)] })
  const lossY = useAxis({ label: 'Loss', range: [0, 4] })

  return (
    <Figure
      title="Generative Adversarial Network (GAN) Adversarial Dynamics"
      purpose="Explore Minimax, Non-saturating, and WGAN-GP training stability, mode collapse, and discriminator decision surface."
      caption={
        'Interactive Generative Adversarial Network training (Goodfellow et al., 2014; Arjovsky et al., 2017). Left: real data points (grey) and generator samples (coloured) over the discriminator surface D(x). Right: generator and critic adversarial loss curves across training. In minimax with a strong critic, vanishing gradients can stall the generator off data modes, while non-saturating and WGAN-GP losses maintain strong gradients toward modes.'
      }
    >
      <ControlRow label="Adversarial setup">
        <Select
          label="Target distribution"
          value={dataChoice}
          options={DATASETS}
          onChange={(v) => setDataChoice(v as any)}
        />
        <Select
          label="Adversarial objective"
          value={game}
          options={GAMES}
          onChange={(v) => setGame(v as AdversarialGame)}
        />
        <Select
          label="Critic steps / G step"
          value={String(criticSteps)}
          options={[
            { value: '1', label: '1 critic step' },
            { value: '2', label: '2 critic steps' },
            { value: '5', label: '5 critic steps' },
          ]}
          onChange={(v) => setCriticSteps(Number(v))}
        />
        <Select
          label="Training updates"
          value={String(steps)}
          options={[
            { value: '100', label: '100 updates' },
            { value: '200', label: '200 updates' },
            { value: '400', label: '400 updates' },
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
        <Plot x={planeAxisX} y={planeAxisY} title="Real data, generator samples & discriminator field">
          {fieldMatrix.length > 0 && gridAxis.length > 0 && (
            <Raster
              x={gridAxis}
              y={gridAxis}
              z={fieldMatrix}
              scale="diverging"
            />
          )}
          <Points
            x={realPoints.x}
            y={realPoints.y}
            slot={0}
            size={4}
            thin={true}
          />
          <Points
            x={fakePoints.x}
            y={fakePoints.y}
            slot={1}
            size={5}
          />
        </Plot>

        <Plot x={lossX} y={lossY} title="Adversarial training losses">
          <Curve
            x={lossSteps}
            y={genLoss}
            slot={0}
          />
          <Curve
            x={lossSteps}
            y={critLoss}
            slot={1}
          />
        </Plot>
      </Plots>

      <ControlRow label="Diagnostics">
        <Readout
          label="Current step"
          value={currentCheckpoint ? currentCheckpoint.step : '—'}
        />
        <Readout
          label="Modes covered"
          value={
            currentCheckpoint?.coverage
              ? `${currentCheckpoint.coverage.hit} / ${currentCheckpoint.coverage.modes}`
              : '—'
          }
        />
        <Readout
          label="Generator loss"
          value={
            genLoss.length > 0 && currentCheckpoint
              ? formatNumber(Number(genLoss[Math.min(genLoss.length - 1, currentCheckpoint.step)].toFixed(3)))
              : '—'
          }
        />
        <Readout
          label="Critic steps"
          value={criticSteps}
        />
      </ControlRow>
    </Figure>
  )
}
