import { useMemo, useState } from 'react'
import {
  ControlRow,
  Curve,
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

const fmt = (v: number, digits = 3) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(digits))) : '—')

const DATASET_OPTIONS = [
  { value: 'moons', label: 'Two interleaved moons (2 classes)' },
  { value: 'pinwheel', label: 'Pinwheel arms (3 classes)' },
]

const MODEL_OPTIONS = [
  { value: 'cvae', label: 'Conditional VAE (conditioned on label y)' },
  { value: 'vae', label: 'Unconditional VAE (latent must encode class)' },
]

export function CvaeExplorer() {
  const [dataChoice, setDataChoice] = useState<'moons' | 'pinwheel'>('moons')
  const [modelChoice, setModelChoice] = useState<'cvae' | 'vae'>('cvae')
  const [targetClass, setTargetClass] = useState<string>('all')
  const [checkpointIndex, setCheckpointIndex] = useState(15)

  // Generate synthetic dataset with classes
  const dataset = useMemo(() => {
    const s = stream('cvae-dataset')
    if (dataChoice === 'moons') {
      const data = moons(s, { n: 360, noise: 0.08 })
      return { x: data.x, y: data.y }
    } else {
      const data = pinwheel(s, { n: 360, arms: 3, noise: 0.05 })
      return { x: data.x, y: data.y }
    }
  }, [dataChoice])

  const numClasses = dataChoice === 'moons' ? 2 : 3

  // Train autoencoder
  const run: AutoencoderRun | null = useMemo(() => {
    const kind: AutoencoderKind = modelChoice === 'cvae' ? 'cvae' : 'vae'
    const gen = autoencoderRun(dataset, {
      kind,
      beta: 1.0,
      latent: 2,
      hidden: [32, 32],
      steps: 220,
      stepSize: 0.003,
      batchSize: 64,
      checkpoints: 16,
      shown: 150,
      samples: 150,
      seed: 42,
    })

    let last: AutoencoderRun | null = null
    for (const snapshot of gen) {
      last = snapshot
    }
    return last
  }, [dataset, modelChoice])

  const checkpoints = run?.checkpoints ?? []
  const numCheckpoints = checkpoints.length
  const currentIdx = Math.min(checkpointIndex, Math.max(0, numCheckpoints - 1))
  const currentCheckpoint: AutoencoderCheckpoint | null = checkpoints[currentIdx] ?? null

  // Real data points and their labels
  const { dataX, dataY, dataLabels } = useMemo(() => {
    if (!run) return { dataX: [], dataY: [], dataLabels: [] }
    const flat = run.data
    const labels = run.labels
    const n = Math.min(180, flat.length / 2)
    const xs: number[] = []
    const ys: number[] = []
    const ls: number[] = []
    for (let i = 0; i < n; i++) {
      xs.push(flat[2 * i])
      ys.push(flat[2 * i + 1])
      ls.push(labels[i])
    }
    return { dataX: xs, dataY: ys, dataLabels: ls }
  }, [run])

  // Latent codes
  const { latentX, latentY, latentLabels, classDist } = useMemo(() => {
    if (!currentCheckpoint || !run) return { latentX: [], latentY: [], latentLabels: [], classDist: 0 }
    const flat = currentCheckpoint.codes
    const labels = run.labels
    const n = Math.min(180, flat.length / 2)
    const xs: number[] = []
    const ys: number[] = []
    const ls: number[] = []

    const centroids: [number, number][] = Array.from({ length: numClasses }, () => [0, 0])
    const counts = new Array(numClasses).fill(0)

    for (let i = 0; i < n; i++) {
      const z1 = flat[2 * i]
      const z2 = flat[2 * i + 1]
      const c = labels[i]
      xs.push(z1)
      ys.push(z2)
      ls.push(c)
      if (c >= 0 && c < numClasses) {
        centroids[c][0] += z1
        centroids[c][1] += z2
        counts[c]++
      }
    }

    // Distance between class 0 and class 1 centroids
    let dist = 0
    if (counts[0] > 0 && counts[1] > 0) {
      const c0: [number, number] = [centroids[0][0] / counts[0], centroids[0][1] / counts[0]]
      const c1: [number, number] = [centroids[1][0] / counts[1], centroids[1][1] / counts[1]]
      dist = Math.sqrt((c0[0] - c1[0]) ** 2 + (c0[1] - c1[1]) ** 2)
    }

    return { latentX: xs, latentY: ys, latentLabels: ls, classDist: dist }
  }, [currentCheckpoint, run, numClasses])

  // Generated samples
  const { sampleX, sampleY, sampleClass } = useMemo(() => {
    if (!currentCheckpoint) return { sampleX: [], sampleY: [], sampleClass: [] }
    const flat = currentCheckpoint.samples
    const labels = currentCheckpoint.sampleLabels
    const n = flat.length / 2
    const xs: number[] = []
    const ys: number[] = []
    const ls: number[] = []

    const filter = targetClass === 'all' ? -1 : Number(targetClass)

    for (let i = 0; i < n; i++) {
      const c = labels[i]
      if (filter === -1 || c === filter) {
        xs.push(flat[2 * i])
        ys.push(flat[2 * i + 1])
        ls.push(c)
      }
    }
    return { sampleX: xs, sampleY: ys, sampleClass: ls }
  }, [currentCheckpoint, targetClass])

  // Filtered real data by target class
  const filteredData = useMemo(() => {
    const filter = targetClass === 'all' ? -1 : Number(targetClass)
    const xs: number[] = []
    const ys: number[] = []
    const groups: number[] = []
    for (let i = 0; i < dataX.length; i++) {
      if (filter === -1 || dataLabels[i] === filter) {
        xs.push(dataX[i])
        ys.push(dataY[i])
        groups.push(dataLabels[i])
      }
    }
    return { xs, ys, groups }
  }, [dataX, dataY, dataLabels, targetClass])

  const filteredLatent = useMemo(() => {
    const filter = targetClass === 'all' ? -1 : Number(targetClass)
    const xs: number[] = []
    const ys: number[] = []
    const groups: number[] = []
    for (let i = 0; i < latentX.length; i++) {
      if (filter === -1 || latentLabels[i] === filter) {
        xs.push(latentX[i])
        ys.push(latentY[i])
        groups.push(latentLabels[i])
      }
    }
    return { xs, ys, groups }
  }, [latentX, latentY, latentLabels, targetClass])

  const lossSteps = run?.history.step ?? []
  const reconLoss = run?.history.reconstruction ?? []
  const klLoss = run?.history.regulariser ?? []

  const lossIdx = currentCheckpoint
    ? lossSteps.findIndex((s) => s >= currentCheckpoint.step)
    : -1
  const curRecon = lossIdx >= 0 ? reconLoss[lossIdx] : reconLoss[reconLoss.length - 1]
  const curKl = lossIdx >= 0 ? klLoss[lossIdx] : klLoss[klLoss.length - 1]

  // Axes
  const dataXAxis = useAxis({ label: 'feature x₁', range: [-2.6, 2.6] })
  const dataYAxis = useAxis({ label: 'feature x₂', range: [-2.6, 2.6] })
  const latentXAxis = useAxis({ label: 'latent coordinate z₁', range: [-3.2, 3.2] })
  const latentYAxis = useAxis({ label: 'latent coordinate z₂', range: [-3.2, 3.2] })
  const lossXAxis = useAxis({ label: 'training step', range: [0, 220] })
  const lossYAxis = useAxis({ label: 'loss (nats)', range: [0, 6] })

  const classNames = dataChoice === 'moons' ? ['Class 0 (top moon)', 'Class 1 (bottom moon)'] : ['Class 0', 'Class 1', 'Class 2']

  const CLASS_OPTIONS = [
    { value: 'all', label: 'All classes' },
    ...classNames.map((name, i) => ({ value: String(i), label: name })),
  ]

  return (
    <Figure
      title="Conditional VAE vs unconditional VAE: latent disentanglement"
      purpose="Contrasts how conditioning on side information y frees the latent space z ~ N(0, I) to capture style and continuous variation rather than categorical identity, enabling controlled one-to-many generation."
      controls={
        <>
          <ControlRow label="Model & task">
            <Select
              label="Architecture"
              value={modelChoice}
              onChange={(v) => setModelChoice(v as 'cvae' | 'vae')}
              options={MODEL_OPTIONS}
            />
            <Select
              label="Synthetic dataset"
              value={dataChoice}
              onChange={(v) => {
                setDataChoice(v as 'moons' | 'pinwheel')
                setTargetClass('all')
              }}
              options={DATASET_OPTIONS}
            />
            <Select
              label="Conditional generation filter"
              value={targetClass}
              onChange={setTargetClass}
              options={CLASS_OPTIONS}
            />
          </ControlRow>
          <ControlRow label="Training checkpoint">
            <Player
              className="col-span-full"
              value={currentIdx}
              onChange={setCheckpointIndex}
              count={Math.max(1, numCheckpoints)}
              label="training progress"
              format={(idx) => `step ${checkpoints[idx]?.step ?? 0}`}
            />
          </ControlRow>
        </>
      }
      readouts={{
        'loss values': (
          <>
            <Readout label="reconstruction loss" value={fmt(curRecon)} />
            <Readout label="KL divergence" value={fmt(curKl)} />
            <Readout label="total conditional ELBO" value={fmt(-(curRecon + curKl))} />
          </>
        ),
        'latent space alignment': (
          <>
            <Readout
              label="centroid separation (class 0 vs 1)"
              value={`${fmt(classDist, 3)} units`}
            />
            <Readout
              label="latent structure"
              value={
                modelChoice === 'cvae'
                  ? 'shared prior N(0, I) for all classes'
                  : 'partitioned into separate clusters'
              }
            />
          </>
        ),
      }}
      caption={`Comparison of conditional and unconditional variational autoencoders. Left: data space showing real data points (small circles) and generated prior samples (larger points) for the selected class condition y. Middle: latent representation q(z | x, y). In the CVAE, the encoder receives the class label y alongside x, so the latent space z encodes only intra-class variation (such as position along the manifold); both classes overlap centrally around (0, 0). In the standard VAE, z must encode class identity, forcing the classes to separate into disconnected latent clusters. Right: training loss curves showing reconstruction error and KL divergence.`}
    >
      <Plots cols={3}>
        <Plot x={dataXAxis} y={dataYAxis} title="data space & conditional generation">
          <Points
            name="real data x"
            x={filteredData.xs}
            y={filteredData.ys}
            group={filteredData.groups}
            groupNames={classNames}
            size={3}
            muted
          />
          <Points
            name="generated samples x ~ p(x | z, y)"
            x={sampleX}
            y={sampleY}
            group={sampleClass}
            groupNames={classNames}
            size={6}
            emphasis
          />
        </Plot>
        <Plot x={latentXAxis} y={latentYAxis} title="latent space q(z | x, y)">
          <Points
            name="latent codes z"
            x={filteredLatent.xs}
            y={filteredLatent.ys}
            group={filteredLatent.groups}
            groupNames={classNames}
            size={5}
          />
          <Curve
            name="prior 2σ unit contour"
            x={Float64Array.from({ length: 60 }, (_, i) => 2 * Math.cos((2 * Math.PI * i) / 59))}
            y={Float64Array.from({ length: 60 }, (_, i) => 2 * Math.sin((2 * Math.PI * i) / 59))}
            muted
            dashed
            thin
          />
        </Plot>
        <Plot x={lossXAxis} y={lossYAxis} title="training loss curves">
          <Curve name="reconstruction loss" x={lossSteps} y={reconLoss} emphasis />
          <Curve name="KL divergence" x={lossSteps} y={klLoss} slot={1} />
        </Plot>
      </Plots>
    </Figure>
  )
}
