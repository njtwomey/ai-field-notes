import { useMemo, useState } from 'react'
import {
  Bars,
  ControlRow,
  Curve,
  Figure,
  Handle,
  Plot,
  Plots,
  Points,
  Raster,
  Readout,
  Select,
  Slider,
  formatNumber,
  useAxis,
} from 'aifn-render'

const fmt = (v: number, digits = 3) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(digits))) : '—')

type DatasetKind = 'clusters' | 'moons' | 'ring'

const DATASET_OPTIONS = [
  { value: 'clusters', label: 'Four separated Gaussian clusters' },
  { value: 'ring', label: 'Continuous circular manifold' },
  { value: 'moons', label: 'Interleaved two moons' },
]

const K_OPTIONS = [
  { value: '4', label: 'K = 4 codes' },
  { value: '8', label: 'K = 8 codes' },
  { value: '12', label: 'K = 12 codes' },
]

/** Seeded PRNG for reproducible synthetic data */
function pseudoRandom(seed: number) {
  let s = seed % 2147483647
  if (s <= 0) s += 2147483646
  return () => {
    s = (s * 16807) % 2147483647
    return (s - 1) / 2147483646
  }
}

function generatePoints(kind: DatasetKind, n: number): [number, number][] {
  const rand = pseudoRandom(42)
  const pts: [number, number][] = []

  if (kind === 'clusters') {
    const centers: [number, number][] = [
      [-1.1, -1.1],
      [1.1, -1.1],
      [-1.1, 1.1],
      [1.1, 1.1],
    ]
    for (let i = 0; i < n; i++) {
      const c = centers[i % centers.length]
      const u1 = Math.max(1e-6, rand())
      const u2 = rand()
      const r = 0.28 * Math.sqrt(-2 * Math.log(u1))
      const theta = 2 * Math.PI * u2
      pts.push([c[0] + r * Math.cos(theta), c[1] + r * Math.sin(theta)])
    }
  } else if (kind === 'ring') {
    for (let i = 0; i < n; i++) {
      const theta = (2 * Math.PI * i) / n + (rand() - 0.5) * 0.1
      const rad = 1.35 + (rand() - 0.5) * 0.25
      pts.push([rad * Math.cos(theta), rad * Math.sin(theta)])
    }
  } else {
    // Moons
    for (let i = 0; i < n; i++) {
      const top = i < n / 2
      const theta = Math.PI * rand()
      const rad = 1.1 + (rand() - 0.5) * 0.2
      if (top) {
        pts.push([rad * Math.cos(theta) - 0.45, rad * Math.sin(theta) + 0.2])
      } else {
        pts.push([-rad * Math.cos(theta) + 0.45, -rad * Math.sin(theta) - 0.2])
      }
    }
  }
  return pts
}

export function VqVaeExplorer() {
  const [kCodes, setKCodes] = useState(8)
  const [beta, setBeta] = useState(0.25)
  const [datasetKind, setDatasetKind] = useState<DatasetKind>('clusters')
  const [trainStep, setTrainStep] = useState(15)
  const [focal, setFocal] = useState<[number, number]>([0.8, 0.4])
  const [deadCodeReset, setDeadCodeReset] = useState(true)

  const numPoints = 160
  const dataPoints = useMemo(() => generatePoints(datasetKind, numPoints), [datasetKind])

  // Simulated codebook evolution: k-means-like adaptation with learning rate, commitment, and optional reset
  const { codebook, usages, focalQuant } = useMemo(() => {
    const K = kCodes
    const rand = pseudoRandom(123)

    // Initial codebook randomly scattered in a tight box [-0.5, 0.5]
    let cb: [number, number][] = Array.from({ length: K }, () => [(rand() - 0.5) * 1.2, (rand() - 0.5) * 1.2])

    const lr = 0.22

    // Evolve over trainStep iterations
    for (let step = 0; step < trainStep; step++) {
      const counts = new Array(K).fill(0)
      const sums: [number, number][] = Array.from({ length: K }, () => [0, 0])

      for (const [x, y] of dataPoints) {
        let bestIdx = 0
        let bestD = Infinity
        for (let j = 0; j < K; j++) {
          const d = (x - cb[j][0]) ** 2 + (y - cb[j][1]) ** 2
          if (d < bestD) {
            bestD = d
            bestIdx = j
          }
        }
        counts[bestIdx]++
        sums[bestIdx][0] += x
        sums[bestIdx][1] += y
      }

      // Update codebook entries toward their assigned clusters
      cb = cb.map(([cx, cy], j) => {
        if (counts[j] > 0) {
          const targetX = sums[j][0] / counts[j]
          const targetY = sums[j][1] / counts[j]
          return [cx + lr * (targetX - cx), cy + lr * (targetY - cy)]
        } else if (deadCodeReset && step % 4 === 0) {
          // Reset dead code to a random data sample
          const pick = dataPoints[Math.floor(rand() * dataPoints.length)]
          return [pick[0] + (rand() - 0.5) * 0.1, pick[1] + (rand() - 0.5) * 0.1]
        }
        return [cx, cy]
      })
    }

    // Final assignment frequencies
    const counts = new Array(K).fill(0)
    for (const [x, y] of dataPoints) {
      let bestIdx = 0
      let bestD = Infinity
      for (let j = 0; j < K; j++) {
        const d = (x - cb[j][0]) ** 2 + (y - cb[j][1]) ** 2
        if (d < bestD) {
          bestD = d
          bestIdx = j
        }
      }
      counts[bestIdx]++
    }

    // Focal point lookup
    let focalIdx = 0
    let focalDist = Infinity
    for (let j = 0; j < K; j++) {
      const d = (focal[0] - cb[j][0]) ** 2 + (focal[1] - cb[j][1]) ** 2
      if (d < focalDist) {
        focalDist = d
        focalIdx = j
      }
    }

    return {
      codebook: cb,
      usages: counts.map((c) => c / numPoints),
      focalQuant: {
        idx: focalIdx,
        entry: cb[focalIdx],
        distSq: focalDist,
      },
    }
  }, [kCodes, trainStep, dataPoints, deadCodeReset, focal])

  // Voronoi cell raster for continuous 2D space
  const GRID_SIZE = 35
  const gridCoords = useMemo(() => {
    const coords: number[] = []
    for (let i = 0; i < GRID_SIZE; i++) coords.push(-2.4 + (4.8 * i) / (GRID_SIZE - 1))
    return coords
  }, [])

  const voronoiRaster = useMemo(() => {
    const z: number[][] = []
    for (let r = 0; r < GRID_SIZE; r++) {
      const y = gridCoords[r]
      const row: number[] = []
      for (let c = 0; c < GRID_SIZE; c++) {
        const x = gridCoords[c]
        let bestIdx = 0
        let bestD = Infinity
        for (let j = 0; j < codebook.length; j++) {
          const d = (x - codebook[j][0]) ** 2 + (y - codebook[j][1]) ** 2
          if (d < bestD) {
            bestD = d
            bestIdx = j
          }
        }
        row.push(bestIdx)
      }
      z.push(row)
    }
    return z
  }, [gridCoords, codebook])

  // Perplexity of codebook usage
  const perplexity = useMemo(() => {
    let entropy = 0
    for (const p of usages) {
      if (p > 1e-9) entropy -= p * Math.log(p)
    }
    return Math.exp(entropy)
  }, [usages])

  const activeCodes = usages.filter((u) => u > 0.01).length

  // Losses at focal point
  const codebookLoss = focalQuant.distSq
  const commitmentLoss = beta * focalQuant.distSq

  // Straight-through gradient: simulated downstream reconstruction gradient pointing left-up
  const gradRecon: [number, number] = [-0.35, 0.4]
  // Codebook update gradient: pulls e toward z_e
  const gradCodebook: [number, number] = [focal[0] - focalQuant.entry[0], focal[1] - focalQuant.entry[1]]
  // Commitment gradient: pulls z_e toward e
  const gradCommitment: [number, number] = [
    beta * (focalQuant.entry[0] - focal[0]),
    beta * (focalQuant.entry[1] - focal[1]),
  ]

  // Axes
  const latentXAxis = useAxis({ label: 'latent coordinate z_1', range: [-2.4, 2.4] })
  const latentYAxis = useAxis({ label: 'latent coordinate z_2', range: [-2.4, 2.4] })

  const barXAxis = useAxis({ label: 'codebook entry index k', range: [0.5, kCodes + 0.5] })
  const barYAxis = useAxis({ label: 'assignment proportion p_k', range: [0, 0.6] })

  const codeIndices = useMemo(() => Array.from({ length: kCodes }, (_, i) => i + 1), [kCodes])
  const codeX = useMemo(() => Float64Array.from(codebook, (c) => c[0]), [codebook])
  const codeY = useMemo(() => Float64Array.from(codebook, (c) => c[1]), [codebook])
  const dataX = useMemo(() => Float64Array.from(dataPoints, (p) => p[0]), [dataPoints])
  const dataY = useMemo(() => Float64Array.from(dataPoints, (p) => p[1]), [dataPoints])

  return (
    <Figure
      title="VQ-VAE: vector quantisation, straight-through estimator, and codebook learning"
      purpose="Visualizes how the VQ-VAE partitions continuous latent space into discrete Voronoi cells, demonstrating how the straight-through estimator copies reconstruction gradients across the non-differentiable argmin while codebook and commitment losses align code entries with data clusters."
      controls={
        <>
          <ControlRow label="Latent configuration">
            <Select
              label="Codebook size K"
              value={String(kCodes)}
              onChange={(v) => setKCodes(Number(v))}
              options={K_OPTIONS}
            />
            <Select
              label="Encoder distribution"
              value={datasetKind}
              onChange={(v) => setDatasetKind(v as DatasetKind)}
              options={DATASET_OPTIONS}
            />
            <Select
              label="Dead code replacement"
              value={deadCodeReset ? 'yes' : 'no'}
              onChange={(v) => setDeadCodeReset(v === 'yes')}
              options={[
                { value: 'yes', label: 'Periodic reset of unused codes' },
                { value: 'no', label: 'No reset (risk of codebook collapse)' },
              ]}
            />
          </ControlRow>
          <ControlRow label="Training & loss weights">
            <Slider
              label="Codebook training iterations"
              value={trainStep}
              onChange={setTrainStep}
              min={0}
              max={30}
              step={1}
            />
            <Slider label="Commitment weight β" value={beta} onChange={setBeta} min={0.05} max={1.0} step={0.05} />
          </ControlRow>
        </>
      }
      readouts={{
        'quantisation at focal z_e': (
          <>
            <Readout label="assigned entry e_{k*}" value={`#${focalQuant.idx + 1}`} />
            <Readout label="quantisation error ||z_e − z_q||²" value={fmt(focalQuant.distSq, 4)} />
            <Readout label="codebook loss ||sg[z_e] − e||²" value={fmt(codebookLoss, 4)} />
            <Readout label="commitment loss β||z_e − sg[e]||²" value={fmt(commitmentLoss, 4)} />
          </>
        ),
        'codebook utilisation': (
          <>
            <Readout label="active codes (>1%)" value={`${activeCodes} of ${kCodes}`} />
            <Readout label="codebook perplexity exp(H)" value={`${fmt(perplexity, 2)} / ${kCodes}`} />
            <Readout
              label="collapse status"
              value={activeCodes < kCodes / 2 ? 'partial collapse (dead codes)' : 'healthy utilisation'}
            />
          </>
        ),
      }}
      caption={`Interactive VQ-VAE latent space. Left: 2D encoder space partitioned into Voronoi cells corresponding to codebook vectors e_k (diamonds). Encoder outputs z_e(x) (dots) are assigned to their nearest codebook vector z_q = e_{k*}. The orange handle denotes a focal encoder output; the dashed line shows quantisation error. The green vector represents the straight-through gradient copied directly to the encoder, while blue and amber vectors denote the codebook update (pulling e toward z_e) and commitment update (pulling z_e toward e). Drag the focal point across cell boundaries to inspect instant index switching. Right: Assignment histogram across the K codebook entries. Without code resets, isolated initial codes receive zero gradients and permanently collapse.`}
    >
      <Plots cols={2}>
        <Plot x={latentXAxis} y={latentYAxis} title="latent partition & straight-through gradients">
          <Raster
            x={gridCoords}
            y={gridCoords}
            z={voronoiRaster}
            scale="categorical"
            fillOpacity={0.16}
            colorBar={false}
          />
          <Points name="encoder outputs z_e" x={dataX} y={dataY} size={3} muted />
          <Points name="codebook vectors e_k" x={codeX} y={codeY} size={9} emphasis />
          {/* Quantisation line from focal z_e to assigned e_{k*} */}
          <Curve
            name="quantisation link"
            x={[focal[0], focalQuant.entry[0]]}
            y={[focal[1], focalQuant.entry[1]]}
            dashed
          />
          {/* Downstream straight-through gradient from z_e */}
          <Curve
            name="straight-through gradient ∇ L_recon"
            x={[focal[0], focal[0] + gradRecon[0]]}
            y={[focal[1], focal[1] + gradRecon[1]]}
            emphasis
          />
          {/* Commitment gradient on z_e */}
          <Curve
            name="commitment pull β(e − z_e)"
            x={[focal[0], focal[0] + gradCommitment[0]]}
            y={[focal[1], focal[1] + gradCommitment[1]]}
            thin
          />
          {/* Codebook loss pull on e_{k*} */}
          <Curve
            name="codebook pull (z_e − e)"
            x={[focalQuant.entry[0], focalQuant.entry[0] + gradCodebook[0] * 0.4]}
            y={[focalQuant.entry[1], focalQuant.entry[1] + gradCodebook[1] * 0.4]}
            thin
          />
          <Handle
            kind="point"
            at={focal}
            onDrag={([x, y]) => setFocal([Math.max(-2.2, Math.min(2.2, x)), Math.max(-2.2, Math.min(2.2, y))])}
            label={`z_e: (${fmt(focal[0], 2)}, ${fmt(focal[1], 2)}) → code #${focalQuant.idx + 1}`}
          />
        </Plot>
        <Plot x={barXAxis} y={barYAxis} title="codebook utilization & frequency p_k">
          <Curve
            name="uniform allocation (1/K)"
            x={[0.5, kCodes + 0.5]}
            y={[1 / kCodes, 1 / kCodes]}
            muted
            dashed
            thin
          />
          <Bars name="assigned frequency" x={codeIndices} y={usages} opacity={0.65} />
        </Plot>
      </Plots>
    </Figure>
  )
}
