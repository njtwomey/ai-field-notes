import { useMemo, useState } from 'react'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, unwrap, type Tensor } from 'aifn-compute/foundation/tensor'
import { modularArithmetic, type ModularOperation } from 'aifn-methods/data/synthetic'
import { grokkingRun, ModularMlp, type GrokkingSnapshot } from 'aifn-methods/neural/grokking'
import {
  ControlGroup,
  Curve,
  Figure,
  NumberSelector,
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

const OPERATIONS = [
  { value: '+', label: 'Addition (a + b mod p)' },
  { value: '-', label: 'Subtraction (a − b mod p)' },
  { value: '*', label: 'Multiplication (a × b mod p)' },
  { value: '/', label: 'Division (a ÷ b mod p)' },
]

/** Coordinates of every residue's embedding in the plane of one frequency's cosine and sine components. */
function circleOf(embedding: Tensor, basis: Tensor, p: number, k: number): { x: number[]; y: number[] } {
  const E = toFlat(embedding)
  const B = toFlat(basis)
  const d = embedding.shape[1]
  const coef = (col: number) => {
    const v = new Float64Array(d)
    for (let a = 0; a < p; a++) {
      for (let j = 0; j < d; j++) {
        v[j] += B[a * p + col] * E[a * d + j]
      }
    }
    return v
  }
  const c = coef(2 * k - 1)
  const s = 2 * k < p ? coef(2 * k) : new Float64Array(d)
  const dot = (u: Float64Array, v: Float64Array) => u.reduce((acc, x, j) => acc + x * v[j], 0)
  const nc = Math.sqrt(dot(c, c)) || 1
  const u1 = c.map((x) => x / nc)
  const r = s.map((x, j) => x - dot(s, u1) * u1[j])
  const nr = Math.sqrt(dot(r, r)) || 1
  const u2 = r.map((x) => x / nr)
  const x: number[] = []
  const y: number[] = []
  for (let a = 0; a < p; a++) {
    const row = Float64Array.from({ length: d }, (_, j) => E[a * d + j])
    x.push(dot(row, u1))
    y.push(dot(row, u2))
  }
  return { x, y }
}

export function GrokkingExplorer() {
  const [op, setOp] = useState<ModularOperation>('+')
  const [p, setP] = useState(31)
  const [fraction, setFraction] = useState(0.5)
  const [weightDecay, setWeightDecay] = useState(2)
  const [stepIdx, setStepIdx] = useState(0)

  // Generate synthetic modular arithmetic dataset
  const data = useMemo(() => {
    return modularArithmetic(stream(`grokking-data-${p}-${op}`), {
      p,
      op,
      fraction,
    })
  }, [p, op, fraction])

  // Run training generator and collect checkpoints
  const run = useMemo(() => {
    const steps = 1200
    const gen = grokkingRun(data, {
      width: 128,
      embed: 32,
      steps,
      stepSize: 0.01,
      weightDecay,
      method: 'adamw',
      recordEvery: 10,
      checkpointEvery: 40,
      seed: 42,
    })
    const snapshots: GrokkingSnapshot[] = []
    for (const snap of gen) {
      snapshots.push(snap)
    }
    return snapshots
  }, [data, weightDecay])

  const currentSnap = run[Math.min(stepIdx, run.length - 1)] ?? run[run.length - 1]
  const curves = currentSnap.curves
  const latestCheckpoint = currentSnap.checkpoints[currentSnap.checkpoints.length - 1]
  const currentParams = latestCheckpoint?.params

  // Fourier circular projection of learned residue embeddings
  const circle = useMemo(() => {
    if (!currentParams) return { x: [], y: [] }
    const basis = data.truth.fourierBasis
    return circleOf(currentParams.embedding, basis, p, 1)
  }, [currentParams, data.truth.fourierBasis, p])

  // Full table evaluation
  const table = useMemo(() => {
    if (!currentParams) return { x: [], y: [], z: [[]] }
    const allA = new Int32Array(p * p)
    const allB = new Int32Array(p * p)
    for (let a = 0; a < p; a++) {
      for (let b = 0; b < p; b++) {
        allA[a * p + b] = a
        allB[a * p + b] = b
      }
    }
    const allPairs = { a: fromData(allA, [p * p]), b: fromData(allB, [p * p]) }
    const model = ModularMlp(currentSnap.config)
    const logits = unwrap(model.apply(currentParams, allPairs)) as Tensor
    const flatLogits = toFlat(logits)
    const decided = new Array(p * p).fill(0)
    for (let i = 0; i < p * p; i++) {
      let maxK = 0
      let maxVal = -Infinity
      for (let k = 0; k < p; k++) {
        const val = flatLogits[i * p + k]
        if (val > maxVal) {
          maxVal = val
          maxK = k
        }
      }
      decided[i] = maxK
    }
    const z: number[][] = []
    for (let r = 0; r < p; r++) {
      z.push(decided.slice(r * p, (r + 1) * p))
    }
    const coords = Array.from({ length: p }, (_, i) => i)
    return { x: coords, y: coords, z }
  }, [currentParams, currentSnap.config, p])

  const stepAxis = useAxis({ label: 'optimization step', hold: 'initial', key: run })
  const accAxis = useAxis({ label: 'accuracy', range: [0, 1.05] })
  const circleX = useAxis({ label: 'Fourier cosine component', nice: false })
  const circleY = useAxis({ label: 'Fourier sine component', nice: false, equal: circleX })
  const tableA = useAxis({ label: 'operand a', range: [0, p - 1] })
  const tableB = useAxis({ label: 'operand b', range: [0, p - 1], equal: tableA })
  const trainAcc = curves.trainAccuracy[curves.trainAccuracy.length - 1] ?? 0
  const testAcc = curves.testAccuracy[curves.testAccuracy.length - 1] ?? 0
  const trainLoss = curves.trainLoss[curves.trainLoss.length - 1] ?? 0
  const testLoss = curves.testLoss[curves.testLoss.length - 1] ?? 0
  const normVal = curves.weightNorm[curves.weightNorm.length - 1] ?? 0

  return (
    <Figure
      title="Grokking: Generalization Beyond Overfitting on Modular Arithmetic"
      purpose="Under weight decay, neural networks fitting modular arithmetic first memorize the training set (training accuracy hits 100%), before suddenly discovering the periodic Fourier solution thousands of steps later."
      defaultSize="L"
      controls={
        <>
          <ControlGroup title="Task Parameters">
            <Select
              label="Modular Operation"
              value={op}
              onChange={(v) => {
                setOp(v as ModularOperation)
                setStepIdx(0)
              }}
              options={OPERATIONS}
            />
            <NumberSelector
              label="Modulus p (prime)"
              value={p}
              onChange={(v) => {
                setP(v)
                setStepIdx(0)
              }}
              min={17}
              max={43}
              step={2}
              suggestions={[23, 31, 37, 43]}
            />
            <NumberSelector
              label="Training Fraction"
              value={fraction}
              onChange={(v) => {
                setFraction(v)
                setStepIdx(0)
              }}
              min={0.3}
              max={0.8}
              step={0.05}
              spacing="lin"
              points={0.05}
            />
            <NumberSelector
              label="Weight Decay λ"
              value={weightDecay}
              onChange={(v) => {
                setWeightDecay(v)
                setStepIdx(0)
              }}
              min={0}
              max={5}
              step={0.5}
              suggestions={[0, 0.5, 1, 2, 4]}
            />
          </ControlGroup>
          <ControlGroup title="Optimization Step Playback">
            <Player
              value={Math.min(stepIdx, run.length - 1)}
              onChange={setStepIdx}
              count={run.length}
              label="Checkpoints"
            />
          </ControlGroup>
        </>
      }
      readouts={{
        Accuracy: (
          <>
            <Readout label="step" value={`${currentSnap.step} / ${currentSnap.steps}`} />
            <Readout label="train accuracy" value={formatNumber(trainAcc)} />
            <Readout label="test accuracy" value={formatNumber(testAcc)} />
          </>
        ),
        'Loss & Regularization': (
          <>
            <Readout label="train loss" value={formatNumber(trainLoss)} />
            <Readout label="test loss" value={formatNumber(testLoss)} />
            <Readout label="‖θ‖₂ weight norm" value={formatNumber(normVal)} />
          </>
        ),
      }}
      caption="Top: Learning curves over optimization steps. Train accuracy (blue) leaps to 100% within the first 100 steps while validation accuracy (orange) stays near chance (1/p). Under continuous AdamW weight decay, the parameter norm gradually compresses, triggering a sudden transition to 100% test accuracy. Bottom left: 2D projection of residue embeddings onto the primary Fourier circle. Bottom right: Complete modular arithmetic table reconstructed by the network."
    >
      <Plots rows={2} heights={[1.2, 1]}>
        <Plot x={stepAxis} y={accAxis}>
          <Curve name="train accuracy" x={curves.steps} y={curves.trainAccuracy} slot={0} />
          <Curve name="test accuracy (generalization)" x={curves.steps} y={curves.testAccuracy} slot={1} emphasis />
        </Plot>
        <Plots cols={2}>
          <Plot x={circleX} y={circleY}>
            <Points name="residue embeddings" x={circle.x} y={circle.y} slot={2} emphasis />
          </Plot>
          <Plot x={tableA} y={tableB}>
            <Raster
              x={table.x}
              y={table.y}
              z={table.z}
              scale="diverging"
              range={[0, p - 1]}
              valueLabel="predicted answer"
            />
          </Plot>
        </Plots>
      </Plots>
    </Figure>
  )
}
