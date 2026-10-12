import { useMemo } from 'react'
import { choice, Figure, int, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { normal, stream } from 'aifn-compute/foundation/random'

const SAMPLE_SIZE = 120

export function FeatureCrossesExplorer() {
  const state = useFigureState({
    modelType: choice<'linear_raw' | 'binned_cross' | 'hashed_cross'>(
      [
        { value: 'linear_raw', label: 'linear model without crosses (additive only)' },
        { value: 'binned_cross', label: 'binned feature cross (K × K grid bins)' },
        { value: 'hashed_cross', label: 'hashed feature cross (B hash buckets)' },
      ],
      'linear_raw',
      { label: 'feature representation' },
    ),
    numBins: slider(2, 6, 4, { step: 1, label: 'bins per dimension K' }),
    hashBuckets: int(8, { min: 4, max: 32, label: 'hash table size B' }),
  })

  const results = useMemo(() => {
    const rng = stream(7)
    const pointsX1: number[] = []
    const pointsX2: number[] = []
    const trueLabels: number[] = []

    // 120 points on a 2x2 XOR checkerboard:
    // If sign(X1) * sign(X2) > 0 => class 1 (red), else class 0 (blue)
    for (let i = 0; i < SAMPLE_SIZE; i++) {
      const quad = i % 4
      let cx = 1
      let cy = 1
      let label = 1
      if (quad === 1) {
        cx = -1
        cy = 1
        label = 0
      } else if (quad === 2) {
        cx = -1
        cy = -1
        label = 1
      } else if (quad === 3) {
        cx = 1
        cy = -1
        label = 0
      }

      const x1 = cx * 1.1 + normal(rng, 0, 0.35)
      const x2 = cy * 1.1 + normal(rng, 0, 0.35)
      pointsX1.push(x1)
      pointsX2.push(x2)
      trueLabels.push(label)
    }

    const K = state.numBins
    const B = state.hashBuckets
    let correct = 0
    let paramCount = 2
    let collisionCount = 0

    const predClasses: number[] = []

    if (state.modelType === 'linear_raw') {
      // Pure additive linear model: w0 + w1*X1 + w2*X2
      // For XOR, best linear hyperplane achieves 50% accuracy!
      paramCount = 2
      for (let i = 0; i < SAMPLE_SIZE; i++) {
        // Linear separator cannot separate XOR: predicts majority or diagonal
        const pred = pointsX1[i] + pointsX2[i] > 0 ? 1 : 0
        predClasses.push(pred)
        if (pred === trueLabels[i]) correct++
      }
    } else if (state.modelType === 'binned_cross') {
      // Binned cross: discretize X1 and X2 into K intervals in [-2.5, 2.5]
      // Full grid has K * K parameters
      paramCount = K * K

      // Learn cell weights by majority vote in each (b1, b2) cell
      const cellVotes: { [key: string]: { pos: number; neg: number } } = {}
      for (let i = 0; i < SAMPLE_SIZE; i++) {
        const b1 = Math.min(K - 1, Math.max(0, Math.floor(((pointsX1[i] + 2.5) / 5.0) * K)))
        const b2 = Math.min(K - 1, Math.max(0, Math.floor(((pointsX2[i] + 2.5) / 5.0) * K)))
        const key = `${b1}_${b2}`
        if (!cellVotes[key]) cellVotes[key] = { pos: 0, neg: 0 }
        if (trueLabels[i] === 1) cellVotes[key].pos++
        else cellVotes[key].neg++
      }

      for (let i = 0; i < SAMPLE_SIZE; i++) {
        const b1 = Math.min(K - 1, Math.max(0, Math.floor(((pointsX1[i] + 2.5) / 5.0) * K)))
        const b2 = Math.min(K - 1, Math.max(0, Math.floor(((pointsX2[i] + 2.5) / 5.0) * K)))
        const key = `${b1}_${b2}`
        const v = cellVotes[key] || { pos: 0, neg: 0 }
        const pred = v.pos >= v.neg ? 1 : 0
        predClasses.push(pred)
        if (pred === trueLabels[i]) correct++
      }
    } else {
      // Hashed feature cross: hash (b1, b2) into B buckets
      paramCount = B
      const bucketOccupants: { [b: number]: Set<string> } = {}
      const bucketVotes: { [b: number]: { pos: number; neg: number } } = {}

      for (let i = 0; i < SAMPLE_SIZE; i++) {
        const b1 = Math.min(K - 1, Math.max(0, Math.floor(((pointsX1[i] + 2.5) / 5.0) * K)))
        const b2 = Math.min(K - 1, Math.max(0, Math.floor(((pointsX2[i] + 2.5) / 5.0) * K)))
        const key = `${b1}_${b2}`
        // Simple hash function
        const hash = Math.abs((b1 * 31 + b2 * 17 + 7) % B)
        if (!bucketOccupants[hash]) bucketOccupants[hash] = new Set()
        bucketOccupants[hash].add(key)

        if (!bucketVotes[hash]) bucketVotes[hash] = { pos: 0, neg: 0 }
        if (trueLabels[i] === 1) bucketVotes[hash].pos++
        else bucketVotes[hash].neg++
      }

      // Count collisions (buckets with >1 active cell key)
      for (let b = 0; b < B; b++) {
        if (bucketOccupants[b] && bucketOccupants[b].size > 1) {
          collisionCount += bucketOccupants[b].size - 1
        }
      }

      for (let i = 0; i < SAMPLE_SIZE; i++) {
        const b1 = Math.min(K - 1, Math.max(0, Math.floor(((pointsX1[i] + 2.5) / 5.0) * K)))
        const b2 = Math.min(K - 1, Math.max(0, Math.floor(((pointsX2[i] + 2.5) / 5.0) * K)))
        const hash = Math.abs((b1 * 31 + b2 * 17 + 7) % B)
        const v = bucketVotes[hash] || { pos: 0, neg: 0 }
        const pred = v.pos >= v.neg ? 1 : 0
        predClasses.push(pred)
        if (pred === trueLabels[i]) correct++
      }
    }

    const accuracy = correct / SAMPLE_SIZE

    // Separate into true positive, false, etc. for visualizer
    const class0X: number[] = []
    const class0Y: number[] = []
    const class1X: number[] = []
    const class1Y: number[] = []

    for (let i = 0; i < SAMPLE_SIZE; i++) {
      if (trueLabels[i] === 0) {
        class0X.push(pointsX1[i])
        class0Y.push(pointsX2[i])
      } else {
        class1X.push(pointsX1[i])
        class1Y.push(pointsX2[i])
      }
    }

    return {
      class0X,
      class0Y,
      class1X,
      class1Y,
      accuracy,
      paramCount,
      collisionCount,
    }
  }, [state.modelType, state.numBins, state.hashBuckets])

  const xAxis = useAxis({ label: 'feature X₁', range: [-2.5, 2.5] })
  const yAxis = useAxis({ label: 'feature X₂', range: [-2.5, 2.5] })

  return (
    <Figure
      title="Non-linear decision boundaries via binned and hashed feature crosses"
      state={state}
      hoverReadout={false}
      caption="Evaluating linear models on a 2D non-linear XOR checkerboard. A standard linear model without crosses can only learn a planar hyperplane, capping accuracy at 50%. Discretizing continuous coordinates and constructing Cartesian crosses (bin(X₁) × bin(X₂)) creates K² synthetic terms, allowing a linear model to learn independent cell weights and achieve 100% accuracy. Hashing crosses into B fixed buckets bounds memory at scale, trading mild collision noise for constant storage."
      readouts={
        <>
          <Readout label="model accuracy" value={`${(results.accuracy * 100).toFixed(1)}%`} />
          <Readout label="feature parameters" value={`${results.paramCount} weights`} />
          <Readout
            label="hash collisions"
            value={state.modelType === 'hashed_cross' ? `${results.collisionCount} collisions` : 'None (exact)'}
          />
          <Readout
            label="expressivity"
            value={results.accuracy > 0.85 ? 'Non-linear Separable' : 'Linearly Underfitting'}
          />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} title="XOR feature space and observation classes">
        <Points
          name="ground truth class 0 (blue quadrants)"
          x={results.class0X}
          y={results.class0Y}
          slot={0}
          size={5}
        />
        <Points
          name="ground truth class 1 (orange quadrants)"
          x={results.class1X}
          y={results.class1Y}
          slot={1}
          size={5}
        />
      </Plot>
    </Figure>
  )
}
