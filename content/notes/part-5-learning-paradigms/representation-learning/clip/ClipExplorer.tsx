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
  Segments,
  Select,
  formatNumber,
  useAxis,
} from 'aifn-render'
import { stream } from 'aifn/foundation/random'
import { slice, toFlat } from 'aifn/foundation/tensor'
import { pairedShapes } from 'aifn-applied/data/synthetic'
import {
  contrastiveTrainingRun,
  embed,
  TwoTower,
  type ContrastiveSnapshot,
  type TemperatureSetting,
} from 'aifn-applied/neural/contrastive'

const TEMPERATURE_MODES = [
  { value: 'learned', label: 'Learned temperature (initial τ = 0.1)' },
  { value: '0.05', label: 'Fixed low τ = 0.05 (sharp logits)' },
  { value: '0.2', label: 'Fixed medium τ = 0.2' },
  { value: '1.0', label: 'Fixed high τ = 1.0 (soft logits)' },
]

export function ClipExplorer() {
  const [tempChoice, setTempChoice] = useState<string>('learned')
  const [steps, setSteps] = useState(200)
  const [checkpointIndex, setCheckpointIndex] = useState(0)

  // Generate synthetic multi-modal pairs and train tiny CLIP
  const runResult = useMemo(() => {
    const sTrain = stream('clip-train-data')
    const sTest = stream('clip-test-data')

    const train = pairedShapes(sTrain, { n: 300, pixelNoise: 0.05 })
    const test = pairedShapes(sTest, { n: 120, pixelNoise: 0.05 })

    const tempSetting: TemperatureSetting =
      tempChoice === 'learned' ? 'learned' : Number(tempChoice)

    const gen = contrastiveTrainingRun(train, test, {
      steps,
      every: 20,
      temperature: tempSetting,
      dim: 2,
      hidden: 32,
      batchSize: 32,
      stepSize: 0.01,
      seed: 'clip-explorer',
    })

    let last: ContrastiveSnapshot | null = null
    for (const snap of gen) {
      last = snap
    }
    return { last, train, test }
  }, [tempChoice, steps])

  const { last: snapshot, test } = runResult
  const checkpoints = snapshot?.checkpoints ?? []
  const numCheckpoints = checkpoints.length
  const currentIdx = Math.min(checkpointIndex, Math.max(0, numCheckpoints - 1))
  const currentCheckpoint = checkpoints[currentIdx]

  // Model instance
  const model = useMemo(() => {
    if (!test) return null
    return TwoTower({
      inA: test.a.shape[1],
      inB: test.b.shape[1],
      hidden: 32,
      dim: 2,
    })
  }, [test])

  // Embeddings at current checkpoint for a test subset (10 pairs)
  const subsetData = useMemo(() => {
    if (!model || !currentCheckpoint || !test) {
      return {
        imageX: [],
        imageY: [],
        textX: [],
        textY: [],
        segments: [],
        similarityMatrix: [],
      }
    }

    const B = 10
    const subA = slice(test.a, [0, 0], [B, test.a.shape[1]])
    const subB = slice(test.b, [0, 0], [B, test.b.shape[1]])

    const za = embed(model, currentCheckpoint.params, subA, 'a')
    const zb = embed(model, currentCheckpoint.params, subB, 'b')

    const zaArr = Float64Array.from(toFlat(za))
    const zbArr = Float64Array.from(toFlat(zb))

    const imgX: number[] = []
    const imgY: number[] = []
    const txtX: number[] = []
    const txtY: number[] = []
    const segs: { from: [number, number]; to: [number, number] }[] = []

    for (let i = 0; i < B; i++) {
      const ix = zaArr[i * 2]
      const iy = zaArr[i * 2 + 1]
      const tx = zbArr[i * 2]
      const ty = zbArr[i * 2 + 1]

      imgX.push(ix)
      imgY.push(iy)
      txtX.push(tx)
      txtY.push(ty)

      segs.push({
        from: [ix, iy],
        to: [tx, ty],
      })
    }

    // Similarity matrix S_ij = u_i · v_j
    const simMat: number[][] = []
    for (let i = 0; i < B; i++) {
      const row: number[] = []
      for (let j = 0; j < B; j++) {
        const dot = zaArr[i * 2] * zbArr[j * 2] + zaArr[i * 2 + 1] * zbArr[j * 2 + 1]
        row.push(dot)
      }
      simMat.push(row)
    }

    return {
      imageX: imgX,
      imageY: imgY,
      textX: txtX,
      textY: txtY,
      segments: segs,
      similarityMatrix: simMat,
    }
  }, [model, currentCheckpoint, test])

  const circleX = useAxis({ label: 'Embedding dimension 1', range: [-1.4, 1.4] })
  const circleY = useAxis({ label: 'Embedding dimension 2', range: [-1.4, 1.4] })

  const matrixAxisX = useAxis({ label: 'Text view j', range: [0, 10] })
  const matrixAxisY = useAxis({ label: 'Image view i', range: [0, 10] })

  const matrixGrid = useMemo(() => Array.from({ length: 10 }, (_, i) => i + 0.5), [])

  const circleRing = useMemo(() => {
    const pts = 80
    const xs: number[] = []
    const ys: number[] = []
    for (let i = 0; i <= pts; i++) {
      const th = (2 * Math.PI * i) / pts
      xs.push(Math.cos(th))
      ys.push(Math.sin(th))
    }
    return { x: xs, y: ys }
  }, [])

  return (
    <Figure
      title="Contrastive Language-Image Pretraining (CLIP) Dual-Encoder"
      purpose="Visualise multi-modal representation alignment on the unit circle S¹ and cross-modal batch similarity matrix under symmetric InfoNCE."
      caption={
        'Interactive CLIP dual-encoder training (Radford et al., 2021). Left: image representations f(x) (dots) and caption representations g(y) (crosses) projected onto the shared unit circle S¹, with connecting lines linking positive pairs. Right: the batch cosine similarity matrix S_ij = u_iᵀ v_j. Under symmetric InfoNCE, positive pairs pull together, repulsive forces spread negatives uniformly, and the diagonal lights up.'
      }
    >
      <ControlRow label="Training settings">
        <Select
          label="Temperature parameter"
          value={tempChoice}
          options={TEMPERATURE_MODES}
          onChange={setTempChoice}
        />
        <Select
          label="Training steps"
          value={String(steps)}
          options={[
            { value: '100', label: '100 steps' },
            { value: '200', label: '200 steps' },
            { value: '400', label: '400 steps' },
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
        <Plot x={circleX} y={circleY} title="Multimodal alignment circle S¹ (lines = paired views)">
          <Curve
            x={circleRing.x}
            y={circleRing.y}
            slot={0}
            thin={true}
          />
          <Segments
            segments={subsetData.segments}
            slot={0}
          />
          <Points
            x={subsetData.imageX}
            y={subsetData.imageY}
            slot={1}
            size={5}
          />
          <Points
            x={subsetData.textX}
            y={subsetData.textY}
            slot={2}
            size={4}
          />
        </Plot>

        <Plot x={matrixAxisX} y={matrixAxisY} title="Batch similarity matrix (diagonal = positive pairs)">
          {subsetData.similarityMatrix.length > 0 && (
            <Raster
              x={matrixGrid}
              y={matrixGrid}
              z={subsetData.similarityMatrix}
              scale="diverging"
            />
          )}
        </Plot>
      </Plots>

      <ControlRow label="Diagnostics">
        <Readout
          label="Step"
          value={currentCheckpoint ? currentCheckpoint.step : '—'}
        />
        <Readout
          label="Zero-shot accuracy"
          value={
            currentCheckpoint
              ? `${formatNumber(Number((currentCheckpoint.zeroShotSeen * 100).toFixed(1)))}%`
              : '—'
          }
        />
        <Readout
          label="Top-1 retrieval"
          value={
            currentCheckpoint
              ? `${formatNumber(Number((currentCheckpoint.retrievalTop1 * 100).toFixed(1)))}%`
              : '—'
          }
        />
        <Readout
          label="Temperature τ"
          value={
            currentCheckpoint
              ? formatNumber(Number(currentCheckpoint.temperature.toFixed(3)))
              : '—'
          }
        />
      </ControlRow>
    </Figure>
  )
}
