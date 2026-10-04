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
import { stream } from 'aifn/foundation/random'
import { implicitFeedback } from 'aifn-methods/data/synthetic'
import {
  recommenderRun,
  type RecommenderCheckpoint,
  type RecommenderKind,
  type RecommenderSnapshot,
} from 'aifn-methods/retrieval/recommenders'

const MODELS: { value: RecommenderKind; label: string }[] = [
  { value: 'bpr', label: 'BPR Matrix Factorisation (Bayesian pairwise ranking)' },
  { value: 'two-tower', label: 'Two-Tower Dual Encoder (MLP embeddings)' },
  { value: 'neural-collaborative-filtering', label: 'Neural Collaborative Filtering (NCF)' },
  { value: 'item-knn', label: 'Item-Item Collaborative Filtering (Cosine kNN)' },
  { value: 'popularity', label: 'Popularity Baseline (unpersonalised)' },
]

export function RecommendersExplorer() {
  const [modelKind, setModelKind] = useState<RecommenderKind>('bpr')
  const [factors, setFactors] = useState(8)
  const [epochs, setEpochs] = useState(30)
  const [checkpointIndex, setCheckpointIndex] = useState(0)

  // Synthetic implicit feedback interactions
  const data = useMemo(() => {
    const s = stream('recsys-benchmark')
    return implicitFeedback(s, {
      users: 60,
      items: 80,
    })
  }, [])

  // Run recommender training
  const runSnapshot: RecommenderSnapshot | null = useMemo(() => {
    const gen = recommenderRun({
      data,
      model: modelKind,
      dimension: factors,
      epochs,
      stepSize: 0.02,
      seed: 42,
    })

    let last: RecommenderSnapshot | null = null
    for (const snap of gen) {
      last = snap
    }
    return last
  }, [data, modelKind, factors, epochs])

  const checkpoints = runSnapshot?.checkpoints ?? []
  const numCheckpoints = checkpoints.length
  const currentIdx = Math.min(checkpointIndex, Math.max(0, numCheckpoints - 1))
  const currentCheckpoint: RecommenderCheckpoint | null = checkpoints[currentIdx] ?? null

  // User and item 2D embeddings (from PCA projection)
  const userCoords = useMemo(() => {
    if (!currentCheckpoint || !currentCheckpoint.userMap) {
      return { x: [], y: [] }
    }
    const map = currentCheckpoint.userMap
    const uN = map.length / 2
    const uX: number[] = []
    const uY: number[] = []
    for (let i = 0; i < uN; i++) {
      uX.push(map[2 * i])
      uY.push(map[2 * i + 1])
    }
    return { x: uX, y: uY }
  }, [currentCheckpoint])

  const itemCoords = useMemo(() => {
    if (!currentCheckpoint || !currentCheckpoint.itemMap) {
      return { x: [], y: [] }
    }
    const map = currentCheckpoint.itemMap
    const iN = map.length / 2
    const iX: number[] = []
    const iY: number[] = []
    for (let i = 0; i < iN; i++) {
      iX.push(map[2 * i])
      iY.push(map[2 * i + 1])
    }
    return { x: iX, y: iY }
  }, [currentCheckpoint])

  const mapAxisX = useAxis({ label: 'Latent PC 1', range: [-2.5, 2.5] })
  const mapAxisY = useAxis({ label: 'Latent PC 2', range: [-2.5, 2.5] })

  // Performance progression curves
  const recallCurve = runSnapshot?.history.recall ?? []
  const ndcgCurve = runSnapshot?.history.ndcg ?? []
  const epochSteps = runSnapshot?.history.epoch ?? []

  const metricAxisX = useAxis({ label: 'Epoch', range: [0, epochs] })
  const metricAxisY = useAxis({ label: 'Ranking quality', range: [0, 0.6] })

  const curRecall = recallCurve[currentIdx]
  const curNdcg = ndcgCurve[currentIdx]

  return (
    <Figure
      title="Collaborative Filtering & Recommender Embeddings"
      purpose="Explore latent matrix factorisation, two-tower dual encoders, and ranking evaluation metrics (Recall@10, NDCG@10)."
      caption={
        'Interactive Recommender Systems (Rendle et al., 2009; Koren et al., 2009). Left: joint 2D PCA projection of user representations (gold dots) and item representations (blue dots). Right: evaluation curves showing Recall@10 and NDCG@10 on held-out user interactions. As Bayesian Personalised Ranking (BPR) optimises pairwise margins, users and their preferred items pull together in latent factor space.'
      }
    >
      <ControlRow label="Algorithm & hyper-parameters">
        <Select
          label="Model architecture"
          value={modelKind}
          options={MODELS}
          onChange={(v) => setModelKind(v as RecommenderKind)}
        />
        <Select
          label="Latent factors"
          value={String(factors)}
          options={[
            { value: '4', label: '4 factors' },
            { value: '8', label: '8 factors' },
            { value: '16', label: '16 factors' },
          ]}
          onChange={(v) => setFactors(Number(v))}
        />
        <Select
          label="Training epochs"
          value={String(epochs)}
          options={[
            { value: '15', label: '15 epochs' },
            { value: '30', label: '30 epochs' },
            { value: '60', label: '60 epochs' },
          ]}
          onChange={(v) => setEpochs(Number(v))}
        />
      </ControlRow>

      <ControlRow label="Epoch progression">
        <Player count={Math.max(1, numCheckpoints)} value={currentIdx} onChange={setCheckpointIndex} />
      </ControlRow>

      <Plots>
        <Plot x={mapAxisX} y={mapAxisY} title="Joint PCA embedding space (users: gold, items: blue)">
          <Points x={userCoords.x} y={userCoords.y} slot={1} size={5} />
          <Points x={itemCoords.x} y={itemCoords.y} slot={0} size={4} />
        </Plot>

        <Plot x={metricAxisX} y={metricAxisY} title="Held-out ranking performance">
          <Curve x={epochSteps} y={recallCurve} slot={0} />
          <Curve x={epochSteps} y={ndcgCurve} slot={1} />
        </Plot>
      </Plots>

      <ControlRow label="Diagnostics">
        <Readout label="Epoch" value={currentCheckpoint ? currentCheckpoint.epoch : '—'} />
        <Readout label="Recall@10" value={curRecall !== undefined ? formatNumber(Number(curRecall.toFixed(3))) : '—'} />
        <Readout label="NDCG@10" value={curNdcg !== undefined ? formatNumber(Number(curNdcg.toFixed(3))) : '—'} />
        <Readout label="Catalogue" value="60 users, 80 items" />
      </ControlRow>
    </Figure>
  )
}
