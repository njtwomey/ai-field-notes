import { useMemo, useState } from 'react'
import {
  Bars,
  Button,
  choice,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Points,
  Readout,
  useAxis,
  useFigureState,
  type Vector,
  Vectors,
} from 'aifn-render'
import { LOSSES, ndcg, rankingLoss, ranks, type LossId } from './ranking-losses'

type Preset = 'graded' | 'one' | 'two'

const PRESETS: Record<Preset, { label: string; rel: number[]; scores: number[] }> = {
  graded: { label: 'graded labels 0–3', rel: [2, 0, 3, 1, 0, 1], scores: [0.4, 1.3, -0.3, 0.8, -1.0, 0.1] },
  one: { label: 'one click', rel: [0, 0, 1, 0, 0, 0], scores: [0.4, 1.3, -0.3, 0.8, -1.0, 0.1] },
  two: { label: 'two clicks', rel: [1, 0, 1, 0, 0, 0], scores: [0.4, 1.3, -0.3, 0.8, -1.0, 0.1] },
}

const ITEMS = [0, 1, 2, 3, 4, 5]
const RANKS = ITEMS.map((r) => r + 1)
const LIMIT = 3
const X_RANGE: [number, number] = [-0.5, 5.5]
const Y_RANGE: [number, number] = [-LIMIT, LIMIT]
const RANK_RANGE: [number, number] = [0.5, 6.5]
const clamp = (v: number) => Math.max(-LIMIT, Math.min(LIMIT, v))

/**
 * One list of six items with draggable scores. For the chosen loss, arrows show −∂L/∂s on every item (the direction a
 * gradient step moves it), scaled so the longest is one unit; the lower panel shows the same push by rank position.
 * Used across the loss-family notes, each opening on its own loss.
 */
export function RankingLossLab({
  initial = 'ranknet',
  losses = Object.keys(LOSSES) as LossId[],
  preset: initialPreset = 'graded',
}: {
  initial?: LossId
  losses?: LossId[]
  preset?: Preset
}) {
  const state = useFigureState({
    loss: choice<LossId>(
      losses.map((id) => ({ value: id, label: LOSSES[id].label })),
      initial,
      { label: 'loss' },
    ),
    preset: choice<Preset>(
      (Object.keys(PRESETS) as Preset[]).map((p) => ({ value: p, label: PRESETS[p].label })),
      initialPreset,
      { label: 'labels' },
    ),
  })
  const preset = state.preset
  // Scores dragged by hand, stored with the labels they belong to: new labels start from that preset's scores.
  const [edited, setEdited] = useState<{ preset: Preset; scores: number[] } | null>(null)
  const scores = edited?.preset === preset ? edited.scores : PRESETS[preset].scores
  const rel = PRESETS[preset].rel

  const value = useMemo(() => rankingLoss(state.loss, scores, rel), [state.loss, scores, rel])
  const pos = useMemo(() => ranks(scores), [scores])

  const groupNames = useMemo(
    () => ['relevance 0', 'relevance 1', 'relevance 2', 'relevance 3'].slice(0, Math.max(...rel) + 1),
    [rel],
  )

  const arrows = useMemo((): Vector[] => {
    const biggest = Math.max(...value.grad.map(Math.abs))
    if (biggest < 1e-9) return []
    return ITEMS.filter((i) => Math.abs(value.grad[i]) > 1e-3 * biggest).map((i) => ({
      from: [i, scores[i]],
      to: [i, scores[i] - value.grad[i] / biggest],
    }))
  }, [value, scores])

  const push = useMemo(() => {
    const out = Array<number>(ITEMS.length)
    ITEMS.forEach((i) => (out[pos[i]] = -value.grad[i]))
    return out
  }, [value, pos])

  const handles: Handle[] = ITEMS.map((i) => ({
    kind: 'point',
    at: [i, scores[i]],
    label: `item ${i + 1}`,
    onDrag: ([, y]) => setEdited({ preset, scores: scores.map((v, j) => (j === i ? clamp(y) : v)) }),
  }))

  const order = [...ITEMS].sort((a, b) => pos[a] - pos[b]).map((i) => `${i + 1}`)
  const total = value.grad.reduce((a, g) => a + Math.abs(g), 0)
  const topShare =
    total > 0 ? ITEMS.filter((i) => pos[i] < 2).reduce((a, i) => a + Math.abs(value.grad[i]), 0) / total : 0

  const xAxis = useAxis({ label: 'item', range: X_RANGE, integer: true })
  const yAxis = useAxis({ label: 'score s', range: Y_RANGE })
  const rankAxis = useAxis({ label: 'rank position (1 = top)', range: RANK_RANGE, integer: true })
  const pushAxis = useAxis({ label: 'push −∂L/∂s', hold: 'union' })
  return (
    <Figure
      title="Where each ranking loss pushes the scores"
      state={state}
      caption="Six items for one user or query, with relevance labels shown by marker. Drag any item's score up or down. The arrows show −∂L/∂s for the chosen loss, the direction one gradient step moves each score, scaled so the longest arrow is one unit. The lower panel shows the same push by rank position, top first. Pointwise losses push every item towards its own label, wherever it is ranked. Pairwise losses push each misordered pair apart. Listwise losses and LambdaRank concentrate the push near the top of the list, where the ranking metrics are decided."
      controls={
        <Button variant="outline" size="sm" onClick={() => setEdited(null)}>
          Reset scores
        </Button>
      }
      readouts={
        <>
          <Readout label="loss" value={formatNumber(value.loss)} />
          <Readout label="NDCG of this order" value={formatNumber(ndcg(scores, rel))} />
          <Readout label="ranking, top first" value={order.join(' › ')} />
          <Readout label="share of push on the top two" value={`${Math.round(topShare * 100)}%`} />
        </>
      }
    >
      <div className="space-y-4">
        <Plot x={xAxis} y={yAxis} height={300}>
          <Points name="items" x={ITEMS} y={scores} group={rel} groupNames={groupNames} />
          <Vectors vectors={arrows} />
          {handles.map((h, i) => (
            <Handle key={i} {...h} />
          ))}
        </Plot>
        <Plot x={rankAxis} y={pushAxis} height={200}>
          <Bars name="−∂L/∂s at each rank" x={RANKS} y={push} slot={2} />
        </Plot>
      </div>
    </Figure>
  )
}
