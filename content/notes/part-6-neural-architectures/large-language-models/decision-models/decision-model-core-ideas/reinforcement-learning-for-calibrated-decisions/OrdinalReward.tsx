import { useMemo } from 'react'
import {
  Bars,
  Figure,
  float,
  formatNumber,
  Handle,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'

const L = 5
const LEVELS = [1, 2, 3, 4, 5]

/** The model puts probability c on level a and spreads the rest evenly over the other levels. */
function prediction(a: number, c: number): number[] {
  return LEVELS.map((l) => (l === a ? c : (1 - c) / (L - 1)))
}

/** Penalties (lower is better) for a prediction q when the true level is y. */
function penalties(q: number[], y: number) {
  const log = -Math.log(q[y - 1])
  const brier = q.reduce((s, p, i) => s + (p - (i + 1 === y ? 1 : 0)) ** 2, 0)
  let F = 0
  let rps = 0
  for (let l = 1; l < L; l++) {
    F += q[l - 1]
    rps += (F - (y <= l ? 1 : 0)) ** 2
  }
  return { log, brier, rps }
}

export function OrdinalReward() {
  const state = useFigureState({
    y: slider(1, 5, 2, { step: 1, label: 'true rating y' }),
    a: slider(1, 5, 4, { step: 1, onChart: true }),
    c: float(0.7, {
      min: 0.21,
      max: 0.99,
      step: 0.01,
      label: 'probability on that level',
      suggestions: [0.5, 0.7, 0.9],
    }),
  })
  const { y, a, c } = state
  const q = useMemo(() => prediction(a, c), [a, c])
  const now = penalties(q, y)

  // Each penalty for every level the model could favour, relative to its worst case, so the shapes can be compared.
  const sweep = useMemo(() => {
    const rows = LEVELS.map((l) => penalties(prediction(l, c), y))
    const rel = (k: 'log' | 'brier' | 'rps') => {
      const m = Math.max(...rows.map((r) => r[k]))
      return rows.map((r) => r[k] / m)
    }
    return { log: rel('log'), brier: rel('brier'), rps: rel('rps') }
  }, [c, y])

  const qx = useAxis({ label: 'rating', range: [0.5, 5.5], integer: true })
  const qy = useAxis({ label: 'predicted probability', range: [0, 1] })
  const sx = useAxis({ label: 'level the model favours', range: [0.5, 5.5], integer: true })
  const sy = useAxis({ label: 'penalty, relative to the worst level', range: [0, 1.05] })
  return (
    <Figure
      title="Order-blind and order-aware rewards on a rating scale"
      state={state}
      caption="Left: a prediction on a 1 to 5 scale that favours one level (drag it) and spreads the rest evenly; the dot marks the true rating. Right: each penalty for every level the model could favour, scaled to its worst case. Log loss and the Brier score charge every wrong level the same, so a near miss earns no credit. The ranked probability score, the sum of Brier scores over the four questions 'is the rating above ℓ?', grows with the distance from the truth."
      readouts={
        <>
          <Readout label="log loss" value={formatNumber(now.log)} />
          <Readout label="Brier" value={formatNumber(now.brier)} />
          <Readout label="ranked probability score" value={formatNumber(now.rps)} />
          <Readout label="expected level" value={formatNumber(q.reduce((s, p, i) => s + p * (i + 1), 0))} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={qx} y={qy} height={260}>
          <Bars name="prediction" x={LEVELS} y={q} slot={0} />
          <Points name="true rating" x={[y]} y={[q[y - 1]]} emphasis size={10} />
          <Handle {...state.handle('a', { label: 'favoured level' })} />
        </Plot>
        <Plot x={sx} y={sy} height={260}>
          <Bars name="log loss" x={LEVELS.map((l) => l - 0.27)} y={sweep.log} width={0.25} slot={1} />
          <Bars name="Brier" x={LEVELS} y={sweep.brier} width={0.25} slot={2} />
          <Bars name="ranked probability score" x={LEVELS.map((l) => l + 0.27)} y={sweep.rps} width={0.25} slot={0} />
        </Plot>
      </div>
    </Figure>
  )
}
