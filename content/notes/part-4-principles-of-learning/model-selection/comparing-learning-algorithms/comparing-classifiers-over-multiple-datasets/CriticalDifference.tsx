import { useMemo } from 'react'
import { Figure, float, formatNumber, int, Plot, Points, Readout, Segments, useAxis, useFigureState } from 'aifn-render'
import { child, normals, stream, uniform } from 'aifn-compute/foundation/random'
import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { friedmanTest, nemenyiCriticalDifference, wilcoxonSignedRank } from 'aifn-compute/probability/tests'

const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—')
const NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

/**
 * Friedman's test and the Nemenyi critical difference on simulated accuracies of k algorithms over N data sets, each
 * algorithm drawn at its average rank with an interval of half the critical difference.
 */
export function CriticalDifference() {
  const state = useFigureState({
    N: int(20, { ge: 3, le: 100, suggestions: [10, 20, 50], label: 'data sets N' }),
    k: int(5, { ge: 2, le: 8, suggestions: [3, 5, 8], label: 'algorithms k' }),
    spread: float(0.03, { ge: 0, le: 0.2, suggestions: [0, 0.03, 0.08], label: 'true spread of accuracy' }),
    noise: float(0.03, { ge: 0.001, le: 0.2, suggestions: [0.01, 0.03, 0.06], label: 'noise per score' }),
    alpha: float(0.05, { ge: 0.01, le: 0.2, suggestions: [0.05, 0.1], label: 'level α' }),
    seed: int(2, { ge: 1, le: 999, label: 'draw' }),
  })
  const { N, k, spread, noise, alpha, seed } = state
  const view = useMemo(() => {
    const s = stream(`comparing/critical-difference/${seed}`)
    // Each data set has its own difficulty; algorithm j adds a fixed offset; each score has its own noise.
    const base = toFlat(uniform(child(s, 'base'), 0.6, 0.95, { shape: [N] }) as Tensor)
    const eps = toFlat(normals(child(s, 'noise'), [N, k]))
    const offsets = NAMES.slice(0, k).map((_, j) => (k === 1 ? 0 : (spread * j) / (k - 1)))
    const scores = Array.from({ length: N }, (_, i) => offsets.map((o, j) => base[i] + o + noise * eps[i * k + j]))
    const friedman = friedmanTest(scores)
    const ranks = Array.from(toFlat(friedman.averageRanks))
    const cd = nemenyiCriticalDifference(k, N, alpha)
    const order = ranks.map((_, j) => j).sort((a, b) => ranks[a] - ranks[b])
    const [best, second] = order
    const wilcoxon = wilcoxonSignedRank(
      scores.map((r) => r[best]),
      scores.map((r) => r[second]),
    )
    const labels = order.map((j) => `${NAMES[j]} +${(offsets[j] * 100).toFixed(1)}`)
    const position = order.map((_, row) => row)
    const differsFromBest = order.map((j) => ranks[j] - ranks[best] >= cd)
    const interval = (row: number) => ({
      from: [ranks[order[row]] - cd / 2, row] as const,
      to: [ranks[order[row]] + cd / 2, row] as const,
    })
    return {
      friedman,
      cd,
      labels,
      rankOf: order.map((j) => ranks[j]),
      position,
      same: position.filter((r) => !differsFromBest[r]).map(interval),
      differ: position.filter((r) => differsFromBest[r]).map(interval),
      wilcoxon,
      pair: `${NAMES[best]} vs ${NAMES[second]}`,
    }
  }, [N, k, spread, noise, alpha, seed])
  const x = useAxis({ label: 'average rank (1 = best)', range: [0.5, k + 0.5], nice: false })
  const y = useAxis({ categories: view.labels, inverse: true })
  return (
    <Figure
      title="Average ranks and the critical difference"
      state={state}
      caption={`Simulated accuracies of k algorithms on N data sets. Each data set has its own base accuracy, drawn between 0.60 and 0.95; algorithm j adds a fixed true offset, spread evenly from 0 to the chosen spread, and every score gets independent Gaussian noise. Within each data set the algorithms are ranked, 1 for the highest accuracy, and the dots are the average ranks. Each bar has half the Nemenyi critical difference on either side, so two algorithms differ significantly exactly when their bars do not overlap. Bars in the second colour differ from the best-ranked algorithm. Each label gives the algorithm's true offset in points of accuracy, the ground truth the ranks try to recover.`}
      readouts={{
        Friedman: (
          <>
            <Readout label="χ²_F" value={fmt(view.friedman.statistic)} />
            <Readout label="p" value={fmt(view.friedman.pValue)} />
          </>
        ),
        'Iman–Davenport': (
          <>
            <Readout label="F_F" value={fmt(view.friedman.imanDavenport.statistic)} />
            <Readout label="p" value={fmt(view.friedman.imanDavenport.pValue)} />
          </>
        ),
        Nemenyi: <Readout label="critical difference" value={fmt(view.cd)} />,
        [`Wilcoxon, ${view.pair}`]: <Readout label="p" value={fmt(view.wilcoxon.pValue)} />,
      }}
    >
      <Plot x={x} y={y}>
        {view.same.length > 0 && (
          <Segments name="not different from the best" segments={view.same} slot={0} width={6} />
        )}
        {view.differ.length > 0 && <Segments name="worse than the best" segments={view.differ} slot={1} width={6} />}
        <Points name="average rank" x={view.rankOf} y={view.position} emphasis size={7} />
      </Plot>
    </Figure>
  )
}
