import { useMemo, useState } from 'react'
import { ArrowDown, ArrowUp, RotateCcw } from 'lucide-react'
import {
  Button,
  choice,
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Points,
  Raster,
  Readout,
  Segments,
  useAxis,
  useFigureState,
} from 'aifn-render'
import {
  applyCoveringRules,
  sequentialCovering,
  type CoveringResult,
  coverMask,
  ruleText,
  thresholdConditions,
} from 'aifn-compute/learning/rules'
import { asGrid, boxSegments, moonRows, MOONS_X, MOONS_Y, predictionGrid } from '../_shared/rule-figures'

const NX = 61
const NY = 41
const GRID = predictionGrid(MOONS_X, MOONS_Y, NX, NY)

type Mode = 'list' | 'vote'
const MODES = [
  { value: 'list', label: 'decision list (first match)' },
  { value: 'vote', label: 'unordered set (covered counts vote)' },
] as const

/**
 * A decision list learned on two moons, its rules reorderable: the regions, the accuracy and the conflicts under
 * first-match and under voting.
 */
export function DecisionListOrder() {
  const fig = useFigureState({
    noise: float(0.2, { min: 0, max: 0.45, step: 0.01, label: 'moons noise' }),
    rules: int(6, { min: 1, max: 12, label: 'rules learned', suggestions: [3, 6, 9] }),
    seed: int(1, { min: 1, max: 999, label: 'data seed' }),
    mode: choice<Mode>(MODES, 'list', { label: 'how rules combine' }),
  })
  const { noise, rules: maxRules, seed, mode } = fig

  const train = useMemo(() => moonRows(`dl-${seed}`, 300, noise), [seed, noise])
  const test = useMemo(() => moonRows(`dl-test-${seed}`, 2000, noise), [seed, noise])
  const learned = useMemo(
    () =>
      sequentialCovering(train.rows, train.y, thresholdConditions(train.rows, 8), {
        measure: 'laplace',
        beam: 3,
        maxConditions: 2,
        minSupport: 5,
        maxRules,
      }),
    [train, maxRules],
  )

  // The order of the rules, kept with the learned list it belongs to so a new list starts in learned order.
  const key = `${seed}/${noise}/${maxRules}`
  const [order, setOrder] = useState<{ key: string; perm: number[]; selected: number }>({ key, perm: [], selected: 0 })
  const perm =
    order.key === key && order.perm.length === learned.rules.length ? order.perm : learned.rules.map((_, i) => i)
  const selected = order.key === key ? Math.min(order.selected, Math.max(0, perm.length - 1)) : 0
  const current: CoveringResult = useMemo(
    () => ({ ...learned, rules: perm.map((i) => learned.rules[i]) }),
    [learned, perm],
  )

  const evaluate = (rows: number[][], y: number[], r: CoveringResult) => {
    const p = applyCoveringRules(r, rows, mode)
    return p.reduce((a, v, i) => a + (v === y[i] ? 1 : 0), 0) / y.length
  }
  const field = useMemo(() => asGrid(applyCoveringRules(current, GRID.rows, mode), NX, NY), [current, mode])
  const stats = useMemo(() => {
    const masks = current.rules.map((r) => coverMask(train.rows, r.conditions))
    let conflicts = 0
    let uncovered = 0
    for (let i = 0; i < train.rows.length; i++) {
      const labels = new Set(current.rules.filter((_, k) => masks[k][i]).map((r) => r.label))
      if (labels.size === 0) uncovered++
      if (labels.size > 1) conflicts++
    }
    return {
      train: evaluate(train.rows, train.y, current),
      test: evaluate(test.rows, test.y, current),
      learnedTest: evaluate(test.rows, test.y, learned),
      conflicts: conflicts / train.rows.length,
      uncovered: uncovered / train.rows.length,
    }
  }, [current, learned, train, test, mode]) // eslint-disable-line react-hooks/exhaustive-deps

  const move = (k: number, by: number) => {
    const j = k + by
    if (j < 0 || j >= perm.length) return
    const next = perm.slice()
    ;[next[k], next[j]] = [next[j], next[k]]
    setOrder({ key, perm: next, selected: j })
  }
  const select = (p: [number, number]) => {
    const k = current.rules.findIndex((r) => coverMask([p], r.conditions)[0])
    if (k >= 0) setOrder({ key, perm, selected: k })
  }

  const xAxis = useAxis({ label: 'x₁', range: MOONS_X, nice: false })
  const yAxis = useAxis({ label: 'x₂', range: MOONS_Y, nice: false, equal: xAxis })
  const chosen = current.rules[selected]

  return (
    <Figure
      title="Reordering a decision list"
      state={fig}
      caption="A decision list learned by sequential covering on 300 two-moons points: each rule is a box of at most two thresholds, learned on the points the rules above it left uncovered, and the last line predicts the majority class of what is left. The shading is the predicted class; the ink box is the selected rule, the grey boxes the others. Click the plot to select the rule that fires there, and move it with the arrows. Moving a later rule up lets it capture points an earlier rule handled, and test accuracy (2000 fresh points) usually drops, because each rule was chosen for the points that reached it. As an unordered set, every covering rule adds the class counts it covered when it was learned (shown after each rule) and the largest total wins, so order no longer matters, but overlapping rules with different classes have to be reconciled by the vote."
      controls={
        <div className="flex w-full flex-col gap-1 text-sm">
          {current.rules.map((r, k) => (
            <div
              key={`${perm[k]}`}
              className={`flex items-center gap-2 rounded px-1 ${k === selected ? 'bg-muted' : ''}`}
            >
              <span className="w-6 text-right text-muted-foreground tabular-nums">{k + 1}.</span>
              <button
                type="button"
                className="flex-1 truncate text-left font-mono text-xs"
                onClick={() => setOrder({ key, perm, selected: k })}
              >
                {mode === 'list' ? (k === 0 ? 'if ' : 'else if ') : 'if '}
                {ruleText(r.conditions)} then class {r.label} ({r.counts.join(' : ')})
              </button>
              <Button
                size="icon"
                variant="ghost"
                aria-label="Move rule up"
                disabled={k === 0}
                onClick={() => move(k, -1)}
              >
                <ArrowUp />
              </Button>
              <Button
                size="icon"
                variant="ghost"
                aria-label="Move rule down"
                disabled={k === current.rules.length - 1}
                onClick={() => move(k, 1)}
              >
                <ArrowDown />
              </Button>
            </div>
          ))}
          <div className="flex items-center gap-2 px-1">
            <span className="w-6" />
            <span className="flex-1 font-mono text-xs">
              {mode === 'list' ? 'else' : 'no rule covers'} class {current.defaultLabel}
            </span>
            <Button
              size="sm"
              variant="outline"
              aria-label="Restore the learned order"
              onClick={() => setOrder({ key, perm: learned.rules.map((_, i) => i), selected: 0 })}
            >
              <RotateCcw /> learned order
            </Button>
          </div>
        </div>
      }
      readouts={
        <>
          <Readout label="training accuracy" value={formatNumber(stats.train)} />
          <Readout label="test accuracy" value={formatNumber(stats.test)} />
          <Readout label="test accuracy, learned order" value={formatNumber(stats.learnedTest)} />
          <Readout label="points covered by rules of both classes" value={formatNumber(stats.conflicts)} />
          <Readout label="points no rule covers" value={formatNumber(stats.uncovered)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} onPlotClick={select}>
        <Raster
          x={GRID.gx}
          y={GRID.gy}
          z={field}
          scale="categorical"
          categoryNames={['class 0', 'class 1']}
          fillOpacity={0.3}
          boundary
        />
        <Points
          name="training points"
          x={train.x1}
          y={train.x2}
          group={train.y}
          groupNames={['class 0', 'class 1']}
          thin
        />
        {current.rules.length > 1 && (
          <Segments
            name="other rules"
            segments={current.rules.flatMap((r, k) =>
              k === selected ? [] : boxSegments(r.conditions, MOONS_X, MOONS_Y),
            )}
            muted
            width={1}
          />
        )}
        {chosen && (
          <Segments
            name="selected rule"
            segments={boxSegments(chosen.conditions, MOONS_X, MOONS_Y)}
            emphasis
            width={2.5}
          />
        )}
      </Plot>
    </Figure>
  )
}
