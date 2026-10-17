import { useMemo, useState } from 'react'
import {
  choice,
  Figure,
  float,
  formatNumber,
  int,
  Player,
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
  type CoveringStep,
  ruleText,
  thresholdConditions,
  type RuleMeasure,
} from 'aifn-compute/learning/rules'
import { asGrid, boxSegments, moonRows, MOONS_X, MOONS_Y, predictionGrid } from '../_shared/rule-figures'

const NX = 61
const NY = 41
const GRID = predictionGrid(MOONS_X, MOONS_Y, NX, NY)
const MEASURES = [
  { value: 'laplace', label: 'Laplace estimate' },
  { value: 'precision', label: 'precision' },
  { value: 'm-estimate', label: 'm-estimate (m = 10)' },
  { value: 'wracc', label: 'weighted relative accuracy' },
] as const
type Measure = (typeof MEASURES)[number]['value']

const KIND: Record<CoveringStep['kind'], string> = {
  grow: 'grow: best rule at this depth',
  accept: 'accept the best rule found',
  remove: 'remove the rows it covers',
  stop: 'stop',
}

/** Sequential covering on two moons, one search depth, acceptance or removal per step. */
export function CoveringStepper() {
  const fig = useFigureState({
    measure: choice<Measure>(MEASURES, 'laplace', { label: 'quality measure' }),
    beam: int(3, { min: 1, max: 20, label: 'beam width', suggestions: [1, 3, 10] }),
    depth: int(2, { min: 1, max: 4, label: 'conditions per rule' }),
    noise: float(0.2, { min: 0, max: 0.45, step: 0.01, label: 'moons noise' }),
    seed: int(2, { min: 1, max: 999, label: 'data seed' }),
  })
  const { measure, beam, depth, noise, seed } = fig

  const train = useMemo(() => moonRows(`cover-${seed}`, 200, noise), [seed, noise])
  const test = useMemo(() => moonRows(`cover-test-${seed}`, 2000, noise), [seed, noise])
  const run = useMemo(
    () =>
      sequentialCovering(train.rows, train.y, thresholdConditions(train.rows, 8), {
        measure: measure as RuleMeasure,
        measureOptions: { m: 10 },
        beam,
        maxConditions: depth,
        minSupport: 3,
        maxRules: 12,
      }),
    [train, measure, beam, depth],
  )
  const final = useMemo(() => {
    const p = applyCoveringRules(run, test.rows)
    return p.reduce((a, v, i) => a + (v === test.y[i] ? 1 : 0), 0) / test.y.length
  }, [run, test])
  const field = useMemo(() => asGrid(applyCoveringRules(run, GRID.rows), NX, NY), [run])

  const settings = `${measure}/${beam}/${depth}/${noise}/${seed}`
  const [pos, setPos] = useState({ settings, at: 0 })
  const steps = run.trace
  const at = pos.settings === settings ? Math.min(pos.at, steps.length - 1) : 0
  const s = steps[at]

  // Rules accepted before this step, and the rule this step shows.
  const accepted = run.rules.filter((_, k) => steps.findIndex((t) => t.kind === 'accept' && t.rule === k) < at)
  const shown = s && (s.kind === 'grow' || s.kind === 'accept') ? s.best : null
  const active = s ? s.active : new Uint8Array(train.rows.length).fill(1)
  const inPlay = train.y.map((_, i) => i).filter((i) => active[i])
  const out = train.y.map((_, i) => i).filter((i) => !active[i])
  const covered = s && (s.kind === 'grow' || s.kind === 'accept') ? s.covered : null

  const xAxis = useAxis({ label: 'x₁', range: MOONS_X, nice: false })
  const yAxis = useAxis({ label: 'x₂', range: MOONS_Y, nice: false, equal: xAxis })
  if (!s) return null
  const pts = (idx: number[]) => ({ x: idx.map((i) => train.x1[i]), y: idx.map((i) => train.x2[i]) })
  const coveredIdx = covered ? inPlay.filter((i) => covered[i]) : []

  return (
    <Figure
      title="Separate and conquer, step by step"
      state={fig}
      caption="Sequential covering on 200 two-moons points. Each rule starts empty; at every depth the beam search extends the kept rules by one threshold condition (x₁ or x₂ at one of 7 quantile cuts) and keeps the best few by the chosen measure; the ink box is the best rule at that depth, and the larger points are the rows in play it covers. The best rule over all depths is accepted, its covered rows leave play (grey) and the next rule is learned on what remains. Accepted rules are drawn as grey boxes. Step through with the player. Precision and the Laplace estimate accept pure boxes and end with a tail of boxes covering three or four points; the m-estimate accepts slightly impure boxes and stops earlier; weighted relative accuracy accepts two to four large, impure boxes and stops, with lower test accuracy. The shading at the last step is the finished decision list."
      controls={<Player value={at} onChange={(v) => setPos({ settings, at: v })} count={steps.length} label="step" />}
      readouts={
        <>
          <Readout label="step" value={KIND[s.kind]} />
          <Readout
            label="rule"
            value={
              s.kind === 'grow' || s.kind === 'accept'
                ? `${s.rule + 1}: if ${ruleText(s.best.conditions)} then class ${s.best.label}`
                : s.kind === 'stop'
                  ? `${s.reason}; default class ${s.defaultLabel}`
                  : `${s.removed} rows removed`
            }
          />
          {(s.kind === 'grow' || s.kind === 'accept') && (
            <Readout
              label="covered in play (rule's class : other)"
              value={`${s.best.p} : ${s.best.n}, score ${formatNumber(s.best.score)}`}
            />
          )}
          {s.kind === 'grow' && (
            <Readout
              label="beam"
              value={s.beam.map((b) => `${ruleText(b.conditions)} (${formatNumber(b.score)})`).join('; ')}
            />
          )}
          <Readout label="rows in play" value={inPlay.length} />
          <Readout label="test accuracy of the finished list" value={formatNumber(final)} />
          <Readout label="rules learned" value={run.rules.length} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        {s.kind === 'stop' && (
          <Raster
            x={GRID.gx}
            y={GRID.gy}
            z={field}
            scale="categorical"
            categoryNames={['class 0', 'class 1']}
            fillOpacity={0.3}
            boundary
          />
        )}
        {out.length > 0 && <Points name="removed" {...pts(out)} muted thin />}
        {inPlay.length > 0 && (
          <Points
            name="in play"
            x={inPlay.map((i) => train.x1[i])}
            y={inPlay.map((i) => train.x2[i])}
            group={inPlay.map((i) => train.y[i])}
            groupNames={['class 0', 'class 1']}
            thin
          />
        )}
        {coveredIdx.length > 0 && (
          <Points
            name="covered by the rule"
            {...pts(coveredIdx)}
            group={coveredIdx.map((i) => train.y[i])}
            groupNames={['class 0', 'class 1']}
            size={10}
          />
        )}
        {accepted.length > 0 && (
          <Segments
            name="accepted rules"
            segments={accepted.flatMap((r) => boxSegments(r.conditions, MOONS_X, MOONS_Y))}
            muted
            width={1.2}
          />
        )}
        {shown && (
          <Segments
            name="current rule"
            segments={boxSegments(shown.conditions, MOONS_X, MOONS_Y)}
            emphasis
            width={2.5}
          />
        )}
      </Plot>
    </Figure>
  )
}
