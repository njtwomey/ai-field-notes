import { useMemo, useState } from 'react'
import {
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
import { growTree, predictTree } from 'aifn-methods/learning/trees-and-ensembles'
import { treeRules, type ShapTree } from 'aifn-compute/learning/explain'
import { tensor, toFlat } from 'aifn-compute/foundation/tensor'
import { applyClassRules, c45Rules, type ClassRule, type TreeRulesStep, ruleText } from 'aifn-compute/learning/rules'
import { asGrid, boxSegments, moonRows, MOONS_X, MOONS_Y, predictionGrid } from '../_shared/rule-figures'

const NX = 61
const NY = 41
const GRID = predictionGrid(MOONS_X, MOONS_Y, NX, NY)

const KIND: Record<TreeRulesStep['kind'], string> = {
  paths: 'one rule per leaf',
  generalise: 'delete a condition (pessimistic error does not rise)',
  merge: 'merge duplicate rules',
  select: "keep the subset of a class's rules with the shortest description",
  order: 'order the classes by false positives; add the default',
}

const accuracy = (pred: ArrayLike<number>, y: number[]) =>
  y.reduce((a, v, i) => a + (pred[i] === v ? 1 : 0), 0) / y.length

/** C4.5rules on two moons: a tree's leaves become rules, which are generalised, selected and ordered, stepped. */
export function TreeToRules() {
  const fig = useFigureState({
    depth: int(9, { min: 1, max: 12, label: 'tree depth', suggestions: [3, 6, 9] }),
    cf: float(0.25, { min: 0.01, max: 0.5, step: 0.01, label: 'confidence level CF', suggestions: [0.1, 0.25, 0.4] }),
    noise: float(0.3, { min: 0, max: 0.5, step: 0.01, label: 'moons noise' }),
    seed: int(3, { min: 1, max: 999, label: 'data seed' }),
  })
  const { depth, cf, noise, seed } = fig

  const train = useMemo(() => moonRows(`c45-${seed}`, 300, noise), [seed, noise])
  const test = useMemo(() => moonRows(`c45-test-${seed}`, 2000, noise), [seed, noise])
  const tree = useMemo(
    () =>
      growTree(undefined, {
        x: tensor(train.rows),
        y: tensor(train.y),
        task: 'classification',
        params: { maxDepth: depth, criterion: 'entropy' },
      }),
    [train, depth],
  )
  const paths: ClassRule[] = useMemo(
    () =>
      treeRules(tree as unknown as ShapTree).map((r) => ({
        conditions: r.rule,
        label: r.value[1] > r.value[0] ? 1 : 0,
      })),
    [tree],
  )
  const result = useMemo(() => c45Rules(train.rows, train.y, paths, { cf }), [train, paths, cf])
  const scores = useMemo(() => {
    const treeLabels = (rows: number[][]) => {
      const p = toFlat(predictTree(tree, tensor(rows)))
      return rows.map((_, i) => (p[2 * i + 1] > p[2 * i] ? 1 : 0))
    }
    return {
      treeTrain: accuracy(treeLabels(train.rows), train.y),
      treeTest: accuracy(treeLabels(test.rows), test.y),
      rulesTrain: accuracy(applyClassRules(result.rules, result.defaultLabel, train.rows), train.y),
      rulesTest: accuracy(applyClassRules(result.rules, result.defaultLabel, test.rows), test.y),
    }
  }, [tree, result, train, test])
  const field = useMemo(() => asGrid(applyClassRules(result.rules, result.defaultLabel, GRID.rows), NX, NY), [result])

  const settings = `${depth}/${cf}/${noise}/${seed}`
  const [pos, setPos] = useState({ settings, at: 0 })
  const steps = result.trace
  const at = pos.settings === settings ? Math.min(pos.at, steps.length - 1) : 0
  const s = steps[at]

  const xAxis = useAxis({ label: 'x₁', range: MOONS_X, nice: false })
  const yAxis = useAxis({ label: 'x₂', range: MOONS_Y, nice: false, equal: xAxis })
  if (!s) return null
  const rules = s.rules
  const focus = s.kind === 'generalise' ? s.index : -1
  const boxes = (label: number) =>
    rules.flatMap((r, k) => (r.label === label && k !== focus ? boxSegments(r.conditions, MOONS_X, MOONS_Y) : []))
  const conditions = rules.reduce((a, r) => a + r.conditions.length, 0)
  const b0 = boxes(0)
  const b1 = boxes(1)

  return (
    <Figure
      title="From a tree to a rule set"
      state={fig}
      caption="C4.5rules on 300 noisy two-moons points. Step 0 shows one rule per leaf of an entropy tree of the chosen depth, as boxes coloured by the class they predict. Each later step deletes one condition from one rule (ink box) when the rule's pessimistic error, the upper end of a one-sided binomial confidence interval at level CF, does not rise, so boxes grow and start to overlap; then duplicates merge, each class keeps the subset of its rules with the shortest description, and the classes are ordered. The shading at the last step is the finished rule list. A deep tree's 30 or so leaves shrink to about a dozen short rules, and on 2000 fresh points the rules usually match or beat the tree; for a shallow tree, which has little to remove, they are often slightly worse. A smaller CF is more pessimistic and deletes more conditions."
      controls={<Player value={at} onChange={(v) => setPos({ settings, at: v })} count={steps.length} label="step" />}
      readouts={{
        step: (
          <>
            <Readout label="step" value={KIND[s.kind]} />
            {s.kind === 'generalise' && (
              <>
                <Readout label="deleted" value={ruleText([s.deletion.dropped])} />
                <Readout
                  label="covered N, errors E, pessimistic U"
                  value={`${s.deletion.before.N}, ${s.deletion.before.E}, ${formatNumber(s.deletion.before.U)} → ${s.deletion.after.N}, ${s.deletion.after.E}, ${formatNumber(s.deletion.after.U)}`}
                />
              </>
            )}
            {s.kind === 'select' && (
              <Readout
                label={`class ${s.label}`}
                value={`keeps ${s.kept.length}, drops ${s.dropped.length}, ${formatNumber(s.bits)} bits`}
              />
            )}
            {s.kind === 'order' && <Readout label="default class" value={s.defaultLabel} />}
            <Readout label="rules, conditions" value={`${rules.length}, ${conditions}`} />
          </>
        ),
        accuracy: (
          <>
            <Readout
              label="tree (train, test)"
              value={`${formatNumber(scores.treeTrain)}, ${formatNumber(scores.treeTest)}`}
            />
            <Readout
              label="final rules (train, test)"
              value={`${formatNumber(scores.rulesTrain)}, ${formatNumber(scores.rulesTest)}`}
            />
            <Readout label="leaves → final rules" value={`${paths.length} → ${result.rules.length}`} />
          </>
        ),
      }}
    >
      <Plot x={xAxis} y={yAxis}>
        {s.kind === 'order' && (
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
        <Points
          name="training points"
          x={train.x1}
          y={train.x2}
          group={train.y}
          groupNames={['class 0', 'class 1']}
          thin
        />
        {b0.length > 0 && <Segments name="rules for class 0" segments={b0} slot={0} width={1.2} />}
        {b1.length > 0 && <Segments name="rules for class 1" segments={b1} slot={1} width={1.2} />}
        {focus >= 0 && (
          <Segments
            name="rule being generalised"
            segments={boxSegments(rules[focus].conditions, MOONS_X, MOONS_Y)}
            emphasis
            width={2.5}
          />
        )}
      </Plot>
    </Figure>
  )
}
