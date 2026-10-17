import { useMemo, useState } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Player,
  Plot,
  Plots,
  Points,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import {
  mineAntecedents,
  nominalConditions,
  ruleText,
  cbaBuild,
  cbaPredict,
  classAssociationRules,
  cmarBuild,
  cmarPredict,
} from 'aifn-compute/learning/rules'
import { titanicRows, TITANIC_LEVELS, TITANIC_NAMES } from '../_shared/rule-figures'

const OUTCOME = ['died', 'survived']
/** The fewest training errors any classifier of class, sex and age makes on the Titanic table. */
const FLOOR = 461

/** CBA on the Titanic data: mined class association rules, ranked, then selected by database coverage. */
export function CbaBuilder() {
  const fig = useFigureState({
    minSupport: float(0.01, {
      min: 0.002,
      max: 0.2,
      step: 0.001,
      label: 'minimum support',
      suggestions: [0.005, 0.01, 0.05],
    }),
    minConfidence: float(0.5, {
      min: 0.3,
      max: 0.95,
      step: 0.01,
      label: 'minimum confidence',
      suggestions: [0.5, 0.7, 0.9],
    }),
    maxLength: int(3, { min: 1, max: 3, label: 'conditions per rule at most' }),
  })
  const { minSupport, minConfidence, maxLength } = fig

  const data = useMemo(() => titanicRows(), [])
  const n = data.y.length
  const cars = useMemo(() => {
    const ants = mineAntecedents(data.rows, nominalConditions([4, 2, 2]), { minSupport, maxLength })
    return classAssociationRules(ants, data.y, 2, { minSupport, minConfidence })
  }, [data, minSupport, minConfidence, maxLength])
  const cba = useMemo(() => cbaBuild(cars, data.y, 2), [cars, data])
  const cmar = useMemo(() => {
    const model = cmarBuild(cars, data.y, 2, { coverage: 4 })
    const p = cmarPredict(model, cars, (r, i) => cars[r].mask[i] === 1, n, 2)
    return { rules: model.rules.length, errors: p.filter((v, i) => v !== data.y[i]).length }
  }, [cars, data, n])
  const cbaErrors = useMemo(
    () => cbaPredict(cba, cars, (r, i) => cars[r].mask[i] === 1, n).filter((v, i) => v !== data.y[i]).length,
    [cba, cars, data, n],
  )

  const settings = `${minSupport}/${minConfidence}/${maxLength}`
  const [pos, setPos] = useState({ settings, at: 0 })
  const steps = cba.trace
  const at = pos.settings === settings ? Math.min(pos.at, Math.max(0, steps.length - 1)) : 0
  const done = steps.length > 0 && at === steps.length - 1
  const step = steps[at]

  const text = (r: number) => ruleText(cars[r].conditions, { names: TITANIC_NAMES, levels: TITANIC_LEVELS })
  const view = useMemo(() => {
    const examined = new Map<number, boolean>()
    for (let k = 0; k <= at && k < steps.length; k++) examined.set(steps[k].rule, steps[k].kept)
    const final = new Set(cba.list)
    const unseen: number[] = []
    const skipped: number[] = []
    const kept: number[] = []
    const dropped: number[] = []
    cars.forEach((_, r) => {
      if (!examined.has(r)) unseen.push(r)
      else if (!examined.get(r)) skipped.push(r)
      else if (done && !final.has(r)) dropped.push(r)
      else kept.push(r)
    })
    const curve = steps.slice(0, at + 1).filter((s) => s.kept)
    return { unseen, skipped, kept, dropped, curve }
  }, [cars, steps, at, done, cba])

  const xa = useAxis({ label: 'support', range: [0, undefined] })
  const ya = useAxis({ label: 'confidence', range: [minConfidence - 0.02, 1.01], nice: false })
  const keptTotal = steps.filter((s) => s.kept).length
  const xb = useAxis({ label: 'rules in the list', range: [0, Math.max(2, keptTotal + 1)], integer: true })
  const yb = useAxis({ label: 'training errors', hold: 'union' })
  const pts = (ids: number[]) => ({ x: ids.map((r) => cars[r].support), y: ids.map((r) => cars[r].confidence) })

  if (cars.length === 0 || !step) {
    return (
      <Figure
        title="CBA: rules mined, ranked and selected by coverage"
        state={fig}
        caption="No class association rule passes both thresholds; lower the minimum support or confidence."
      >
        <Plot x={xa} y={ya}>
          <Curve name="" x={[0, 1]} y={[minConfidence, minConfidence]} muted silent />
        </Plot>
      </Figure>
    )
  }

  const r = cars[step.rule]
  const cutLength = cba.list.length
  return (
    <Figure
      title="CBA: rules mined, ranked and selected by coverage"
      state={fig}
      caption={`The ${n} people aboard the Titanic, described by class, sex and age. Left: every class association rule that passes the minimum support (rows that meet the conditions and have the predicted outcome, as a share of all rows) and the minimum confidence (the share of covered rows with that outcome), placed by its support and confidence and coloured by the outcome it predicts. CBA ranks the rules by confidence, then support, then fewer conditions, and walks down the ranking: a rule that classifies at least one remaining row correctly joins the list and removes every remaining row it covers; a rule that covers no remaining row correctly is skipped (small grey marks). Right: after each rule joins, the training errors of the list with the majority of the remaining rows as its default class; the list is cut after the first rule with the fewest errors, and rules after the cut are dropped (large grey marks). The dotted line is ${FLOOR}, the fewest errors any classifier of these three attributes can make. Step through the ranking with the player.`}
      controls={<Player value={at} onChange={(v) => setPos({ settings, at: v })} count={steps.length} label="rule" />}
      readouts={{
        [`rank ${step.rule + 1} of ${cars.length}`]: (
          <>
            <Readout label="rule" value={`${text(step.rule)} → ${OUTCOME[r.label]}`} />
            <Readout
              label="on all rows"
              value={`${r.ruleCount} of ${r.condCount}, support ${formatNumber(r.support)}, confidence ${formatNumber(r.confidence)}`}
            />
            <Readout label="remaining rows covered, correct" value={`${step.covered}, ${step.correct}`} />
            <Readout
              label="outcome"
              value={
                step.kept
                  ? `joins; default ${OUTCOME[step.defaultLabel]}, ${step.totalErrors} errors, ${step.remaining} rows left`
                  : 'skipped: no remaining row classified correctly'
              }
            />
          </>
        ),
        classifiers: (
          <>
            <Readout
              label="CBA after the cut"
              value={`${cutLength} rules, default ${OUTCOME[cba.defaultLabel]}, ${cbaErrors} errors (accuracy ${formatNumber(1 - cbaErrors / n)})`}
            />
            <Readout
              label="CMAR (δ = 4, weighted χ²)"
              value={`${cmar.rules} rules, ${cmar.errors} errors (accuracy ${formatNumber(1 - cmar.errors / n)})`}
            />
          </>
        ),
      }}
    >
      <Plots cols={2} widths={[58, 42]}>
        <Plot x={xa} y={ya}>
          {view.unseen.length > 0 && <Points name="not yet examined" {...pts(view.unseen)} muted thin />}
          {view.skipped.length > 0 && <Points name="skipped" {...pts(view.skipped)} muted shape={2} />}
          {view.dropped.length > 0 && (
            <Points name="dropped after the cut" {...pts(view.dropped)} muted shape={4} size={9} />
          )}
          {view.kept.filter((k) => cars[k].label === 0).length > 0 && (
            <Points name="in the list: died" {...pts(view.kept.filter((k) => cars[k].label === 0))} slot={0} />
          )}
          {view.kept.filter((k) => cars[k].label === 1).length > 0 && (
            <Points name="in the list: survived" {...pts(view.kept.filter((k) => cars[k].label === 1))} slot={1} />
          )}
          <Points
            name="current rule"
            x={[r.support]}
            y={[r.confidence]}
            labels={[text(step.rule)]}
            emphasis
            size={13}
            shape={3}
          />
        </Plot>
        <Plot x={xb} y={yb}>
          {view.curve.length > 0 && (
            <Curve
              name="errors of the list so far"
              x={view.curve.map((_, k) => k + 1)}
              y={view.curve.map((s) => s.totalErrors)}
              showPoints
              slot={2}
            />
          )}
          {done && <Points name="cut" x={[cutLength]} y={[cba.errors]} labels={['cut']} emphasis size={12} />}
          <Curve name="fewest possible errors" x={[0, Math.max(2, keptTotal + 1)]} y={[FLOOR, FLOOR]} dashed muted />
        </Plot>
      </Plots>
    </Figure>
  )
}
