import { useMemo, useState } from 'react'
import { learnerResponses } from 'aifn-methods/data/synthetic'
import {
  learnerModelRun,
  type LearnerModelKind,
  type LearnerModelResult,
  type LearnerModelRunResult,
} from 'aifn-methods/inference/learner-models'
import { stream } from 'aifn/foundation/random'
import {
  Bars,
  ControlRow,
  Curve,
  Figure,
  NumberSelector,
  Plot,
  Plots,
  Points,
  Readout,
  Select,
  formatNumber,
  useAxis,
} from 'aifn-render'

const fmt = (v: number | undefined, digits = 3) =>
  v !== undefined && Number.isFinite(v) ? formatNumber(Number(v.toPrecision(digits))) : '—'

const MODELS: Record<LearnerModelKind, { name: string; slot: number }> = {
  irt: { name: 'IRT (2PL)', slot: 0 },
  ktm: { name: 'linear KTM', slot: 1 },
  zilm: { name: 'IRT-ZILM', slot: 2 },
}
const ORDER: readonly LearnerModelKind[] = ['irt', 'ktm', 'zilm']
const GROUPS = ['no condition', 'dyslexia', 'dyscalculia', 'SPD', 'two or more']
const COUNTS = ['no condition', 'one condition', 'two or more']
const ZEROS = ['structural (context)', 'incorrect (ability)']

export function ZeroInflatedLearnerExplorer() {
  const [students, setStudents] = useState(120)
  const [inflationRate, setInflationRate] = useState(0.6)
  const [baselineModel, setBaselineModel] = useState<LearnerModelKind>('irt')

  const items = 40
  const attempts = 20
  const spread = 2
  const seed = 1
  const maxSteps = 120

  const run: LearnerModelRunResult = useMemo(() => {
    const data = learnerResponses(stream(`zilm/${seed}`), {
      students,
      items,
      attempts: Math.min(attempts, items),
      difficultySpread: spread,
      inflation: { dyslexia: inflationRate, dyscalculia: inflationRate, spd: inflationRate },
      baseRate: 0.02,
    })
    let last: LearnerModelRunResult | null = null
    for (const step of learnerModelRun(data, { seed, maxSteps })) {
      last = step
    }
    return last!
  }, [students, inflationRate, items, attempts, spread, seed, maxSteps])

  const of = (m: LearnerModelKind): LearnerModelResult | undefined => run?.results.find((r) => r.model === m)
  const baseline = of(baselineModel)
  const zilm = of('zilm')

  const counts = run ? Array.from(run.truth.conditionCount, (c) => Math.min(2, c)) : []
  const truthAbility = run ? Array.from(run.truth.ability) : []

  const tAxis = useAxis({ label: 'true ability θ', hold: 'union', key: `${students}/${inflationRate}` })
  const eAxis = useAxis({
    label: 'estimated ability θ̂',
    hold: 'union',
    key: `${students}/${inflationRate}`,
    equal: tAxis,
  })
  const eAxis2 = useAxis({
    label: 'estimated ability θ̂',
    hold: 'union',
    key: `${students}/${inflationRate}`,
    equal: tAxis,
  })
  const tAxis2 = useAxis({ label: 'true ability θ', hold: 'union', key: `${students}/${inflationRate}` })
  const gAxis = useAxis({ label: 'group', categories: GROUPS })
  const bAxis = useAxis({ label: 'mean θ̂ − θ (bias)', hold: 'union', key: `${students}/${inflationRate}` })
  const rAxis = useAxis({ label: 'true θ − b of student and item', hold: 'union', key: `${students}/${inflationRate}` })
  const pAxis = useAxis({ label: 'posterior that zero is structural', range: [0, 1] })

  const diagonal = (r: LearnerModelResult | undefined) => {
    if (!r || !run) return null
    const lo = Math.min(...truthAbility, ...r.fit.ability)
    const hi = Math.max(...truthAbility, ...r.fit.ability)
    return <Curve name="θ̂ = θ" x={[lo, hi]} y={[lo, hi]} emphasis dashed />
  }

  const zeros = zilm?.zeros
  const zeroX =
    zeros && run ? Array.from(zeros.student, (p, j) => run.truth.ability[p] - run.truth.difficulty[zeros.item[j]]) : []
  const zeroKind = zeros ? Array.from(zeros.structural, (s) => (s ? 0 : 1)) : []

  return (
    <Figure
      title="Ability estimates with and without zero inflation"
      purpose="Standard learner models attribute every incorrect answer to low ability. When an item format is unsuitable for a neurodivergent student, zeros arise from context rather than inability. IRT-ZILM models structural zeros with probability π, removing ability bias."
      controls={
        <>
          <ControlRow label="Learners & format penalty">
            <NumberSelector
              label="Students"
              value={students}
              onChange={setStudents}
              min={60}
              max={300}
              step={30}
              suggestions={[60, 120, 200]}
            />
            <NumberSelector
              label="Unsuitable format zero rate r"
              value={inflationRate}
              onChange={setInflationRate}
              min={0.1}
              max={0.9}
              step={0.1}
              suggestions={[0.2, 0.4, 0.6, 0.8]}
            />
          </ControlRow>
          <ControlRow label="Baseline comparator">
            <Select
              label="Baseline model"
              value={baselineModel}
              onChange={(v) => setBaselineModel(v as LearnerModelKind)}
              options={[
                { value: 'irt', label: 'IRT (2PL)' },
                { value: 'ktm', label: 'linear KTM' },
              ]}
            />
          </ControlRow>
        </>
      }
      readouts={{
        comparisons: (
          <>
            {ORDER.map((m) => {
              const r = of(m)
              return (
                <Readout
                  key={m}
                  label={`${MODELS[m].name}: equity gap · RMSE · test NLL`}
                  value={r ? `${fmt(r.equity.gap)} · ${fmt(r.equity.rmse)} · ${fmt(r.test.nll)}` : '—'}
                />
              )
            })}
            <Readout label="Structural-zero AUROC" value={fmt(zeros?.auroc)} />
          </>
        ),
      }}
      caption="Simulated students with abilities θ ~ N(0, 1) and neurodivergent conditions attempt items with varying formats. Standard IRT underestimates students facing unsuitable formats (points shifted below diagonal). IRT-ZILM separates structural zeros caused by delivery from ability-related errors, restoring calibration and closing the equity gap."
    >
      <Plots cols={2}>
        <Plot x={tAxis} y={eAxis} title={baseline ? MODELS[baseline.model].name : 'Baseline model'}>
          {diagonal(baseline)}
          {baseline && (
            <Points
              name="students"
              x={truthAbility}
              y={Array.from(baseline.fit.ability)}
              group={counts}
              groupNames={COUNTS}
              size={5}
            />
          )}
        </Plot>
        <Plot x={tAxis2} y={eAxis2} title="IRT-ZILM">
          {diagonal(zilm)}
          {zilm && (
            <Points
              name="students"
              x={truthAbility}
              y={Array.from(zilm.fit.ability)}
              group={counts}
              groupNames={COUNTS}
              size={5}
            />
          )}
        </Plot>
        <Plot x={gAxis} y={bAxis} title="Ability bias by condition group">
          {ORDER.map((m, j) => {
            const r = of(m)
            return r ? (
              <Bars
                key={m}
                name={MODELS[m].name}
                slot={j}
                x={r.equity.groups.map((g) => g.group + (j - 1) * 0.27)}
                y={r.equity.groups.map((g) => (Number.isFinite(g.bias) ? g.bias : 0))}
                width={0.25}
              />
            ) : null
          })}
          <Curve name="no bias" x={[-0.5, GROUPS.length - 0.5]} y={[0, 0]} muted dashed />
        </Plot>
        <Plot x={rAxis} y={pAxis} title="Structural zero posterior probability">
          {zeros &&
            ZEROS.map((label, kind) => {
              const pick = zeroKind.flatMap((k, j) => (k === kind ? [j] : []))
              return (
                <Points
                  key={label}
                  name={label}
                  slot={kind}
                  x={pick.map((j) => zeroX[j])}
                  y={pick.map((j) => zeros.posterior[j])}
                  size={4}
                />
              )
            })}
        </Plot>
      </Plots>
    </Figure>
  )
}
