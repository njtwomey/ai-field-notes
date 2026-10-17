import { useMemo, useState } from 'react'
import {
  Bars,
  choice,
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
import { stream, uniform } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { aggregateBinary, booleanRuleColumnGeneration, type DnfClause } from 'aifn-compute/learning/rules'
import { titanicRows } from '../_shared/rule-figures'

type DataKind = 'titanic' | 'planted'
const DATA = [
  { value: 'titanic', label: 'Titanic: survived' },
  { value: 'planted', label: 'planted (x1 ∧ x2) ∨ (x3 ∧ ¬x4)' },
] as const

const TITANIC_FEATURES = [
  'class = 1st',
  'class = 2nd',
  'class = 3rd',
  'class = crew',
  'class ≠ 1st',
  'class ≠ 2nd',
  'class ≠ 3rd',
  'class ≠ crew',
  'sex = male',
  'sex = female',
  'age = child',
  'age = adult',
]
const TITANIC_GROUP = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2]
const PLANTED_FEATURES = [1, 2, 3, 4, 5, 6].map((j) => `x${j}`).concat([1, 2, 3, 4, 5, 6].map((j) => `¬x${j}`))

/** The binarised data: Titanic one-hot levels and their negations, or six fair coins with a planted DNF and noise. */
function binaryData(kind: DataKind, noise: number, seed: number) {
  if (kind === 'titanic') {
    const t = titanicRows()
    const rows = t.rows.map(([c, s, a]) =>
      [0, 1, 2, 3]
        .map((v) => (c === v ? 1 : 0))
        .concat(
          [0, 1, 2, 3].map((v) => (c !== v ? 1 : 0)),
          [s === 0 ? 1 : 0, s === 1 ? 1 : 0, a === 0 ? 1 : 0, a === 1 ? 1 : 0],
        ),
    )
    // Two levels of one attribute never hold together, and a level beside a negation is contradictory or redundant.
    const compatible = (a: number, b: number) =>
      TITANIC_GROUP[a] !== TITANIC_GROUP[b] || (a >= 4 && a < 8 && b >= 4 && b < 8)
    return { rows, y: t.y, names: TITANIC_FEATURES, compatible }
  }
  const n = 400
  const u = toFlat(uniform(stream(`brcg-planted-${seed}`), 0, 1, { shape: [n, 7] }))
  const rows: number[][] = []
  const y: number[] = []
  for (let i = 0; i < n; i++) {
    const x = Array.from({ length: 6 }, (_, j) => (u[7 * i + j] < 0.5 ? 1 : 0))
    let label = (x[0] && x[1]) || (x[2] && !x[3]) ? 1 : 0
    if (u[7 * i + 6] < noise) label = 1 - label
    rows.push([...x, ...x.map((v) => 1 - v)])
    y.push(label)
  }
  return { rows, y, names: PLANTED_FEATURES, compatible: (a: number, b: number) => a % 6 !== b % 6 }
}

/** Column generation for a DNF rule set: the restricted master LP, the clause priced in at each iteration, the bound. */
export function ColumnGeneration() {
  const fig = useFigureState({
    data: choice<DataKind>(DATA, 'titanic', { label: 'data' }),
    budget: int(6, { min: 2, max: 16, label: 'complexity budget C', suggestions: [4, 6, 10] }),
    maxDegree: int(3, { min: 1, max: 3, label: 'features per clause at most' }),
    noise: float(0.1, { min: 0, max: 0.3, step: 0.01, label: 'label noise (planted)', suggestions: [0, 0.1, 0.2] }),
    seed: int(1, { min: 1, max: 999, label: 'seed (planted)' }),
  })
  const { data: kind, budget, maxDegree, noise, seed } = fig

  const binary = useMemo(() => binaryData(kind, noise, seed), [kind, noise, seed])
  const agg = useMemo(() => aggregateBinary(binary.rows, binary.y), [binary])
  const result = useMemo(
    () => booleanRuleColumnGeneration(agg, { budget, maxDegree, compatible: binary.compatible }),
    [agg, budget, maxDegree, binary],
  )

  const settings = `${kind}/${budget}/${maxDegree}/${noise}/${seed}`
  const [pos, setPos] = useState({ settings, at: 0 })
  const steps = result.trace
  const at = pos.settings === settings ? Math.min(pos.at, steps.length - 1) : 0
  const s = steps[at]
  const last = at === steps.length - 1

  const text = (c: DnfClause) => c.map((j) => binary.names[j]).join(' ∧ ')
  const positives = binary.y.reduce((a, v) => a + v, 0)
  const xa = useAxis({ label: 'iteration', range: [0, Math.max(1, steps.length - 1)], integer: true })
  const ya = useAxis({ label: 'Hamming loss (rows)', hold: 'union' })
  const xb = useAxis({
    label: 'clause, in order generated',
    range: [0.4, Math.max(2, result.clauses.length) + 0.6],
    integer: true,
  })
  const yb = useAxis({ label: 'LP weight w', range: [0, 1.05], nice: false })
  if (!s) return null
  const upTo = steps.slice(0, at + 1)
  const weights = s.weights

  return (
    <Figure
      title="Boolean rules by column generation"
      state={fig}
      caption={`A rule set in disjunctive normal form predicts positive when any clause (a conjunction of binary features) holds. The integer program minimises the Hamming loss, the positives no clause covers plus, for each negative, the number of clauses covering it, under a budget C on the complexity Σ(1 + clause length). Column generation solves its linear relaxation with only the clauses generated so far: the duals of the LP price every clause of at most ${maxDegree} features, and the clause of most negative reduced cost joins. Left: the objective of the restricted LP at each iteration, the lower bound ⌈LP + (C/2) × least reduced cost⌉ on the best rule set, and the Hamming loss of the rule set the restricted integer program finally selects (dashed). When no clause has negative reduced cost the LP is solved; if the integer solution meets the bound it is certified optimal. Right: the LP weight of each clause generated so far; the clauses of the final rule set are in ink once the last iteration is reached. Step through the iterations with the player.`}
      controls={
        <Player value={at} onChange={(v) => setPos({ settings, at: v })} count={steps.length} label="iteration" />
      }
      readouts={{
        [`iteration ${at}`]: (
          <>
            <Readout label="restricted LP objective" value={formatNumber(s.lpObjective)} />
            <Readout label="budget dual λ" value={formatNumber(s.lambda)} />
            <Readout label="clause priced" value={s.clause.length > 0 ? text(s.clause) : '—'} />
            <Readout
              label="its reduced cost"
              value={`${formatNumber(s.reducedCost)} ${s.added ? '(joins)' : '(none negative: LP solved)'}`}
            />
            <Readout label="lower bound on the best rule set" value={s.lowerBound} />
          </>
        ),
        'final rule set': (
          <>
            {result.selected.map((k) => (
              <Readout key={k} label={`clause ${k + 1}`} value={text(result.clauses[k])} />
            ))}
            <Readout
              label="Hamming loss, 0–1 errors"
              value={`${result.hammingLoss}, ${result.errors} of ${binary.y.length} (${positives} positive)`}
            />
            <Readout
              label="certified"
              value={
                result.lpOptimal && result.hammingLoss === result.lowerBound
                  ? 'yes: loss equals the bound'
                  : `gap ${result.hammingLoss - result.lowerBound} rows`
              }
            />
          </>
        ),
      }}
    >
      <Plots cols={2} widths={[55, 45]}>
        <Plot x={xa} y={ya}>
          <Curve
            name="restricted LP objective"
            x={upTo.map((_, k) => k)}
            y={upTo.map((u) => u.lpObjective)}
            showPoints
            slot={0}
          />
          <Points
            name="lower bound"
            x={upTo.map((_, k) => k).filter((k) => upTo[k].lowerBound > 0)}
            y={upTo.map((u) => u.lowerBound).filter((v) => v > 0)}
            slot={1}
          />
          <Curve
            name="selected rule set"
            x={[0, Math.max(1, steps.length - 1)]}
            y={[result.hammingLoss, result.hammingLoss]}
            dashed
            emphasis
          />
        </Plot>
        <Plot x={xb} y={yb}>
          {weights.length > 0 && (
            <Bars
              name="LP weight"
              x={weights.map((_, k) => k + 1)}
              y={weights.map((w) => Math.min(1, w))}
              width={0.6}
              slot={0}
            />
          )}
          {last && result.selected.length > 0 && (
            <Points
              name="in the final rule set"
              x={result.selected.filter((k) => k < weights.length).map((k) => k + 1)}
              y={result.selected.filter((k) => k < weights.length).map(() => 1.02)}
              emphasis
              shape={1}
            />
          )}
          {weights.length === 0 && <Curve name="" x={[0.4, 1]} y={[0, 0]} muted silent />}
        </Plot>
      </Plots>
    </Figure>
  )
}
