import { useMemo } from 'react'
import {
  choice,
  cn,
  Figure,
  Readout,
  row,
  setting,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  useFigureState,
} from 'aifn-render'
import {
  conjoin,
  disj,
  evaluateFormula,
  formulaText,
  implies,
  interpretations,
  negate,
  prop,
  type Formula,
} from 'aifn-compute/logic/propositional'

const ATOMS = ['rain', 'sprinkler', 'wet']
const rain = prop('rain')
const sprinkler = prop('sprinkler')
const wet = prop('wet')

type PremiseKey = 'rainWet' | 'sprinklerWet' | 'either' | 'dry'

/** The premises the reader can put in the knowledge base. */
const PREMISES: { key: PremiseKey; formula: Formula }[] = [
  { key: 'rainWet', formula: implies(rain, wet) },
  { key: 'sprinklerWet', formula: implies(sprinkler, wet) },
  { key: 'either', formula: disj(rain, sprinkler) },
  { key: 'dry', formula: negate(wet) },
]

const QUERIES: Record<string, Formula> = {
  wet,
  rain,
  sprinkler,
  'not-rain': negate(rain),
  'rain-or-sprinkler': disj(rain, sprinkler),
  'rain-and-sprinkler': conjoin(rain, sprinkler),
}

// Everything the table shows is fixed by the 8 interpretations, so it is computed once, when the module loads: each
// row's atom values, each premise's and each query's truth value in it, and the labels as plain text.
const ROWS = interpretations(ATOMS)
const ATOM_MARKS = ROWS.map((I) => ATOMS.map((a) => I[a]))
const PREMISE_TRUTH: Record<PremiseKey, boolean[]> = Object.fromEntries(
  PREMISES.map((p) => [p.key, ROWS.map((I) => evaluateFormula(p.formula, I))]),
) as Record<PremiseKey, boolean[]>
const QUERY_TRUTH: Record<string, boolean[]> = Object.fromEntries(
  Object.entries(QUERIES).map(([k, f]) => [k, ROWS.map((I) => evaluateFormula(f, I))]),
)
const PREMISE_TEXT: Record<PremiseKey, string> = Object.fromEntries(
  PREMISES.map((p) => [p.key, formulaText(p.formula)]),
) as Record<PremiseKey, string>
const QUERY_TEXT: Record<string, string> = Object.fromEntries(
  Object.entries(QUERIES).map(([k, f]) => [k, formulaText(f)]),
)

const schema = {
  kb: row('knowledge base', {
    rainWet: setting(true, 'rain → wet'),
    sprinklerWet: setting(true, 'sprinkler → wet'),
    either: setting(true, 'rain ∨ sprinkler'),
    dry: setting(false, '¬wet'),
  }),
  query: choice(
    Object.keys(QUERIES).map((value) => ({ value, label: QUERY_TEXT[value] })),
    'wet',
    { label: 'query φ' },
  ),
}

const mark = (b: boolean) => (b ? 'T' : 'F')

/** A truth table: the models of a knowledge base and whether a query holds in all of them. */
export function EntailmentTable() {
  const state = useFigureState(schema)
  const active = PREMISES.filter((p) => state.kb[p.key]).map((p) => p.key)
  const activeKey = active.join(',')
  const query = QUERY_TRUTH[state.query]
  // Models: the rows where every active premise is true; counter-models: models where φ is false.
  const { models, counter } = useMemo(() => {
    const keys = activeKey ? (activeKey.split(',') as PremiseKey[]) : []
    const m = ROWS.map((_, r) => keys.every((k) => PREMISE_TRUTH[k][r]))
    return { models: m, counter: m.map((isModel, r) => isModel && !query[r]) }
  }, [activeKey, query])
  const nModels = models.filter(Boolean).length
  const nCounter = counter.filter(Boolean).length

  return (
    <Figure
      title="Entailment by checking every interpretation"
      state={state}
      hoverReadout={false}
      readouts={
        <>
          <Readout label="interpretations" value={ROWS.length} />
          <Readout label="models of the knowledge base" value={nModels} />
          <Readout label="counter-models (KB true, φ false)" value={nCounter} />
          <Readout
            label="verdict"
            value={nModels === 0 ? 'KB unsatisfiable: it entails every φ' : nCounter === 0 ? 'KB ⊨ φ' : 'KB ⊭ φ'}
          />
        </>
      }
      caption="Switch premises in or out of the knowledge base KB and pick a query φ. Each row is one interpretation of the three atoms. Shaded rows are the models of KB, the interpretations that make every premise true. KB entails φ when φ is true in every shaded row; a shaded row with φ false is a counter-model and is marked. Removing a premise can only add models, so it can only remove entailments."
    >
      <div className="h-full overflow-auto">
        <Table className="font-mono text-sm">
          <TableHeader>
            <TableRow>
              {ATOMS.map((a) => (
                <TableHead key={a} className="text-center">
                  {a}
                </TableHead>
              ))}
              {active.map((k) => (
                <TableHead key={k} className="text-center">
                  {PREMISE_TEXT[k]}
                </TableHead>
              ))}
              <TableHead className="text-center">KB</TableHead>
              <TableHead className="text-center">φ = {QUERY_TEXT[state.query]}</TableHead>
              <TableHead />
            </TableRow>
          </TableHeader>
          <TableBody>
            {ROWS.map((_, r) => (
              <TableRow
                key={r}
                className={cn(models[r] && 'bg-muted font-semibold', !models[r] && 'text-muted-foreground')}
              >
                {ATOM_MARKS[r].map((v, j) => (
                  <TableCell key={j} className="text-center">
                    {mark(v)}
                  </TableCell>
                ))}
                {active.map((k) => (
                  <TableCell key={k} className="text-center">
                    {mark(PREMISE_TRUTH[k][r])}
                  </TableCell>
                ))}
                <TableCell className="text-center">{mark(models[r])}</TableCell>
                <TableCell className="text-center">{mark(query[r])}</TableCell>
                <TableCell className="font-sans text-xs">
                  {counter[r] ? 'counter-model' : models[r] ? 'model' : ''}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </Figure>
  )
}
