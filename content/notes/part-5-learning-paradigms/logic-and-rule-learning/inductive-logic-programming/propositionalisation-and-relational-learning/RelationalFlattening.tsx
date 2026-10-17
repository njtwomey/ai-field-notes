import { useMemo, useState } from 'react'
import { applyRuleList, ruleList, type Predicate } from 'aifn-compute/learning/explain'
import {
  choice,
  cn,
  Figure,
  formatNumber,
  int,
  Readout,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  useFigureState,
} from 'aifn-render'
import {
  aggregateFeatures,
  attributeFeatures,
  existentialFeatures,
  featureTable,
  wordify,
  type PropositionalFeature,
} from 'aifn-compute/logic/induction'
import { CUSTOMERS } from './customers'

const schema = {
  method: choice(
    [
      { value: 'existential', label: 'first-order features (RSD)' },
      { value: 'aggregate', label: 'counts (RELAGGS)' },
      { value: 'words', label: 'Wordification' },
    ] as const,
    'existential',
    { label: 'construction' },
  ),
  tests: int(2, { label: 'tests per feature at most', min: 1, max: 3, suggestions: [1, 2, 3] }),
}

const PEOPLE = CUSTOMERS.individuals
const ORDERS = CUSTOMERS.children[0].rows
const LABELS = PEOPLE.map((p) => p.label ?? 0)

const condition = (f: PropositionalFeature, p: Predicate) => {
  const lo = Number.isFinite(p.lower) ? `${formatNumber(p.lower)} < ` : ''
  const hi = Number.isFinite(p.upper) ? ` ≤ ${formatNumber(p.upper)}` : ''
  return `${lo}[${f.name}]${hi}`
}

/** A customers-and-orders database turned into one row per customer by three propositionalisation methods. */
export function RelationalFlattening() {
  const state = useFigureState(schema)
  const features = useMemo((): PropositionalFeature[] => {
    if (state.method === 'existential')
      return [...attributeFeatures(CUSTOMERS), ...existentialFeatures(CUSTOMERS, { maxTests: state.tests })]
    if (state.method === 'aggregate') return [...attributeFeatures(CUSTOMERS), ...aggregateFeatures(CUSTOMERS)]
    return wordify(CUSTOMERS, { maxWords: state.tests })
  }, [state.method, state.tests])
  const learnt = useMemo(() => {
    const X = featureTable(CUSTOMERS, features)
    const list = ruleList(X, LABELS, { minSupport: 1, bins: 4, maxConditions: 2, maxRules: 4 })
    const predicted = Array.from(applyRuleList(list, X).labels)
    const rowKey = X.map((r) => r.join(','))
    const conflicts: string[] = []
    for (let i = 0; i < PEOPLE.length; i++)
      for (let j = i + 1; j < PEOPLE.length; j++)
        if (rowKey[i] === rowKey[j] && LABELS[i] !== LABELS[j]) conflicts.push(`${PEOPLE[i].id} and ${PEOPLE[j].id}`)
    return { list, errors: predicted.filter((p, i) => p !== LABELS[i]).length, conflicts }
  }, [features])

  const key = `${state.method}/${state.tests}`
  const [picked, setPicked] = useState<{ key: string; name: string } | null>(null)
  const selected = features.find((f) => picked?.key === key && f.name === picked.name) ?? null
  const witness = new Set(selected?.witnesses?.flat() ?? [])
  const value = (v: number) => (Number.isInteger(v) ? `${v}` : v.toFixed(2))

  return (
    <Figure
      title="From two linked tables to one row per customer"
      state={state}
      defaultSize="L"
      hoverReadout={false}
      readouts={
        <>
          <Readout label="features" value={features.length} />
          <Readout
            label="customers no feature tells apart"
            value={learnt.conflicts.length ? learnt.conflicts.join('; ') : 'none'}
          />
          <Readout
            label="decision list learnt on the flat table"
            value={
              learnt.list.rules
                .map(
                  (r) =>
                    `if ${r.rule.map((p) => condition(features[p.feature], p)).join(' and ')} then ${r.label ? 'churn' : 'stay'}`,
                )
                .join('; ') + `; else ${learnt.list.defaultLabel ? 'churn' : 'stay'}`
            }
          />
          <Readout label="training errors" value={`${learnt.errors} of ${PEOPLE.length}`} />
        </>
      }
      caption="Eight customers (left, top) and their fifteen orders (left, bottom); a customer churned when one of their orders was both late and of high value. Pick a construction and the most attribute tests a feature may combine. The table on the right has one column per customer and one row per constructed feature: an existential feature is 1 when some order of the customer passes all its tests, a count is the number of orders that pass its test, and a Wordification word counts the orders that contain it. Click a feature to highlight the orders behind it. A decision list (sequential covering) is then learnt on the flat table. With single tests, Amy and Ben get identical rows and no learner can separate them."
    >
      <div className="grid h-full gap-3 md:grid-cols-[minmax(0,0.75fr)_minmax(0,1.25fr)]">
        <div className="h-full overflow-auto">
          <Table className="text-xs">
            <TableHeader>
              <TableRow>
                <TableHead>customer</TableHead>
                <TableHead>plan</TableHead>
                <TableHead>churned</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {PEOPLE.map((p, i) => (
                <TableRow key={p.id}>
                  <TableCell className="font-mono">{p.id}</TableCell>
                  <TableCell>{p.attributes.plan}</TableCell>
                  <TableCell>{LABELS[i] ? 'yes' : 'no'}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          <Table className="mt-2 text-xs">
            <TableHeader>
              <TableRow>
                <TableHead>order</TableHead>
                <TableHead>customer</TableHead>
                <TableHead>delivery</TableHead>
                <TableHead>value</TableHead>
                <TableHead>channel</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {ORDERS.map((o) => (
                <TableRow key={o.id} className={cn(witness.has(o.id) && 'bg-muted font-semibold')}>
                  <TableCell className="font-mono">{o.id}</TableCell>
                  <TableCell className="font-mono">{o.parent}</TableCell>
                  <TableCell>{o.attributes.delivery}</TableCell>
                  <TableCell>{o.attributes.value}</TableCell>
                  <TableCell>{o.attributes.channel}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
        <div className="h-full overflow-auto">
          <Table className="text-xs">
            <TableHeader>
              <TableRow>
                <TableHead>feature</TableHead>
                {PEOPLE.map((p) => (
                  <TableHead key={p.id} className="text-center font-mono">
                    {p.id}
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              <TableRow>
                <TableCell className="text-muted-foreground">churned (label)</TableCell>
                {LABELS.map((l, i) => (
                  <TableCell key={i} className="text-center text-muted-foreground">
                    {l}
                  </TableCell>
                ))}
              </TableRow>
              {features.map((f) => (
                <TableRow
                  key={f.name}
                  className={cn('cursor-pointer', selected?.name === f.name && 'bg-muted font-semibold')}
                  onClick={() => setPicked({ key, name: f.name })}
                  aria-selected={selected?.name === f.name}
                >
                  <TableCell className="font-mono whitespace-nowrap">{f.name}</TableCell>
                  {f.values.map((v, i) => (
                    <TableCell key={i} className={cn('text-center', v === 0 && 'text-muted-foreground')}>
                      {value(v)}
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </div>
    </Figure>
  )
}
