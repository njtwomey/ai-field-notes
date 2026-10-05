import { useMemo, useState } from 'react'
import {
  Bars,
  Curve,
  Figure,
  formatNumber,
  Handle,
  int,
  Player,
  Plot,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { categorical, stream } from 'aifn-compute/foundation/random'

const MAX_DOCS = 200

type Franchise = {
  /** Topics (dishes) in use after each document. */
  topics: number[]
  /** Tables in the franchise after each document. */
  tables: number[]
  /** For each document, the set of topics it uses. */
  uses: number[][]
}

/** The Chinese restaurant franchise under the prior: words sit at tables, tables order topics from a shared menu. */
function simulate(alpha0: number, gamma: number, words: number, seed: number): Franchise {
  const r = stream(seed)
  const dishTables: number[] = []
  let tableCount = 0
  const out: Franchise = { topics: [], tables: [], uses: [] }
  for (let d = 0; d < MAX_DOCS; d++) {
    const customers: number[] = []
    const dishOf: number[] = []
    for (let i = 0; i < words; i++) {
      const t = categorical(r, [...customers, alpha0])
      if (t === customers.length) {
        const k = categorical(r, [...dishTables, gamma])
        if (k === dishTables.length) dishTables.push(0)
        dishTables[k]++
        tableCount++
        customers.push(1)
        dishOf.push(k)
      } else customers[t]++
    }
    out.topics.push(dishTables.length)
    out.tables.push(tableCount)
    out.uses.push([...new Set(dishOf)])
  }
  return out
}

/** Expected number of distinct topics when m tables each pick from a CRP with concentration γ. */
const expectedTopics = (m: number, gamma: number) => {
  let s = 0
  for (let i = 1; i <= m; i++) s += gamma / (gamma + i - 1)
  return s
}

export function TopicGrowth() {
  const state = useFigureState({
    alpha0: slider(0.2, 10, 2, { step: 0.1, label: 'document concentration α₀' }),
    gamma: slider(0.2, 10, 3, { step: 0.1, label: 'top-level concentration γ' }),
    words: int(50, { ge: 5, le: 200, suggestions: [10, 50, 100, 200], label: 'words per document' }),
    seed: int(1, { ge: 0, label: 'seed' }),
  })
  const { alpha0, gamma, words, seed } = state
  // Documents seen, minus one: the walk-through position, also moved by the document line on the chart.
  const [docIndex, setDocIndex] = useState(0)
  const sim = useMemo(() => simulate(alpha0, gamma, words, seed), [alpha0, gamma, words, seed])
  const D = docIndex + 1
  const xs = Array.from({ length: D }, (_, d) => d + 1)
  const expected = useMemo(() => sim.tables.map((m) => expectedTopics(m, gamma)), [sim, gamma])

  const docsPerTopic = useMemo(() => {
    const c = new Array<number>(sim.topics[D - 1]).fill(0)
    for (const u of sim.uses.slice(0, D)) for (const k of u) c[k]++
    return c
  }, [sim, D])

  const xAxis = useAxis({ label: 'documents', range: [1, MAX_DOCS] })
  const yAxis = useAxis({ label: 'count', range: [0, undefined], hold: 'union' })
  const xAxis2 = useAxis({ label: 'topic (order of first use)', hold: 'union' })
  const yAxis2 = useAxis({ label: 'documents using it', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="The number of topics grows with the data"
      state={state}
      caption="Documents arrive one at a time under the HDP prior (the Chinese restaurant franchise, no words observed). Left: topics in use (solid) and tables in the franchise, divided by 10 (dashed), against the number of documents, with the expected number of topics given the tables, the sum of γ/(γ + i − 1) over tables i (grey). Right: how many of the documents so far use each topic, in order of first appearance. Early topics are shared by most documents; new topics keep appearing, but ever more slowly. Drag the document line or play through the documents."
      controls={
        <Player
          value={docIndex}
          onChange={setDocIndex}
          count={MAX_DOCS}
          label="documents D"
          format={(k) => String(k + 1)}
        />
      }
      readouts={
        <>
          <Readout label="topics in use" value={sim.topics[D - 1]} />
          <Readout label="expected given tables" value={formatNumber(expected[D - 1])} />
          <Readout label="tables" value={sim.tables[D - 1]} />
          <Readout
            label="topics per document (mean)"
            value={formatNumber(sim.uses.slice(0, D).reduce((s, u) => s + u.length, 0) / D)}
          />
        </>
      }
    >
      <div className="grid gap-4 lg:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={300}>
          <Curve name="topics" x={xs} y={sim.topics.slice(0, D)} slot={0} />
          <Curve name="tables ÷ 10" x={xs} y={sim.tables.slice(0, D).map((t) => t / 10)} slot={1} dashed />
          <Curve name="expected topics" x={xs} y={expected.slice(0, D)} muted />
          <Handle
            kind="x"
            at={D}
            label="D"
            onDrag={(x) => setDocIndex(Math.min(MAX_DOCS, Math.max(1, Math.round(x))) - 1)}
          />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={300}>
          <Bars name="documents per topic" x={docsPerTopic.map((_, k) => k + 1)} y={docsPerTopic} slot={0} />
        </Plot>
      </div>
    </Figure>
  )
}
