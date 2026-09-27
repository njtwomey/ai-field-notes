import { useMemo, useState } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam } from '@/components/viz'
import { rng } from '@/lib/math'
import { categorical } from '../_shared/random'

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
  const r = rng(seed)
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
  const docs = useParam(40, { min: 1, max: MAX_DOCS, step: 1 })
  const [alpha0, setAlpha0] = useState(2)
  const [gamma, setGamma] = useState(3)
  const [words, setWords] = useState(50)
  const [seed, setSeed] = useState(1)
  const sim = useMemo(() => simulate(alpha0, gamma, words, seed), [alpha0, gamma, words, seed])
  const D = docs.value
  const xs = Array.from({ length: D }, (_, d) => d + 1)
  const expected = useMemo(() => sim.tables.map((m) => expectedTopics(m, gamma)), [sim, gamma])

  const docsPerTopic = useMemo(() => {
    const c = new Array<number>(sim.topics[D - 1]).fill(0)
    for (const u of sim.uses.slice(0, D)) for (const k of u) c[k]++
    return c
  }, [sim, D])

  return (
    <Interactive
      title="The number of topics grows with the data"
      caption="Documents arrive one at a time under the HDP prior (the Chinese restaurant franchise, no words observed). Left: topics in use (solid) and tables in the franchise, divided by 10 (dashed), against the number of documents, with the expected number of topics given the tables, the sum of γ/(γ + i − 1) over tables i (grey). Right: how many of the documents so far use each topic, in order of first appearance. Early topics are shared by most documents; new topics keep appearing, but ever more slowly. Drag the document line or step through documents with the arrows."
      controls={
        <>
          <ParamSlider label="documents D" param={docs} withArrows />
          <ParamSlider
            label="document concentration α₀"
            value={alpha0}
            onChange={setAlpha0}
            min={0.2}
            max={10}
            step={0.1}
          />
          <ParamSlider
            label="top-level concentration γ"
            value={gamma}
            onChange={setGamma}
            min={0.2}
            max={10}
            step={0.1}
          />
          <ParamSlider label="words per document" value={words} onChange={setWords} min={5} max={200} step={5} />
          <ParamSlider label="seed" value={seed} onChange={setSeed} min={1} max={10} step={1} />
        </>
      }
      readout={
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
        <XYChart
          height={300}
          xLabel="documents"
          yLabel="count"
          xRange={[1, MAX_DOCS]}
          yRange={[0, undefined]}
          handles={[{ kind: 'x', at: D, onDrag: docs.set, label: 'D' }]}
          series={[
            { name: 'topics', type: 'line', x: xs, y: sim.topics.slice(0, D), slot: 0 },
            {
              name: 'tables ÷ 10',
              type: 'line',
              x: xs,
              y: sim.tables.slice(0, D).map((t) => t / 10),
              slot: 1,
              dashed: true,
            },
            { name: 'expected topics', type: 'line', x: xs, y: expected.slice(0, D), muted: true },
          ]}
        />
        <XYChart
          height={300}
          xLabel="topic (order of first use)"
          yLabel="documents using it"
          yRange={[0, undefined]}
          series={[
            {
              name: 'documents per topic',
              type: 'bar',
              x: docsPerTopic.map((_, k) => k + 1),
              y: docsPerTopic,
              slot: 0,
            },
          ]}
        />
      </div>
    </Interactive>
  )
}
