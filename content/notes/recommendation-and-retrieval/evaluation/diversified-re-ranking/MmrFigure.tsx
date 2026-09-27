import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'
import { rng } from '@/lib/math'

type Item = { x: number; y: number; rel: number; group: number }

/** Twenty-four candidate items in a 2-D embedding space, in three taste clusters of decreasing relevance. */
const ITEMS: Item[] = (() => {
  const r = rng(5)
  const centres = [
    { x: 0.25, y: 0.7, rel: 0.9 },
    { x: 0.75, y: 0.7, rel: 0.7 },
    { x: 0.5, y: 0.2, rel: 0.55 },
  ]
  return Array.from({ length: 24 }, (_, i) => {
    const c = centres[i % 3]
    return {
      x: c.x + 0.07 * r.normal(),
      y: c.y + 0.07 * r.normal(),
      rel: Math.min(1, Math.max(0, c.rel + 0.08 * r.normal())),
      group: i % 3,
    }
  })
})()

const sim = (a: Item, b: Item) => Math.exp(-((a.x - b.x) ** 2 + (a.y - b.y) ** 2) / 0.05)

/** Greedy maximal marginal relevance: each step adds argmax λ·rel − (1 − λ)·max similarity to the chosen items. */
function mmr(lambda: number, k: number): number[] {
  const chosen: number[] = []
  while (chosen.length < k) {
    let best = -1
    let bestScore = -Infinity
    ITEMS.forEach((it, i) => {
      if (chosen.includes(i)) return
      const redundancy = chosen.length ? Math.max(...chosen.map((j) => sim(it, ITEMS[j]))) : 0
      const score = lambda * it.rel - (1 - lambda) * redundancy
      if (score > bestScore) {
        bestScore = score
        best = i
      }
    })
    chosen.push(best)
  }
  return chosen
}

/** MMR re-ranking of a candidate set: the trade-off weight λ and the list built one pick at a time. */
export function MmrFigure() {
  const lambda = useParam(0.7, { min: 0, max: 1, step: 0.05 })
  const k = useParam(6, { min: 1, max: 10, step: 1 })
  const chosen = useMemo(() => mmr(lambda.value, k.value), [lambda.value, k.value])

  const meanRel = chosen.reduce((s, i) => s + ITEMS[i].rel, 0) / chosen.length
  let pairs = 0
  let dist = 0
  for (let a = 0; a < chosen.length; a++)
    for (let b = a + 1; b < chosen.length; b++) {
      dist += 1 - sim(ITEMS[chosen[a]], ITEMS[chosen[b]])
      pairs++
    }
  const clusters = new Set(chosen.map((i) => ITEMS[i].group)).size

  const series: XYSeries[] = [
    {
      name: 'candidates',
      type: 'scatter',
      x: ITEMS.map((it) => it.x),
      y: ITEMS.map((it) => it.y),
      group: ITEMS.map((it) => it.group),
      groupNames: ['cluster A (most relevant)', 'cluster B', 'cluster C'],
    },
    {
      name: 'selected, in order',
      type: 'line',
      x: chosen.map((i) => ITEMS[i].x),
      y: chosen.map((i) => ITEMS[i].y),
      emphasis: true,
    },
  ]

  return (
    <Interactive
      title="Maximal marginal relevance"
      caption="Twenty-four candidates in an embedding space, in three clusters; cluster A is the most relevant. MMR adds one item at a time, choosing the item with the best trade-off between relevance and similarity to the items already chosen. The ink path joins the selected items in the order chosen. With λ = 1 the list is the top items by relevance, all from cluster A; lowering λ pulls in items from the other clusters. Step the list length with the arrows."
      controls={
        <>
          <ParamSlider label="relevance weight λ" param={lambda} />
          <ParamSlider label="list length k" param={k} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="mean relevance" value={formatNumber(meanRel)} />
          <Readout label="intra-list diversity" value={pairs ? formatNumber(dist / pairs) : '–'} />
          <Readout label="clusters covered" value={`${clusters} of 3`} />
        </>
      }
    >
      <XYChart
        series={series}
        xLabel="embedding dimension 1"
        yLabel="embedding dimension 2"
        xRange={[0, 1]}
        yRange={[0, 1]}
        equalAspect
      />
    </Interactive>
  )
}
