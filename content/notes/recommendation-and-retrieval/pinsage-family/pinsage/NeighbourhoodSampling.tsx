import { useMemo } from 'react'
import { MathText } from 'aifn-render'
import { Diagram } from 'aifn-render'
import { link, variable } from 'aifn-render'
import type { DiagramEdge, DiagramNode } from 'aifn-render'
import { Interactive, ParamSlider, Readout, XYChart, useParam } from 'aifn-render'
import { rng } from '@/lib/math'

/** A toy pin–board graph: each board lists the pins saved to it. Pin 1 is the query. */
const BOARDS: number[][] = [
  [1, 2, 3, 4],
  [1, 3, 5],
  [4, 6, 7],
  [5, 8, 9],
  [2, 6],
]
const PINS = 9
const QUERY = 1
const MAX_WALKS = 60

const PIN_IDS = Array.from({ length: PINS }, (_, i) => i + 1)
const pinId = (p: number) => `p${p}`
const boardId = (b: number) => `b${b + 1}`
const boardsOf = PIN_IDS.map((p) => BOARDS.flatMap((pins, b) => (pins.includes(p) ? [b] : [])))

const PIN_GAP = 1.3
const NODES: DiagramNode[] = [
  ...PIN_IDS.map((p) => variable(pinId(p), PIN_GAP * (p - 1), 2.6, `$p_${p}$`, { w: 0.85, h: 0.85 })),
  ...BOARDS.map((_, b) => variable(boardId(b), PIN_GAP * (0.5 + 1.75 * b), 0, `$b_${b + 1}$`, { w: 0.85, h: 0.85 })),
]
const BASE_EDGES: DiagramEdge[] = BOARDS.flatMap((pins, b) => pins.map((p) => link(boardId(b), pinId(p), false)))

type Walk = { pins: number[]; boards: number[] }

/** Walks from the query: each step goes pin → random board of that pin → random pin of that board. */
function simulateWalks(length: number): Walk[] {
  const r = rng(11 + length)
  const pick = <T,>(xs: T[]) => xs[Math.floor(r.uniform() * xs.length)]
  return Array.from({ length: MAX_WALKS }, () => {
    const walk: Walk = { pins: [QUERY], boards: [] }
    let pin = QUERY
    for (let s = 0; s < length; s++) {
      const board = pick(boardsOf[pin - 1])
      pin = pick(BOARDS[board])
      walk.boards.push(board)
      walk.pins.push(pin)
    }
    return walk
  })
}

/** Random-walk neighbourhood sampling and importance pooling, as in PinSage, on a nine-pin, five-board graph. */
export function NeighbourhoodSampling() {
  const walks = useParam(12, { min: 0, max: MAX_WALKS, step: 1 })
  const length = useParam(2, { min: 1, max: 4, step: 1 })
  const top = useParam(3, { min: 1, max: 6, step: 1 })
  const n = walks.value
  const T = top.value

  const all = useMemo(() => simulateWalks(length.value), [length.value])

  const counts = useMemo(() => {
    const c = new Array<number>(PINS + 1).fill(0)
    for (const w of all.slice(0, n)) for (const p of w.pins.slice(1)) c[p]++
    return c
  }, [all, n])

  // Neighbourhood: the T non-query pins with the highest visit counts (ties broken by pin number), visited at least once.
  const neighbours = useMemo(
    () =>
      PIN_IDS.filter((p) => p !== QUERY && counts[p] > 0)
        .sort((a, b) => counts[b] - counts[a] || a - b)
        .slice(0, T),
    [counts, T],
  )
  const pooledTotal = neighbours.reduce((s, p) => s + counts[p], 0)

  const last = n > 0 ? all[n - 1] : undefined
  const spec = useMemo(() => {
    const hood = new Set(neighbours)
    const walkEdges: DiagramEdge[] = last
      ? last.boards.flatMap((b, s) => [
          link(pinId(last.pins[s]), boardId(b), true, { highlight: true }),
          link(boardId(b), pinId(last.pins[s + 1]), true, { highlight: true }),
        ])
      : []
    const nodes = NODES.map((node) => {
      const pin = node.id.startsWith('p') ? Number(node.id.slice(1)) : 0
      return { ...node, filled: hood.has(pin), highlight: pin === QUERY }
    })
    return { nodes, edges: [...BASE_EDGES, ...walkEdges] }
  }, [last, neighbours])

  const barSeries = useMemo(() => {
    const hood = new Set(neighbours)
    return [
      {
        name: 'in the top-T neighbourhood',
        type: 'bar' as const,
        x: PIN_IDS,
        y: PIN_IDS.map((p) => (hood.has(p) ? counts[p] : 0)),
        slot: 0,
      },
      {
        name: 'other pins (and the query)',
        type: 'bar' as const,
        x: PIN_IDS,
        y: PIN_IDS.map((p) => (hood.has(p) ? 0 : counts[p])),
        muted: true,
      },
    ]
  }, [counts, neighbours])

  const path = last
    ? last.pins
        .map((p, s) => (s < last.boards.length ? `$p_${p} \\to b_${last.boards[s] + 1} \\to$ ` : `$p_${p}$`))
        .join('')
    : 'none yet'
  const weights = neighbours.length
    ? neighbours.map((p) => `$p_${p}$: ${(counts[p] / pooledTotal).toFixed(2)}`).join(', ')
    : 'empty'

  return (
    <Interactive
      title="Random-walk neighbourhoods and importance pooling"
      caption={
        <MathText text="Pins sit on the bottom row and boards on the top; an edge means the pin is saved to the board. Every walk starts at the query pin $p_1$ (coloured) and repeats a step pin → random board → random pin of that board. Each pin reached is counted. Step through the walks with the arrows: the latest walk is drawn with arrows, the counts are the bars, and the $T$ most-visited pins other than $p_1$ (shaded) form the neighbourhood. Their pooling weights are the counts divided by the neighbourhood's total. Longer walks reach pins that share no board with $p_1$, such as $p_7$ and $p_8$." />
      }
      controls={
        <>
          <ParamSlider label="walks simulated" param={walks} format={(v) => String(v)} withArrows />
          <ParamSlider label="steps per walk" param={length} format={(v) => String(v)} withArrows />
          <ParamSlider label="neighbourhood size T" param={top} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="latest walk" value={<MathText text={path} />} />
          <Readout label="pooling weights" value={<MathText text={weights} />} />
        </>
      }
    >
      <Diagram
        spec={spec}
        ariaLabel="Bipartite graph of nine pins and five boards with the latest random walk from pin 1 highlighted"
      />
      <XYChart
        series={barSeries}
        xLabel="pin"
        yLabel="visits"
        xRange={[0, PINS + 1]}
        yRange={[0, undefined]}
        height={220}
        ariaLabel="Visit counts per pin, with the top-T neighbourhood highlighted"
      />
    </Interactive>
  )
}
