import { useMemo, useState } from 'react'
import { Interactive, ParamSlider, Readout } from '@/components/viz'
import { MathText } from '@/components/content/MathText'
import { Diagram } from '@/components/diagram/Diagram'
import type { DiagramEdge, DiagramNode, DiagramSpec } from '@/components/diagram/types'

/** A tree-shaped factor graph: variables x1..x4, pairwise factors a, b, c and unary factors on x1 and x3. */
const VARS: Record<string, [number, number]> = { x1: [0, 0], x2: [3.2, 0], x3: [6.4, 0], x4: [3.2, 3] }
const FACTORS: Record<string, [number, number]> = {
  f1: [0, 1.5],
  fa: [1.6, 0],
  fb: [4.8, 0],
  fc: [3.2, 1.5],
  f3: [6.4, 1.5],
}
const LINKS: [string, string][] = [
  ['f1', 'x1'],
  ['x1', 'fa'],
  ['fa', 'x2'],
  ['x2', 'fb'],
  ['fb', 'x3'],
  ['x3', 'f3'],
  ['x2', 'fc'],
  ['fc', 'x4'],
]

/** The two-sweep schedule with root x2: [step, from, to]. Steps 1–3 flow in, 4–6 flow out. */
const MESSAGES: [number, string, string][] = [
  [1, 'f1', 'x1'],
  [1, 'f3', 'x3'],
  [1, 'x4', 'fc'],
  [2, 'x1', 'fa'],
  [2, 'x3', 'fb'],
  [2, 'fc', 'x2'],
  [3, 'fa', 'x2'],
  [3, 'fb', 'x2'],
  [4, 'x2', 'fa'],
  [4, 'x2', 'fb'],
  [4, 'x2', 'fc'],
  [5, 'fa', 'x1'],
  [5, 'fb', 'x3'],
  [5, 'fc', 'x4'],
  [6, 'x1', 'f1'],
  [6, 'x3', 'f3'],
]
const INWARD_STEPS = 3
const LAST = 6

const texName = (id: string) => (id.startsWith('x') ? `x_${id.slice(1)}` : `f_${id.slice(1)}`)

/** A variable's marginal is available once every neighbouring factor has sent to it. */
function hasMarginal(v: string, step: number): boolean {
  const factors = LINKS.filter(([a, b]) => a === v || b === v).map(([a, b]) => (a === v ? b : a))
  return factors.every((f) => MESSAGES.some(([s, from, to]) => s <= step && from === f && to === v))
}

function spec(step: number): DiagramSpec {
  const nodes: DiagramNode[] = [
    ...Object.entries(VARS).map(([id, [x, y]]) => ({
      id,
      x,
      y,
      shape: 'circle' as const,
      label: `$${texName(id)}$`,
      tone: 'ink' as const,
      filled: hasMarginal(id, step),
      highlight: id === 'x2',
    })),
    ...Object.entries(FACTORS).map(([id, [x, y]]) => ({
      id,
      x,
      y,
      shape: 'factor' as const,
      label: `$${texName(id)}$`,
      labelSide: (y === 0 ? 'n' : 'e') as 'n' | 'e',
    })),
  ]
  const edges: DiagramEdge[] = [
    ...LINKS.map(([a, b]) => ({ from: a, to: b, route: 'straight' as const, arrow: 'none' as const })),
    ...MESSAGES.filter(([s]) => s <= step).map(([s, from, to]) => ({
      from,
      to,
      route: 'curve' as const,
      bend: 0.3,
      tone: s <= INWARD_STEPS ? 0 : 1,
      dashed: s < step,
    })),
  ]
  return { unit: 56, nodes, edges }
}

/** Step through the two sweeps of the sum-product algorithm on a small tree. */
export function MessageSchedule() {
  const [step, setStep] = useState(3)
  const diagram = useMemo(() => spec(step), [step])
  const now = MESSAGES.filter(([s]) => s === step)
    .map(([, from, to]) => `$\\mu_{${texName(from)} \\to ${texName(to)}}$`)
    .join(', ')
  const marginals = Object.keys(VARS).filter((v) => hasMarginal(v, step))
  return (
    <Interactive
      title="The two sweeps of the sum-product algorithm"
      caption="The root is x₂ (outlined). Each step sends every message whose sender has heard from all its other neighbours; the step's messages are solid, earlier ones dashed. The inward sweep (blue) ends when x₂ has heard from every factor, so its marginal is ready. The outward sweep (orange) sends messages back to the leaves. A variable is shaded once its marginal is available; after six steps every edge has carried one message each way."
      controls={<ParamSlider label="step" value={step} onChange={setStep} min={0} max={LAST} step={1} withArrows />}
      readout={
        <>
          <Readout label="messages this step" value={step === 0 ? 'none' : <MathText text={now} />} />
          <Readout
            label="marginals ready"
            value={marginals.length ? <MathText text={marginals.map((v) => `$${texName(v)}$`).join(', ')} /> : 'none'}
          />
        </>
      }
    >
      <Diagram spec={diagram} ariaLabel="A tree factor graph with messages drawn as curved arrows along its edges" />
    </Interactive>
  )
}
