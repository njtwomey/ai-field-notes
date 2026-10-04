import { useMemo, useState } from 'react'
import {
  Figure,
  formatNumber,
  Handle,
  Player,
  Plot,
  Readout,
  type Segment,
  seriesLayers,
  type SeriesSpec,
  slider,
  useAxis,
  useFigureState,
  Vectors,
} from 'aifn-render'

type Vec = [number, number]
const R = 4
const STEPS = 3
const coord = (initial: number) => slider(-R, R, initial, { step: 0.1, onChart: true })
const fmt = (v: Vec) => `(${formatNumber(v[0])}, ${formatNumber(v[1])})`

const STAGES = [
  'Start: two independent vectors v₁ and v₂.',
  'Step 1: q₁ = v₁ / ‖v₁‖.',
  'Step 2: subtract the projection (q₁ᵀv₂) q₁ from v₂, leaving w₂.',
  'Step 3: q₂ = w₂ / ‖w₂‖.',
]

/** Gram–Schmidt on two vectors in the plane, one step at a time. */
export function GramSchmidtStepper() {
  const state = useFigureState({ v1x: coord(3), v1y: coord(1), v2x: coord(1.5), v2y: coord(2.5) })
  const v1: Vec = useMemo(() => [state.v1x, state.v1y], [state.v1x, state.v1y])
  const v2: Vec = useMemo(() => [state.v2x, state.v2y], [state.v2x, state.v2y])
  const [step, setStep] = useState(0)

  const r = useMemo(() => {
    const n1 = Math.hypot(...v1)
    const q1: Vec = n1 > 0 ? [v1[0] / n1, v1[1] / n1] : [1, 0]
    const r12 = q1[0] * v2[0] + q1[1] * v2[1]
    const proj: Vec = [r12 * q1[0], r12 * q1[1]]
    const w2: Vec = [v2[0] - proj[0], v2[1] - proj[1]]
    const r22 = Math.hypot(...w2)
    const q2: Vec = r22 > 1e-9 ? [w2[0] / r22, w2[1] / r22] : [0, 0]
    return { n1, q1, r12, proj, w2, r22, q2 }
  }, [v1, v2])

  const { series, vectors } = useMemo(() => {
    // The inputs are thin muted segments from the origin; the arrows are the orthonormal outputs.
    const series: SeriesSpec[] = [
      { name: 'input v₁', type: 'line', x: [0, v1[0]], y: [0, v1[1]], muted: true },
      { name: 'input v₂', type: 'line', x: [0, v2[0]], y: [0, v2[1]], muted: true },
    ]
    const vectors: Segment[] = []
    if (step >= 1) vectors.push({ from: [0, 0], to: r.q1 })
    if (step >= 2) {
      series.push(
        { name: 'projection (q₁ᵀv₂) q₁', type: 'line', x: [0, r.proj[0]], y: [0, r.proj[1]], slot: 1 },
        {
          name: 'w₂ = v₂ − projection',
          type: 'line',
          x: [r.proj[0], v2[0]],
          y: [r.proj[1], v2[1]],
          slot: 2,
          dashed: true,
        },
      )
    }
    if (step >= 3) vectors.push({ from: [0, 0], to: r.q2 })
    return { series, vectors }
  }, [r, step, v1, v2])

  const xAxis = useAxis({ label: 'x₁', range: [-R, R] })
  const yAxis = useAxis({ label: 'x₂', range: [-R, R], equal: xAxis })
  return (
    <Figure
      title="Gram–Schmidt, one step at a time"
      state={state}
      caption={`${STAGES[step]} Drag the ends of v₁ and v₂ to change the inputs at any step. The arrows are the orthonormal basis q₁, q₂. Make v₂ nearly parallel to v₁ and w₂ becomes tiny: dividing by its length is where rounding errors grow.`}
      controls={<Player value={step} onChange={setStep} count={STEPS + 1} label="step" />}
      readouts={
        <>
          <Readout label="r₁₁ = ‖v₁‖" value={formatNumber(r.n1)} />
          <Readout label="r₁₂ = q₁ᵀv₂" value={step >= 2 ? formatNumber(r.r12) : '·'} />
          <Readout label="r₂₂ = ‖w₂‖" value={step >= 3 ? formatNumber(r.r22) : '·'} />
          <Readout label="q₁" value={step >= 1 ? fmt(r.q1) : '·'} />
          <Readout label="q₂" value={step >= 3 ? fmt(r.q2) : '·'} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-lg">
        <Plot x={xAxis} y={yAxis}>
          {seriesLayers(series)}
          <Vectors vectors={vectors} />
          <Handle {...state.handle(['v1x', 'v1y'], { label: 'v₁' })} />
          <Handle {...state.handle(['v2x', 'v2y'], { label: 'v₂' })} />
        </Plot>
      </div>
    </Figure>
  )
}
