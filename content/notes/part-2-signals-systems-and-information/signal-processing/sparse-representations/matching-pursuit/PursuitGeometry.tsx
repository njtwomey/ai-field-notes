import { useMemo, useState } from 'react'
import { matchingPursuitSteps, orthogonalMatchingPursuitSteps, type PursuitState } from 'aifn-compute/signal/sparse'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import {
  choice,
  Curve,
  Figure,
  formatNumber,
  Handle,
  Player,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
  type Vector,
  Vectors,
} from 'aifn-render'

type Vec = [number, number]
const R = 2.2
const MAX_STEPS = 30
const FLOOR = 1e-12
const SUB = ['₁', '₂', '₃', '₄']

/** Atom directions in degrees: three or four spread out, or a basis of two atoms 25° apart. */
const DICTIONARIES: Record<string, number[]> = {
  spread: [0, 70, 125],
  coherent: [0, 25],
  four: [0, 45, 100, 150],
}

const coord = (initial: number) => slider(-2, 2, initial, { step: 0.05, onChart: true })
const atomName = (j: number) => `d${SUB[j]}`

/** Matching pursuit and orthogonal matching pursuit in the plane, with an overcomplete dictionary of unit atoms. */
export function PursuitGeometry() {
  const state = useFigureState({
    dictionary: choice(
      [
        { value: 'spread', label: 'three spread atoms' },
        { value: 'coherent', label: 'two atoms 25° apart' },
        { value: 'four', label: 'four atoms' },
      ],
      'spread',
      { label: 'dictionary' },
    ),
    method: choice(
      [
        { value: 'mp', label: 'matching pursuit' },
        { value: 'omp', label: 'orthogonal matching pursuit' },
      ],
      'mp',
      { label: 'pursuit drawn' },
    ),
    yx: coord(1.3),
    yy: coord(1.1),
  })
  const [step, setStep] = useState(0)

  const atoms = useMemo(
    () =>
      DICTIONARIES[state.dictionary].map((a): Vec => [Math.cos((a * Math.PI) / 180), Math.sin((a * Math.PI) / 180)]),
    [state.dictionary],
  )
  const y: Vec = useMemo(() => [state.yx, state.yy], [state.yx, state.yy])

  const runs = useMemo(() => {
    const D = [atoms.map((d) => d[0]), atoms.map((d) => d[1])]
    const mp = trace(matchingPursuitSteps(D, y), undefined, MAX_STEPS).steps as PursuitState[]
    const omp = trace(orthogonalMatchingPursuitSteps(D, y), undefined, MAX_STEPS).steps as PursuitState[]
    return { mp, omp }
  }, [atoms, y])

  const states = state.method === 'mp' ? runs.mp : runs.omp
  const count = Math.max(runs.mp.length, runs.omp.length, 3)
  const t = Math.min(step, count - 1)
  const at = (list: PursuitState[]) => list[Math.min(t, list.length - 1)]
  const current = at(states)

  const layers = useMemo(() => {
    const spans: { x: number[]; y: number[] } = { x: [], y: [] }
    for (const d of atoms) {
      spans.x.push(-3 * d[0], 3 * d[0], NaN)
      spans.y.push(-3 * d[1], 3 * d[1], NaN)
    }
    const dictionary: Vector[] = atoms.map((d, j) => ({ from: [0, 0], to: d, label: atomName(j), muted: true }))
    const path: Vector[] = []
    const slot = state.method === 'mp' ? 0 : 1
    if (state.method === 'mp') {
      // Each matching pursuit step adds a multiple of one atom: the approximation walks parallel to the atoms.
      let a: Vec = [0, 0]
      for (let i = 1; i <= Math.min(t, states.length - 1); i++) {
        const x = toFlat(states[i].x)
        const next: Vec = [0, 0]
        x.forEach((c, j) => {
          next[0] += c * atoms[j][0]
          next[1] += c * atoms[j][1]
        })
        path.push({ from: a, to: next, slot, width: 1.5, head: 7 })
        a = next
      }
    } else {
      // Orthogonal matching pursuit refits every chosen coefficient: draw x_j d_j tip to tail in the order picked.
      const x = toFlat(current.x)
      let a: Vec = [0, 0]
      for (const j of current.support) {
        const next: Vec = [a[0] + x[j] * atoms[j][0], a[1] + x[j] * atoms[j][1]]
        path.push({ from: a, to: next, slot, width: 1.5, head: 7, label: `${formatNumber(x[j])} ${atomName(j)}` })
        a = next
      }
    }
    const r = toFlat(current.residual)
    const approx: Vec = [y[0] - r[0], y[1] - r[1]]
    return { spans, dictionary, path, approx }
  }, [atoms, current, state.method, states, t, y])

  const decay = useMemo(() => {
    const line = (list: PursuitState[]) => ({
      x: list.map((s) => s.t),
      y: list.map((s) => Math.max(s.residualNorm, FLOOR)),
    })
    return { mp: line(runs.mp), omp: line(runs.omp) }
  }, [runs])
  const mpNow = at(runs.mp)
  const ompNow = at(runs.omp)

  const xAxis = useAxis({ label: 'first coordinate', range: [-R, R] })
  const yAxis = useAxis({ label: 'second coordinate', range: [-R, R], equal: xAxis })
  const sAxis = useAxis({ label: 'step t', range: [0, count - 1], integer: true })
  const nAxis = useAxis({ label: 'residual norm ‖r‖', log: true, range: [FLOOR, 10] })

  const picked = current.support.map(atomName).join(', ') || 'none'
  return (
    <Figure
      title="Pursuit in the plane: one atom per step"
      state={state}
      caption="Left: the atoms d₁, d₂, … are unit arrows and their spans are faint lines. Drag the signal y. The coloured arrows build the approximation Dx, and the dashed line is the residual r = y − Dx. Matching pursuit adds a multiple of one atom per step, so its path runs parallel to the atoms and zigzags between two of them, closing the gap by a constant factor each time. Orthogonal matching pursuit refits both chosen coefficients at step 2 and lands on y exactly. Right: the residual norm of both pursuits on a log scale. With only two atoms 25° apart, each matching pursuit step shrinks the residual by the factor cos 25° ≈ 0.91, so the zigzag is slow."
      controls={<Player value={t} onChange={setStep} count={count} label="step" />}
      readouts={
        <>
          <Readout label="atoms picked" value={picked} />
          <Readout label="‖r‖, matching pursuit" value={formatNumber(mpNow.residualNorm)} />
          <Readout label="‖r‖, orthogonal matching pursuit" value={formatNumber(ompNow.residualNorm)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} legend={false}>
          <Curve name="atom spans" x={layers.spans.x} y={layers.spans.y} muted silent />
          <Vectors vectors={layers.dictionary} />
          <Vectors vectors={layers.path} />
          <Curve name="residual r" x={[layers.approx[0], y[0]]} y={[layers.approx[1], y[1]]} slot={2} dashed live />
          <Points name="signal y" x={[y[0]]} y={[y[1]]} emphasis size={9} live />
          <Handle {...state.handle(['yx', 'yy'], { label: 'y' })} />
        </Plot>
        <Plot x={sAxis} y={nAxis}>
          <Curve name="matching pursuit" x={decay.mp.x} y={decay.mp.y} slot={0} showPoints />
          <Curve name="orthogonal matching pursuit" x={decay.omp.x} y={decay.omp.y} slot={1} showPoints />
          <Points
            name="step t"
            x={[mpNow.t, ompNow.t]}
            y={[Math.max(mpNow.residualNorm, FLOOR), Math.max(ompNow.residualNorm, FLOOR)]}
            emphasis
            size={9}
            live
          />
        </Plot>
      </div>
    </Figure>
  )
}
