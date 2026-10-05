import { useMemo, useState } from 'react'
import { moons } from 'aifn-methods/data/synthetic'
import { dualDecision, smoSteps, type SmoState } from 'aifn-methods/learning/kernel-methods'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { grid2d } from 'aifn-compute/numerics/geometry'
import { rbf } from 'aifn-compute/learning/kernels'
import {
  Contours,
  ControlGroup,
  Curve,
  Figure,
  NumberSelector,
  Player,
  Plot,
  Plots,
  Points,
  Raster,
  Readout,
  formatNumber,
  useAxis,
} from 'aifn-render'

export function SmoOptimization() {
  const [C, setC] = useState(1)
  const [lengthscale, setLengthscale] = useState(0.6)
  const [step, setStep] = useState(0)

  // Subsample moons dataset for clear visualization of working pairs
  const data = useMemo(() => {
    const d = moons(stream('lab/classify/moons'), { n: 160, noise: 0.15 })
    const rows = toRows(d.x)
    // Every other point keeps the working pairs easily readable
    const keep = rows.map((_, i) => i).filter((i) => i % 2 === 0)
    return {
      x: fromData(Float64Array.from(keep.flatMap((i) => rows[i])), [keep.length, 2]),
      y: fromData(Float64Array.from(keep.map((i) => (toFlat(d.y!)[i] === 1 ? 1 : -1))), [keep.length]),
    }
  }, [])

  const kernel = useMemo(() => rbf({ lengthscale }), [lengthscale])

  const run = useMemo(
    () =>
      trace(smoSteps({ x: data.x, y: data.y, C, kernel }), {}, 400, {
        record: { dual: (s) => s.dualObjective, gap: (s) => s.gap },
      }),
    [data, C, kernel],
  )

  const k = Math.min(step, run.steps.length - 1)
  const state: SmoState = run.steps[k]
  const rows = useMemo(() => toRows(data.x), [data])
  const labels = useMemo(() => toFlat(data.y).map((v) => (v > 0 ? 1 : 0)), [data])
  const field = useMemo(() => grid2d([-1.6, 2.6], [-1.1, 1.6], 70), [])

  const f = useMemo(() => {
    const values = toFlat(dualDecision({ x: data.x, y: data.y, C, kernel }, state.alpha, state.bias)(field.points))
    const [ny, nx] = field.shape
    return Array.from({ length: ny }, (_, i) => values.slice(i * nx, (i + 1) * nx))
  }, [state, data, C, kernel, field])

  const [pi, pj] = state.pair
  const pts = useMemo(
    () => ({ x: rows.map((r) => r[0]), y: rows.map((r) => r[1]), fx: toFlat(field.x), fy: toFlat(field.y) }),
    [rows, field],
  )

  const sv = useMemo(() => {
    const idx = toFlat(state.alpha).flatMap((a, i) => (a > 1e-9 ? [i] : []))
    return { n: idx.length, x: idx.map((i) => rows[i][0]), y: idx.map((i) => rows[i][1]) }
  }, [state, rows])

  const pair = useMemo(
    () => (pi >= 0 ? { x: [rows[pi][0], rows[pj][0]], y: [rows[pi][1], rows[pj][1]] } : null),
    [pi, pj, rows],
  )

  const gap = useMemo(() => ({ x: Array.from(run.index), y: toFlat(run.series.gap) }), [run])
  const now = useMemo(() => ({ x: [gap.x[k]], y: [gap.y[k]] }), [gap, k])

  const x0 = useAxis({ label: 'x₀', range: [-1.6, 2.6], nice: false })
  const x1 = useAxis({ label: 'x₁', range: [-1.1, 1.6], nice: false, equal: x0 })
  const stepAxis = useAxis({ label: 'step', hold: 'initial', key: run })
  const gapAxis = useAxis({ label: 'KKT gap m(α) − M(α)', hold: 'initial', key: run })

  return (
    <Figure
      title="Sequential Minimal Optimisation (SMO) step by step"
      purpose="SMO optimizes the SVM dual by selecting a violating pair of multipliers at each step and solving the 2D subproblem analytically along yᵢαᵢ + yⱼαⱼ = const."
      defaultSize="L"
      controls={
        <>
          <ControlGroup title="Model Parameters">
            <NumberSelector
              label="penalty C"
              value={C}
              onChange={setC}
              min={0.05}
              max={10}
              step={0.5}
              spacing="log"
              points_per_decade={2}
              logTransform="value-is-real"
            />
            <NumberSelector
              label="RBF length-scale ℓ"
              value={lengthscale}
              onChange={setLengthscale}
              min={0.2}
              max={2}
              step={0.1}
              spacing="lin"
              points={0.1}
            />
          </ControlGroup>
          <ControlGroup title="Optimization Playback">
            <Player value={k} onChange={setStep} count={run.steps.length} label="SMO step" />
          </ControlGroup>
        </>
      }
      readouts={{
        'This Step': (
          <>
            <Readout label="step" value={`${k} of ${run.steps.length - 1}`} />
            <Readout label="working pair" value={pi < 0 ? '—' : `(${pi}, ${pj})`} />
            <Readout label="clipped at bound" value={pi < 0 ? '—' : state.clipped ? 'yes' : 'no'} />
          </>
        ),
        Convergence: (
          <>
            <Readout label="dual objective" value={formatNumber(state.dualObjective)} />
            <Readout label="KKT gap" value={formatNumber(state.gap)} />
            <Readout label="support vectors (α > 0)" value={sv.n} />
            <Readout label="bias b" value={formatNumber(state.bias)} />
          </>
        ),
      }}
      caption="Step through the run: the line highlights the working pair (xᵢ, xⱼ) updated at this step; ringed points have αᵢ > 0 (support vectors). The heatmap shows the continuous decision field f(x) with the solid black curve as the decision boundary f(x) = 0. Below, the KKT optimality gap m(α) − M(α) contracts steadily until it meets the tolerance 10⁻³, where optimization terminates."
    >
      <Plots rows={2} heights={[2, 1]}>
        <Plot x={x0} y={x1}>
          <Raster x={pts.fx} y={pts.fy} z={f} scale="diverging" range={[-2, 2]} valueLabel="f(x)" />
          <Contours x={pts.fx} y={pts.fy} z={f} levels={[0]} />
          <Points name="points" x={pts.x} y={pts.y} group={labels} groupNames={['y = −1', 'y = +1']} />
          <Points name="α > 0" x={sv.x} y={sv.y} emphasis />
          {pair && <Curve name="working pair" x={pair.x} y={pair.y} showPoints emphasis />}
        </Plot>
        <Plot x={stepAxis} y={gapAxis} legend={false}>
          <Curve name="KKT gap" x={gap.x} y={gap.y} />
          <Points name="now" x={now.x} y={now.y} emphasis />
        </Plot>
      </Plots>
    </Figure>
  )
}
