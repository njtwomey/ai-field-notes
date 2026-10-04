import { useMemo, useState } from 'react'
import { fista, ista, proxL1 } from 'aifn/optim/proximal'
import { type IterateState, type ObjectiveFn } from 'aifn/optim'
import { child, normals, stream } from 'aifn/foundation/random'
import { toFlat } from 'aifn/foundation/tensor'
import { trace } from 'aifn/foundation/trace'
import { Bars, ControlRow, Curve, Figure, formatNumber, Plot, Plots, Readout, Slider, useAxis } from 'aifn-render'

export function ProximalExplorer() {
  const [logLambda, setLogLambda] = useState(-0.5)
  const lambda = 10 ** logLambda

  const problem = useMemo(() => {
    // Least squares ½‖Ax − b‖² with A 40×20 Gaussian and a sparse truth.
    const s = stream('optim/lasso-explorer')
    const A = toFlat(normals(child(s, 'A'), [40, 20]))
    const truth = Array.from({ length: 20 }, (_, j) => (j % 5 === 0 ? 3 - j / 5 : 0))
    const noise = toFlat(normals(child(s, 'noise'), 40, 0, 0.1))
    const b = Array.from({ length: 40 }, (_, i) => truth.reduce((acc, t, j) => acc + A[i * 20 + j] * t, 0) + noise[i])
    const f: ObjectiveFn = (x) => {
      const v = toFlat(x)
      const r = b.map((bi, i) => v.reduce((acc, vj, j) => acc + A[i * 20 + j] * vj, 0) - bi)
      const grad = v.map((_, j) => r.reduce((acc, ri, i) => acc + A[i * 20 + j] * ri, 0))
      return { value: 0.5 * r.reduce((acc, ri) => acc + ri * ri, 0), grad }
    }
    return { f, x0: new Array<number>(20).fill(0), truth }
  }, [])

  const runs = useMemo(() => {
    const g = proxL1(lambda)
    const options = { lr: 1, backtracking: true, tolerance: 0 }
    const record = { F: (s: IterateState) => s.value }
    const a = trace(ista(problem.f, g, options), { x0: problem.x0 }, 300, { record })
    const b = trace(fista(problem.f, g, options), { x0: problem.x0 }, 300, { record })
    const best = Math.min(...toFlat(a.series.F), ...toFlat(b.series.F))
    return { a, b, best }
  }, [problem, lambda])

  const lines = useMemo(
    () =>
      (
        [
          ['ISTA (O(1/k))', runs.a, 0],
          ['FISTA (O(1/k²))', runs.b, 1],
        ] as const
      ).map(([name, tr, slot]) => ({
        name,
        slot,
        x: Array.from(tr.index),
        y: toFlat(tr.series.F).map((v) => Math.max(v - runs.best, 1e-14)),
      })),
    [runs],
  )

  const finalA = runs.a.steps.at(-1)
  const finalB = runs.b.steps.at(-1)
  const coefsA = finalA ? toFlat(finalA.x) : []
  const coefsB = finalB ? toFlat(finalB.x) : []
  const nonZeroA = coefsA.filter((v) => Math.abs(v) > 1e-6).length
  const nonZeroB = coefsB.filter((v) => Math.abs(v) > 1e-6).length

  const featureIdx = Array.from({ length: 20 }, (_, j) => j)

  const xSteps = useAxis({ label: 'iteration k' })
  const yGap = useAxis({ label: 'F(x_k) − F*', log: true, range: [1e-14, 1e2] })
  const xFeat = useAxis({ label: 'coefficient index j', range: [-0.5, 19.5] })
  const yVal = useAxis({ label: 'coefficient value' })

  return (
    <Figure
      title="ISTA versus FISTA on Lasso (L1 proximal gradient)"
      purpose="Compare standard ISTA against accelerated FISTA for non-smooth L1-penalized optimization and observe sparsity patterns induced by the soft-thresholding proximal operator."
      caption="Left: objective gap F(x_k) - F* on a log scale over 300 iterations. ISTA converges at O(1/k); accelerated FISTA (with Nesterov extrapolation) attains O(1/k²). Right: recovered coefficients x_j against true values; increasing λ increases sparsity via soft-thresholding."
    >
      <ControlRow>
        <Slider
          label="Regularisation log₁₀ λ"
          value={logLambda}
          min={-2}
          max={1.5}
          step={0.1}
          onChange={setLogLambda}
        />
      </ControlRow>

      <div className="my-2 flex flex-wrap gap-4 font-mono text-xs text-muted-foreground">
        <Readout label="λ" value={formatNumber(lambda)} />
        <Readout label="non-zero (ISTA)" value={`${nonZeroA} / 20`} />
        <Readout label="non-zero (FISTA)" value={`${nonZeroB} / 20`} />
        <Readout label="true non-zero" value="4 / 20" />
      </div>

      <Plots cols={2}>
        <Plot x={xSteps} y={yGap} title="Convergence rate comparison (log gap)">
          {lines.map((l) => (
            <Curve key={l.name} name={l.name} x={l.x} y={l.y} slot={l.slot} />
          ))}
        </Plot>

        <Plot x={xFeat} y={yVal} title="Estimated coefficients vs Ground Truth">
          <Bars name="FISTA estimate" x={featureIdx} y={coefsB} slot={1} />
          <Curve name="Ground truth" x={featureIdx} y={problem.truth} slot={2} showPoints />
        </Plot>
      </Plots>
    </Figure>
  )
}
