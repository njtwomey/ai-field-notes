import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'
import { eigSym, type Vec2 } from '@/lib/math/mat2'

type Sym = [number, number, number] // [[p, q], [q, r]]

const A: Vec2 = [1, 1]
const H_A: Sym = [2, 1.5, 2]
const H_B: Sym = [1, 0, 0.2]
const DIAG_A: Sym = [H_A[0], 0, H_A[2]]
const LEVELS = [0.25, 1]
const LAMBDAS = Array.from({ length: 81 }, (_, i) => 10 ** (-2 + (4 * i) / 80))
const SLOTS = { a: 0, b: 1, full: 2, diag: 3 }

const quad = ([p, q, r]: Sym, c: Vec2, t: Vec2) => {
  const dx = t[0] - c[0]
  const dy = t[1] - c[1]
  return 0.5 * (p * dx * dx + 2 * q * dx * dy + r * dy * dy)
}

/** The level set ½(θ − c)ᵀH(θ − c) = level, as a closed polyline. */
function ellipse(h: Sym, c: Vec2, level: number): { x: number[]; y: number[] } {
  const { values, vectors } = eigSym(...h)
  const x: number[] = []
  const y: number[] = []
  for (let k = 0; k <= 96; k++) {
    const phi = (2 * Math.PI * k) / 96
    const s0 = Math.sqrt((2 * level) / values[0]) * Math.cos(phi)
    const s1 = Math.sqrt((2 * level) / values[1]) * Math.sin(phi)
    x.push(c[0] + s0 * vectors[0][0] + s1 * vectors[1][0])
    y.push(c[1] + s0 * vectors[0][1] + s1 * vectors[1][1])
  }
  return { x, y }
}

/** Minimiser of L_B(θ) + λ/2 (θ − a)ᵀF(θ − a): solve (H_B + λF)θ = H_B b + λF a. */
function ewc(f: Sym, b: Vec2, lambda: number): Vec2 {
  const p = H_B[0] + lambda * f[0]
  const q = H_B[1] + lambda * f[1]
  const r = H_B[2] + lambda * f[2]
  const rhs: Vec2 = [
    H_B[0] * b[0] + H_B[1] * b[1] + lambda * (f[0] * A[0] + f[1] * A[1]),
    H_B[1] * b[0] + H_B[2] * b[1] + lambda * (f[1] * A[0] + f[2] * A[1]),
  ]
  const d = p * r - q * q
  return [(r * rhs[0] - q * rhs[1]) / d, (p * rhs[1] - q * rhs[0]) / d]
}

/** Two quadratic tasks; EWC with the exact curvature of task A against its diagonal approximation. */
export function EwcFigure() {
  const logLambda = useParam(0, { min: -2, max: 2, step: 0.05 })
  const b1 = useParam(-1, { min: -2.5, max: 2.5, step: 0.05 })
  const b2 = useParam(0.5, { min: -2.5, max: 2.5, step: 0.05 })
  const lambda = 10 ** logLambda.value
  const b: Vec2 = useMemo(() => [b1.value, b2.value], [b1.value, b2.value])

  const paths = useMemo(() => {
    const full = LAMBDAS.map((l) => ewc(H_A, b, l))
    const diag = LAMBDAS.map((l) => ewc(DIAG_A, b, l))
    return { full, diag }
  }, [b])
  const full = ewc(H_A, b, lambda)
  const diag = ewc(DIAG_A, b, lambda)

  const series: XYSeries[] = [
    ...LEVELS.map((lv): XYSeries => {
      const e = ellipse(H_A, A, lv)
      return { name: 'task A loss contours', type: 'line', x: e.x, y: e.y, slot: SLOTS.a }
    }),
    ...LEVELS.map((lv): XYSeries => {
      const e = ellipse(H_B, b, lv)
      return { name: 'task B loss contours', type: 'line', x: e.x, y: e.y, slot: SLOTS.b }
    }),
    {
      name: 'EWC path, exact curvature',
      type: 'line',
      x: paths.full.map((p) => p[0]),
      y: paths.full.map((p) => p[1]),
      slot: SLOTS.full,
      dashed: true,
    },
    {
      name: 'EWC path, diagonal',
      type: 'line',
      x: paths.diag.map((p) => p[0]),
      y: paths.diag.map((p) => p[1]),
      slot: SLOTS.diag,
      dashed: true,
    },
    { name: 'EWC solution, exact', type: 'scatter', x: [full[0]], y: [full[1]], slot: SLOTS.full },
    { name: 'EWC solution, diagonal', type: 'scatter', x: [diag[0]], y: [diag[1]], slot: SLOTS.diag },
    { name: 'task optima', type: 'scatter', x: [A[0], b[0]], y: [A[1], b[1]], emphasis: true },
  ]

  return (
    <Interactive
      title="Elastic weight consolidation on two quadratic tasks"
      caption="Task A has its minimum at (1, 1) with correlated curvature; task B's minimum can be dragged. Training on B alone would move the weights to B's minimum and raise A's loss. EWC adds the penalty λ/2 (θ − θ_A)ᵀF(θ − θ_A). With the exact curvature of A (the Laplace approximation, exact for a quadratic) and λ = 1, the solution is the joint minimiser of both losses. The diagonal approximation ignores the correlation between the two weights, so at λ = 1 it misses the joint minimiser: with the initial optima the summed loss is 1.40 against 1.08. Large λ pins the weights to θ_A; small λ lets them go to B's minimum."
      controls={
        <>
          <ParamSlider label="penalty strength λ" param={logLambda} format={(v) => formatNumber(10 ** v)} />
          <ParamSlider label="task B optimum, θ₁" param={b1} />
          <ParamSlider label="task B optimum, θ₂" param={b2} />
        </>
      }
      readout={
        <>
          <Readout label="exact: L_A" value={formatNumber(quad(H_A, A, full))} />
          <Readout label="L_B" value={formatNumber(quad(H_B, b, full))} />
          <Readout label="diagonal: L_A" value={formatNumber(quad(H_A, A, diag))} />
          <Readout label="L_B" value={formatNumber(quad(H_B, b, diag))} />
        </>
      }
    >
      <XYChart
        series={series}
        xLabel="θ₁"
        yLabel="θ₂"
        xRange={[-3, 3]}
        yRange={[-3, 3]}
        equalAspect
        height={380}
        handles={[
          {
            kind: 'point',
            at: b,
            label: 'task B optimum',
            onDrag: ([x, y]) => {
              b1.set(x)
              b2.set(y)
            },
          },
        ]}
      />
    </Interactive>
  )
}
