import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { cholesky2, eig2, type Mat2 } from 'aifn/numerics/linalg'

const H = 20
const CIRCLE = Array.from({ length: 121 }, (_, i) => (2 * Math.PI * i) / 120)

const mul = (a: Mat2, b: Mat2): Mat2 => [
  [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
  [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
]

export function VarImpulse() {
  const state = useFigureState({
    a11: float(0.5, { min: -1, max: 1.2, step: 0.05, label: 'a₁₁', format: (v) => v.toFixed(2) }),
    a12: float(0.2, { min: -1, max: 1, step: 0.05, label: 'a₁₂ (effect of y₂ on y₁)', format: (v) => v.toFixed(2) }),
    a21: float(0.3, { min: -1, max: 1, step: 0.05, label: 'a₂₁ (effect of y₁ on y₂)', format: (v) => v.toFixed(2) }),
    a22: float(0.4, { min: -1, max: 1.2, step: 0.05, label: 'a₂₂', format: (v) => v.toFixed(2) }),
    rho: slider(-0.9, 0.9, 0.5, { step: 0.05, label: 'error correlation ρ', format: (v) => v.toFixed(2) }),
    shock: choice<'0' | '1'>(
      [
        { value: '0', label: 'y₁' },
        { value: '1', label: 'y₂' },
      ],
      '0',
      { label: 'shock to' },
    ),
    kind: choice<'raw' | 'orth'>(
      [
        { value: 'orth', label: 'orthogonalised' },
        { value: 'raw', label: 'unit error' },
      ],
      'orth',
      { label: 'response' },
    ),
  })

  const result = useMemo(() => {
    const A: Mat2 = [
      [state.a11, state.a12],
      [state.a21, state.a22],
    ]
    const P = cholesky2([
      [1, state.rho],
      [state.rho, 1],
    ]) ?? [
      [1, 0],
      [0, 1],
    ]
    const j = Number(state.shock)
    // Phi_h = A^h for a VAR(1). The response to shock j is column j of Phi_h, or of Phi_h P when orthogonalised.
    let Phi: Mat2 = [
      [1, 0],
      [0, 1],
    ]
    const r1: number[] = []
    const r2: number[] = []
    for (let h = 0; h <= H; h++) {
      const M = state.kind === 'orth' ? mul(Phi, P) : Phi
      r1.push(M[0][j])
      r2.push(M[1][j])
      Phi = mul(A, Phi)
    }
    const e = eig2(A)
    const eigs: [number, number][] =
      e.kind === 'real'
        ? [
            [e.values[0], 0],
            [e.values[1], 0],
          ]
        : [
            [e.re, e.im],
            [e.re, -e.im],
          ]
    const radius = Math.max(...eigs.map(([x, y]) => Math.hypot(x, y)))
    return { r1, r2, eigs, radius }
  }, [state.a11, state.a12, state.a21, state.a22, state.rho, state.shock, state.kind])

  const charts = useMemo(() => {
    const h = Array.from({ length: H + 1 }, (_, i) => i)
    const irf = [
      { name: 'response of y₁', x: h, y: result.r1, slot: 0 },
      { name: 'response of y₂', x: h, y: result.r2, slot: 1 },
    ] as const
    const plane = [
      { name: 'unit circle', x: CIRCLE.map(Math.cos), y: CIRCLE.map(Math.sin), muted: true },
      {
        name: 'eigenvalues of A',
        x: result.eigs.map((p) => p[0]),
        y: result.eigs.map((p) => p[1]),
        emphasis: true,
      },
    ] as const
    return { irf, plane }
  }, [result])

  const stable = result.radius < 1
  const eigText =
    result.eigs[0][1] === 0
      ? `${formatNumber(result.eigs[0][0])}, ${formatNumber(result.eigs[1][0])}`
      : `${formatNumber(result.eigs[0][0])} ± ${formatNumber(Math.abs(result.eigs[0][1]))}i`

  const xAxis = useAxis({ label: 'horizon h', range: [0, H] })
  const yAxis = useAxis({ label: 'response', hold: 'union' })
  const xAxis2 = useAxis({ label: 'real part', range: [-2.5, 2.5] })
  const yAxis2 = useAxis({ label: 'imaginary part', range: [-2.5, 2.5], equal: xAxis2 })
  return (
    <Figure
      title="Impulse responses of a two-variable VAR(1)"
      state={state}
      caption="The sliders set the coefficient matrix A of y_t = A y_{t−1} + u_t and the correlation ρ between the two errors (each with variance 1). The left panel shows how y₁ and y₂ respond over h steps to a one-standard-deviation shock in the chosen equation. The orthogonalised response uses the Cholesky factor of the error covariance, with y₁ ordered first, so a shock to y₁ also moves y₂ at h = 0 by ρ. The right panel shows the eigenvalues of A. Inside the unit circle the responses die out; complex eigenvalues make them oscillate; an eigenvalue outside the circle makes them explode."

      readouts={
        <>
          <Readout label="eigenvalues of A" value={eigText} />
          <Readout label="largest modulus" value={formatNumber(result.radius)} />
          <Readout label="VAR" value={stable ? 'stable' : 'not stable'} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-[2fr_1fr] md:items-center">
        <Plot x={xAxis} y={yAxis} height={280}>
          <Curve {...charts.irf[0]} />
          <Curve {...charts.irf[1]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2}>
          <Curve {...charts.plane[0]} />
          <Points {...charts.plane[1]} />
        </Plot>
      </div>
    </Figure>
  )
}
