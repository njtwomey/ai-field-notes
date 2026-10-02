import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from 'aifn-render'
import { cholesky2, eig2, type Mat2 } from '@/lib/math/mat2'

const H = 20
const CIRCLE = Array.from({ length: 121 }, (_, i) => (2 * Math.PI * i) / 120)

const mul = (a: Mat2, b: Mat2): Mat2 => [
  [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
  [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
]

export function VarImpulse() {
  const a11 = useParam(0.5, { min: -1, max: 1.2, step: 0.05 })
  const a12 = useParam(0.2, { min: -1, max: 1, step: 0.05 })
  const a21 = useParam(0.3, { min: -1, max: 1, step: 0.05 })
  const a22 = useParam(0.4, { min: -1, max: 1.2, step: 0.05 })
  const rho = useParam(0.5, { min: -0.9, max: 0.9, step: 0.05 })
  const [shock, setShock] = useState<'0' | '1'>('0')
  const [kind, setKind] = useState<'raw' | 'orth'>('orth')

  const result = useMemo(() => {
    const A: Mat2 = [
      [a11.value, a12.value],
      [a21.value, a22.value],
    ]
    const P = cholesky2(1, rho.value, 1) ?? [
      [1, 0],
      [0, 1],
    ]
    const j = Number(shock)
    // Phi_h = A^h for a VAR(1). The response to shock j is column j of Phi_h, or of Phi_h P when orthogonalised.
    let Phi: Mat2 = [
      [1, 0],
      [0, 1],
    ]
    const r1: number[] = []
    const r2: number[] = []
    for (let h = 0; h <= H; h++) {
      const M = kind === 'orth' ? mul(Phi, P) : Phi
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
  }, [a11.value, a12.value, a21.value, a22.value, rho.value, shock, kind])

  const charts = useMemo(() => {
    const h = Array.from({ length: H + 1 }, (_, i) => i)
    const irf: XYSeries[] = [
      { name: 'response of y₁', type: 'line', x: h, y: result.r1, slot: 0 },
      { name: 'response of y₂', type: 'line', x: h, y: result.r2, slot: 1 },
    ]
    const plane: XYSeries[] = [
      { name: 'unit circle', type: 'line', x: CIRCLE.map(Math.cos), y: CIRCLE.map(Math.sin), muted: true },
      {
        name: 'eigenvalues of A',
        type: 'scatter',
        x: result.eigs.map((p) => p[0]),
        y: result.eigs.map((p) => p[1]),
        emphasis: true,
      },
    ]
    return { irf, plane }
  }, [result])

  const stable = result.radius < 1
  const eigText =
    result.eigs[0][1] === 0
      ? `${formatNumber(result.eigs[0][0])}, ${formatNumber(result.eigs[1][0])}`
      : `${formatNumber(result.eigs[0][0])} ± ${formatNumber(Math.abs(result.eigs[0][1]))}i`

  return (
    <Interactive
      title="Impulse responses of a two-variable VAR(1)"
      caption="The sliders set the coefficient matrix A of y_t = A y_{t−1} + u_t and the correlation ρ between the two errors (each with variance 1). The left panel shows how y₁ and y₂ respond over h steps to a one-standard-deviation shock in the chosen equation. The orthogonalised response uses the Cholesky factor of the error covariance, with y₁ ordered first, so a shock to y₁ also moves y₂ at h = 0 by ρ. The right panel shows the eigenvalues of A. Inside the unit circle the responses die out; complex eigenvalues make them oscillate; an eigenvalue outside the circle makes them explode."
      controls={
        <>
          <ParamSlider label="a₁₁" param={a11} format={(v) => v.toFixed(2)} />
          <ParamSlider label="a₁₂ (effect of y₂ on y₁)" param={a12} format={(v) => v.toFixed(2)} />
          <ParamSlider label="a₂₁ (effect of y₁ on y₂)" param={a21} format={(v) => v.toFixed(2)} />
          <ParamSlider label="a₂₂" param={a22} format={(v) => v.toFixed(2)} />
          <ParamSlider label="error correlation ρ" param={rho} format={(v) => v.toFixed(2)} />
          <ParamChoice
            label="shock to"
            value={shock}
            onChange={setShock}
            options={[
              { value: '0', label: 'y₁' },
              { value: '1', label: 'y₂' },
            ]}
          />
          <ParamChoice
            label="response"
            value={kind}
            onChange={setKind}
            options={[
              { value: 'orth', label: 'orthogonalised' },
              { value: 'raw', label: 'unit error' },
            ]}
          />
        </>
      }
      readout={
        <>
          <Readout label="eigenvalues of A" value={eigText} />
          <Readout label="largest modulus" value={formatNumber(result.radius)} />
          <Readout label="VAR" value={stable ? 'stable' : 'not stable'} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-[2fr_1fr] md:items-center">
        <XYChart series={charts.irf} xLabel="horizon h" yLabel="response" xRange={[0, H]} height={280} />
        <XYChart
          series={charts.plane}
          xLabel="real part"
          yLabel="imaginary part"
          xRange={[-2.5, 2.5]}
          yRange={[-2.5, 2.5]}
          equalAspect
        />
      </div>
    </Interactive>
  )
}
