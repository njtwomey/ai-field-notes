import { useMemo, useState } from 'react'
import { MathText } from '@/components/content/MathText'
import { Heatmap, Interactive, ParamChoice, ParamSlider, ParamSwitch, Readout, formatNumber } from '@/components/viz'
import type { HeatmapOverlay } from '@/components/viz'
import { linspace } from '@/lib/math'
import { logGamma } from '@/lib/math/special'

type View = 'simplex' | 'softmax'

const N = 64
const SIMPLEX_AXIS = linspace(0.5 / N, 1 - 0.5 / N, N)
const SOFTMAX_AXIS = linspace(-6, 6, N)
// Integration grid for the total variation distance, in the softmax basis where both densities are smooth.
const TV_AXIS = linspace(-20, 20, 201)
const TV_CELL = (TV_AXIS[1] - TV_AXIS[0]) ** 2
const EDGE: HeatmapOverlay[] = [{ name: 'simplex edge', type: 'line', x: [0, 1], y: [1, 0], emphasis: true }]
const NO_OVERLAY: HeatmapOverlay[] = []

type Gauss2 = { m: [number, number]; c11: number; c12: number; c22: number }

/**
 * Laplace approximation in the softmax basis θ = softmax(h), with h constrained to sum to zero (MacKay 1998;
 * Hennig et al. 2012): mean log α − mean(log α), covariance P diag(1/α) P with P the centring matrix. ProdLDA keeps only
 * the diagonal. The 2-D coordinates y = (h₁ − h₃, h₂ − h₃) are Gaussian with covariance B Σ Bᵀ, B = [[1,0,−1],[0,1,−1]].
 */
function laplace(alpha: number[], diagonal: boolean) {
  const K = 3
  const logs = alpha.map(Math.log)
  const meanLog = logs.reduce((a, b) => a + b, 0) / K
  const mu = logs.map((l) => l - meanLog)
  const inv = alpha.map((a) => 1 / a)
  const s = inv.reduce((a, b) => a + b, 0)
  // Σ = P D P: Σᵢⱼ = δᵢⱼ/αᵢ − (1/αᵢ + 1/αⱼ)/K + Σₗ(1/αₗ)/K².
  const full = (i: number, j: number) => (i === j ? inv[i] : 0) - (inv[i] + inv[j]) / K + s / (K * K)
  const S = (i: number, j: number) => (diagonal && i !== j ? 0 : full(i, j))
  const cov = (a: [number, number, number], b: [number, number, number]) => {
    let t = 0
    for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) t += a[i] * b[j] * S(i, j)
    return t
  }
  const b1: [number, number, number] = [1, 0, -1]
  const b2: [number, number, number] = [0, 1, -1]
  const g: Gauss2 = { m: [mu[0] - mu[2], mu[1] - mu[2]], c11: cov(b1, b1), c12: cov(b1, b2), c22: cov(b2, b2) }
  return { mu, diag: [0, 1, 2].map((i) => S(i, i)), g }
}

function gaussLogPdf(g: Gauss2, y1: number, y2: number): number {
  const det = g.c11 * g.c22 - g.c12 * g.c12
  const d1 = y1 - g.m[0]
  const d2 = y2 - g.m[1]
  const q = (g.c22 * d1 * d1 - 2 * g.c12 * d1 * d2 + g.c11 * d2 * d2) / det
  return -Math.log(2 * Math.PI) - 0.5 * Math.log(det) - 0.5 * q
}

/** log θ for the softmax of (y₁, y₂, 0). */
function logSoftmax(y1: number, y2: number): [number, number, number] {
  const m = Math.max(y1, y2, 0)
  const lse = m + Math.log(Math.exp(y1 - m) + Math.exp(y2 - m) + Math.exp(-m))
  return [y1 - lse, y2 - lse, -lse]
}

/** Laplace approximation of a three-topic Dirichlet by a logistic normal, on the simplex or in the softmax basis. */
export function LaplaceDirichlet() {
  const [a1, setA1] = useState(2)
  const [a2, setA2] = useState(1)
  const [a3, setA3] = useState(1)
  const [view, setView] = useState<View>('simplex')
  const [diagonal, setDiagonal] = useState(true)

  const { dirichlet, normal, range, tv, mu, diag } = useMemo(() => {
    const alpha = [a1, a2, a3]
    const logNorm = logGamma(a1 + a2 + a3) - logGamma(a1) - logGamma(a2) - logGamma(a3)
    const { mu, diag, g } = laplace(alpha, diagonal)
    // Densities in the softmax basis: the Dirichlet picks up the Jacobian θ₁θ₂θ₃, so its exponents are α, not α − 1.
    const dirY = (y1: number, y2: number) => {
      const l = logSoftmax(y1, y2)
      return Math.exp(logNorm + a1 * l[0] + a2 * l[1] + a3 * l[2])
    }
    const lnY = (y1: number, y2: number) => Math.exp(gaussLogPdf(g, y1, y2))
    let dirichlet: number[][]
    let normal: number[][]
    if (view === 'softmax') {
      dirichlet = SOFTMAX_AXIS.map((y2) => SOFTMAX_AXIS.map((y1) => dirY(y1, y2)))
      normal = SOFTMAX_AXIS.map((y2) => SOFTMAX_AXIS.map((y1) => lnY(y1, y2)))
    } else {
      // Densities over (θ₁, θ₂); the logistic normal divides by the Jacobian θ₁θ₂θ₃ of y = log(θ₁:₂ / θ₃).
      const cell = (t1: number, t2: number, f: (t3: number) => number) => {
        const t3 = 1 - t1 - t2
        return t3 <= 0 ? 0 : f(t3)
      }
      dirichlet = SIMPLEX_AXIS.map((t2) =>
        SIMPLEX_AXIS.map((t1) =>
          cell(t1, t2, (t3) =>
            Math.exp(logNorm + (a1 - 1) * Math.log(t1) + (a2 - 1) * Math.log(t2) + (a3 - 1) * Math.log(t3)),
          ),
        ),
      )
      normal = SIMPLEX_AXIS.map((t2) =>
        SIMPLEX_AXIS.map((t1) =>
          cell(t1, t2, (t3) => Math.exp(gaussLogPdf(g, Math.log(t1 / t3), Math.log(t2 / t3))) / (t1 * t2 * t3)),
        ),
      )
    }
    // Share one colour scale, capped at the 98th percentile: Dirichlet densities with α < 1 are infinite at the corners.
    const values = [...dirichlet.flat(), ...normal.flat()].filter((v) => v > 0).sort((p, q) => p - q)
    const hi = values[Math.floor(0.98 * (values.length - 1))] ?? 1
    let tv = 0
    for (const y2 of TV_AXIS) for (const y1 of TV_AXIS) tv += Math.abs(dirY(y1, y2) - lnY(y1, y2))
    return { dirichlet, normal, range: [0, hi] as [number, number], tv: 0.5 * tv * TV_CELL, mu, diag }
  }, [a1, a2, a3, view, diagonal])

  const axis = view === 'simplex' ? SIMPLEX_AXIS : SOFTMAX_AXIS
  const xLabel = view === 'simplex' ? 'θ₁' : 'h₁ − h₃'
  const yLabel = view === 'simplex' ? 'θ₂' : 'h₂ − h₃'
  const overlay = view === 'simplex' ? EDGE : NO_OVERLAY
  const fmt = (xs: number[]) => `(${xs.map((x) => formatNumber(x)).join(', ')})`

  return (
    <Interactive
      title="The Laplace approximation of a Dirichlet"
      caption={
        <MathText text="Left: the density of $\Dir(\alphavec)$ for three topics. Right: its Laplace approximation, a logistic normal whose Gaussian lives in the softmax basis $\thetavec = \operatorname{softmax}(\hvec)$. On the simplex, $\theta_3 = 1 - \theta_1 - \theta_2$ and the blank triangle beyond the edge is outside the simplex. In the softmax basis the Dirichlet is always unimodal, which is why the approximation is taken there. With the diagonal switched off, the covariance is the full $\Pmat\,\diag(1/\alphavec)\,\Pmat$; ProdLDA keeps only its diagonal. Colours share one scale, capped at the 98th percentile." />
      }
      controls={
        <>
          <ParamSlider label="α₁" value={a1} onChange={setA1} min={0.3} max={10} step={0.1} />
          <ParamSlider label="α₂" value={a2} onChange={setA2} min={0.3} max={10} step={0.1} />
          <ParamSlider label="α₃" value={a3} onChange={setA3} min={0.3} max={10} step={0.1} />
          <ParamChoice
            label="coordinates"
            value={view}
            onChange={setView}
            options={[
              { value: 'simplex', label: 'simplex θ' },
              { value: 'softmax', label: 'softmax basis h' },
            ]}
          />
          <ParamSwitch label="diagonal covariance (ProdLDA)" checked={diagonal} onChange={setDiagonal} />
        </>
      }
      readout={
        <>
          <Readout label="μ₀" value={fmt(mu)} />
          <Readout label="diag Σ₀" value={fmt(diag)} />
          <Readout label="total variation distance" value={formatNumber(tv)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <div>
          <p className="mb-1 text-center text-xs text-muted-foreground">Dirichlet</p>
          <Heatmap
            x={axis}
            y={axis}
            z={dirichlet}
            range={range}
            overlay={overlay}
            xLabel={xLabel}
            yLabel={yLabel}
            valueLabel="density"
            height={300}
            ariaLabel="Dirichlet density"
          />
        </div>
        <div>
          <p className="mb-1 text-center text-xs text-muted-foreground">Laplace approximation (logistic normal)</p>
          <Heatmap
            x={axis}
            y={axis}
            z={normal}
            range={range}
            overlay={overlay}
            xLabel={xLabel}
            yLabel={yLabel}
            valueLabel="density"
            height={300}
            ariaLabel="Logistic-normal density of the Laplace approximation"
          />
        </div>
      </div>
    </Interactive>
  )
}
