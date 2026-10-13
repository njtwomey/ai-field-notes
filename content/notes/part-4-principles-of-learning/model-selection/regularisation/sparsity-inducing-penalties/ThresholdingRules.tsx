import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  Plot,
  Readout,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { normals, stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { thresholding, thresholdingPenalty, type ThresholdRule } from 'aifn-compute/optim/proximal'

const Z = Array.from({ length: 401 }, (_, i) => -5 + i * 0.025)
const THETA = Array.from({ length: 241 }, (_, i) => i * 0.025)
const TRUE = Array.from({ length: 61 }, (_, i) => i * 0.1)
const DRAWS = 4000
const RULES: { rule: ThresholdRule; name: string; slot: number }[] = [
  { rule: 'hard', name: 'hard (L⁰)', slot: 0 },
  { rule: 'soft', name: 'soft (L¹, lasso)', slot: 1 },
  { rule: 'scad', name: 'SCAD', slot: 2 },
  { rule: 'mcp', name: 'MCP', slot: 3 },
]
const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—')

/**
 * The thresholding rules of four sparsity penalties (`thresholding`), their penalties (`thresholdingPenalty`), and
 * their risk for z ~ N(θ, 1) by simulation with common noise, against least squares.
 */
export function ThresholdingRules() {
  const state = useFigureState({
    lambda: float(1, { min: 0.2, max: 3, step: 0.05, label: 'threshold λ' }),
    gammaMcp: float(3, { min: 1.2, max: 8, step: 0.1, label: 'MCP concavity γ' }),
    aScad: float(3.7, { min: 2.2, max: 8, step: 0.1, label: 'SCAD concavity a' }),
  })
  const { lambda, gammaMcp, aScad } = state
  const options = (rule: ThresholdRule) => ({
    rule,
    lambda,
    gamma: rule === 'mcp' ? gammaMcp : rule === 'scad' ? aScad : undefined,
  })
  const curves = useMemo(
    () =>
      RULES.map(({ rule }) => ({
        rule: toFlat(thresholding(Z, options(rule))),
        penalty: toFlat(thresholdingPenalty(THETA, options(rule))),
      })),
    [lambda, gammaMcp, aScad],
  )
  const risk = useComputed(
    () => {
      const noise = toFlat(normals(stream('thresholding-risk'), [DRAWS]))
      return RULES.map(({ rule }) =>
        TRUE.map((theta) => {
          const z = noise.map((e) => theta + e)
          const est = toFlat(thresholding(z, options(rule)))
          return est.reduce((a, v) => a + (v - theta) ** 2, 0) / DRAWS
        }),
      )
    },
    [lambda, gammaMcp, aScad],
    { mode: 'release' },
  )
  const at = (k: number, theta: number) => risk.value[k][Math.round(theta * 10)]

  const zx = useAxis({ label: 'least-squares coefficient z', range: [-5, 5], nice: false })
  const zy = useAxis({ label: 'estimate Θ(z)', range: [-5, 5], nice: false, equal: zx })
  const px = useAxis({ label: 'coefficient size |w|', range: [0, 6], nice: false })
  const py = useAxis({ label: 'penalty p(|w|)', hold: 'union', key: 'penalty' })
  const tx = useAxis({ label: 'true coefficient θ (noise σ = 1)', range: [0, 6], nice: false })
  const ty = useAxis({ label: 'risk E(Θ(z) − θ)²', hold: 'union', key: 'risk' })
  const dragLambda = (x: number) => state.set('lambda', Math.max(0.2, Math.min(3, Math.round(Math.abs(x) * 20) / 20)))

  return (
    <Figure
      title="Thresholding rules of sparsity penalties"
      state={state}
      hoverReadout={false}
      caption="Left: the estimate each penalty gives a coefficient whose least-squares value is z, with orthonormal features (the solution of ½(z − w)² + p(|w|)); the diagonal is least squares. Hard thresholding keeps or kills; soft thresholding kills small values and moves every other one towards zero by λ; SCAD and MCP kill small values, are continuous, and return to the diagonal beyond aλ and γλ. Middle: the penalties. The lasso's keeps rising; SCAD and MCP level off, so large coefficients pay no extra and are not shrunk. Right: the risk of each rule when z ~ N(θ, 1), estimated from 4,000 common noise draws per θ, against the risk 1 of least squares (dashed). Soft thresholding wins near θ = 0 and loses λ² for large θ; the others recover risk 1 for large θ, and every rule is worse than least squares near the threshold. Drag the vertical line on the left to move λ."
      readouts={
        <>
          {RULES.map(({ name }, k) => (
            <Readout
              key={name}
              label={`${name}: risk at θ = 0 / 2 / 5`}
              value={`${fmt(at(k, 0))} / ${fmt(at(k, 2))} / ${fmt(at(k, 5))}`}
            />
          ))}
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <Plot x={zx} y={zy} title="thresholding rule Θ(z)">
          <Curve name="least squares" x={[-5, 5]} y={[-5, 5]} muted dashed />
          {RULES.map(({ name, slot }, k) => (
            <Curve key={name} name={name} x={Z} y={curves[k].rule} slot={slot} />
          ))}
          <Handle kind="x" at={lambda} label="λ" onDrag={dragLambda} />
        </Plot>
        <Plot x={px} y={py} title="penalty p(|w|)">
          {RULES.map(({ name, slot }, k) => (
            <Curve key={name} name={name} x={THETA} y={curves[k].penalty} slot={slot} />
          ))}
        </Plot>
        <Plot x={tx} y={ty} title="risk against the true value">
          <Curve name="least squares" x={[0, 6]} y={[1, 1]} emphasis dashed />
          {RULES.map(({ name, slot }, k) => (
            <Curve key={name} name={name} x={TRUE} y={risk.value[k]} slot={slot} stale={risk.stale} />
          ))}
        </Plot>
      </div>
    </Figure>
  )
}
