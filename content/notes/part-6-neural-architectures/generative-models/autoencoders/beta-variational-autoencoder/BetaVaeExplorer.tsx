import { useMemo, useState } from 'react'
import {
  ControlRow,
  Curve,
  Figure,
  Plot,
  Plots,
  Points,
  Readout,
  Select,
  Slider,
  formatNumber,
  useAxis,
} from 'aifn-render'

const fmt = (v: number, digits = 3) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(digits))) : '—')

const BETA_PRESETS = [
  { value: '0.2', label: 'β = 0.2 (low regularization: sharp, entangled)' },
  { value: '1.0', label: 'β = 1.0 (standard VAE: balanced ELBO, entangled)' },
  { value: '4.0', label: 'β = 4.0 (β-VAE: axis-aligned disentanglement)' },
  { value: '12.0', label: 'β = 12.0 (over-regularized: posterior collapse)' },
]

export function BetaVaeExplorer() {
  const [beta, setBeta] = useState(4.0)
  const [traversalAxis, setTraversalAxis] = useState<'z1' | 'z2'>('z1')
  const [fixedCoord, setFixedCoord] = useState(0.0)

  // Disentanglement mechanics:
  // True independent factors: v1 in [-1.5, 1.5] (X position), v2 in [-1.5, 1.5] (Y position / scale)
  // For small beta, encoder rotates the factors by an entanglement angle phi.
  // As beta increases to ~4-6, the factorized prior forces phi -> 0 (disentanglement).
  // For excessive beta (> 8), capacity collapses and z2 shrinks to 0.
  const entanglementAngle = useMemo(() => {
    if (beta < 1.0) return Math.PI / 4 // 45 deg, fully coupled
    return (Math.PI / 4) / (1 + 0.8 * (beta - 1.0) ** 1.5)
  }, [beta])

  const z2Variance = useMemo(() => {
    // Dimension 2 collapses when beta > 8
    if (beta <= 4.0) return 1.0
    return Math.max(0.05, 1.0 - 0.12 * (beta - 4.0))
  }, [beta])

  // Synthetic population of 120 points with ground-truth factors v1, v2
  const data = useMemo(() => {
    const pts: { v1: number; v2: number; z1: number; z2: number }[] = []
    const cosP = Math.cos(entanglementAngle)
    const sinP = Math.sin(entanglementAngle)
    const n = 11

    for (let i = 0; i < n; i++) {
      const v1 = -1.5 + (3.0 * i) / (n - 1)
      for (let j = 0; j < n; j++) {
        const v2 = -1.5 + (3.0 * j) / (n - 1)
        // Entangled latent coordinates
        const z1 = v1 * cosP - v2 * sinP
        const z2 = (v1 * sinP + v2 * cosP) * Math.sqrt(z2Variance)
        pts.push({ v1, v2, z1, z2 })
      }
    }
    return pts
  }, [entanglementAngle, z2Variance])

  // Latent traversals along chosen axis
  const traversalSteps = 7
  const traversalValues = useMemo(
    () => Float64Array.from({ length: traversalSteps }, (_, i) => -1.8 + (3.6 * i) / (traversalSteps - 1)),
    [traversalSteps],
  )

  // Decoded output positions for each traversal step: (x_decoded, y_decoded)
  const decodedShapes = useMemo(() => {
    const cosP = Math.cos(entanglementAngle)
    const sinP = Math.sin(entanglementAngle)
    const invScale2 = 1 / Math.max(1e-4, Math.sqrt(z2Variance))

    return Array.from(traversalValues, (val) => {
      const z1 = traversalAxis === 'z1' ? val : fixedCoord
      const z2 = traversalAxis === 'z2' ? val : fixedCoord
      // Invert the decoder map to find reconstructed (v1, v2)
      const v1Rec = z1 * cosP + (z2 * invScale2) * sinP
      const v2Rec = -z1 * sinP + (z2 * invScale2) * cosP
      return { val, v1Rec, v2Rec }
    })
  }, [traversalAxis, fixedCoord, traversalValues, entanglementAngle, z2Variance])

  // Rate-Distortion curve values
  const rdPoints = useMemo(() => {
    const betas = [0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    return betas.map((b) => {
      const rate = 8.0 / (1 + 0.6 * b)
      const dist = 0.2 + 0.35 * b ** 0.8
      return { b, rate, dist }
    })
  }, [])

  const currentRate = 8.0 / (1 + 0.6 * beta)
  const currentDist = 0.2 + 0.35 * beta ** 0.8

  const z1Values = useMemo(() => Float64Array.from(data, (p) => p.z1), [data])
  const z2Values = useMemo(() => Float64Array.from(data, (p) => p.z2), [data])

  // Decoded shapes representation: sequence of generated points in output observation space
  const recXs = useMemo(() => Float64Array.from(decodedShapes, (d) => d.v1Rec), [decodedShapes])
  const recYs = useMemo(() => Float64Array.from(decodedShapes, (d) => d.v2Rec), [decodedShapes])

  // Axes
  const latentXAxis = useAxis({ label: 'latent coordinate z₁', range: [-2.6, 2.6] })
  const latentYAxis = useAxis({ label: 'latent coordinate z₂', range: [-2.6, 2.6] })

  const outputXAxis = useAxis({ label: 'output factor 1 (e.g. position x)', range: [-2.4, 2.4] })
  const outputYAxis = useAxis({ label: 'output factor 2 (e.g. position y / scale)', range: [-2.4, 2.4] })

  const rateAxis = useAxis({ label: 'rate R = KL(q ‖ p) (nats)', range: [0, 8.5] })
  const distAxis = useAxis({ label: 'distortion D = −E[log p(x | z)]', range: [0, 4.2] })

  const rdCurveR = useMemo(() => Float64Array.from(rdPoints, (p) => p.rate), [rdPoints])
  const rdCurveD = useMemo(() => Float64Array.from(rdPoints, (p) => p.dist), [rdPoints])

  const disentanglementScore = Math.max(0, Math.min(100, Math.round((1 - (4 * entanglementAngle) / Math.PI) * 100)))

  return (
    <Figure
      title="β-VAE: the rate–distortion trade-off and latent disentanglement"
      purpose="Visualizes how scaling the KL weight β forces factorized latent coordinates to align with ground-truth generative factors (disentanglement) at the cost of higher reconstruction distortion, and reveals posterior collapse under excessive β."
      controls={
        <>
          <ControlRow label="Model regularisation">
            <Slider
              label="KL weight β"
              value={beta}
              onChange={setBeta}
              min={0.2}
              max={12.0}
              step={0.2}
            />
            <Select
              label="Preset configuration"
              value={String(beta)}
              onChange={(v) => setBeta(Number(v))}
              options={BETA_PRESETS}
            />
          </ControlRow>
          <ControlRow label="Latent traversal inspection">
            <Select
              label="Traversal axis"
              value={traversalAxis}
              onChange={(v) => setTraversalAxis(v as 'z1' | 'z2')}
              options={[
                { value: 'z1', label: 'Vary z₁ across [−1.8, 1.8] (fixed z₂)' },
                { value: 'z2', label: 'Vary z₂ across [−1.8, 1.8] (fixed z₁)' },
              ]}
            />
            <Slider
              label="Fixed coordinate value"
              value={fixedCoord}
              onChange={setFixedCoord}
              min={-1.5}
              max={1.5}
              step={0.25}
            />
          </ControlRow>
        </>
      }
      readouts={{
        'rate–distortion trade-off': (
          <>
            <Readout label="rate R (KL divergence)" value={`${fmt(currentRate, 3)} nats`} />
            <Readout label="distortion D (recon error)" value={fmt(currentDist, 3)} />
            <Readout label="β-weighted objective (D + βR)" value={fmt(currentDist + beta * currentRate, 3)} />
          </>
        ),
        'disentanglement & capacity': (
          <>
            <Readout label="axis alignment score" value={`${disentanglementScore}%`} />
            <Readout
              label="latent coordinate z₂ status"
              value={z2Variance < 0.3 ? 'collapsed (inactive)' : `${fmt(z2Variance * 100, 1)}% variance`}
            />
            <Readout
              label="traversal effect"
              value={
                disentanglementScore > 75
                  ? `moves purely factor ${traversalAxis === 'z1' ? '1 (horizontal)' : '2 (vertical)'}`
                  : 'couples both factors (tilted traversal)'
              }
            />
          </>
        ),
      }}
      caption={`Interactive β-VAE latent traversals. Left: Latent coordinates (z₁, z₂) for 120 points generated from two independent ground-truth factors. At β = 1.0 (standard VAE), the coordinates are rotated by 45° relative to the axes: the two generative factors are entangled. At β = 4.0, the factorized isotropic Gaussian prior p(z) = N(0, I) penalizes cross-covariance, rotating the coordinates into orthogonal alignment with the axes (disentanglement). Center: Decoded observation space under latent traversal. In a disentangled model, sweeping z₁ changes only factor 1 along a clean horizontal line. At β = 1.0, sweeping z₁ slants across both factors. At β > 8.0, coordinate z₂ collapses to 0. Right: The rate–distortion curve D vs R, showing how β sets the tradeoff between compression and fidelity.`}
    >
      <Plots cols={3}>
        <Plot x={latentXAxis} y={latentYAxis} title="latent space q(z | x) (color = factor 1)">
          <Points
            name="latent codes"
            x={z1Values}
            y={z2Values}
            size={4}
          />
          {/* Traversal path in latent space */}
          <Curve
            name="traversal trajectory"
            x={
              traversalAxis === 'z1'
                ? Float64Array.from([-1.8, 1.8])
                : Float64Array.from([fixedCoord, fixedCoord])
            }
            y={
              traversalAxis === 'z1'
                ? Float64Array.from([fixedCoord, fixedCoord])
                : Float64Array.from([-1.8, 1.8])
            }
            emphasis
          />
        </Plot>
        <Plot x={outputXAxis} y={outputYAxis} title="decoded traversal in output space">
          <Curve name="neutral axes" x={[-2.4, 2.4]} y={[0, 0]} muted dashed thin />
          <Curve name="neutral vertical" x={[0, 0]} y={[-2.4, 2.4]} muted dashed thin />
          <Points
            name="decoded steps"
            x={recXs}
            y={recYs}
            size={8}
            emphasis
          />
          <Curve
            name="traversal line in output"
            x={recXs}
            y={recYs}
            emphasis
          />
        </Plot>
        <Plot x={rateAxis} y={distAxis} title="rate–distortion trade-off">
          <Curve name="R–D frontier" x={rdCurveR} y={rdCurveD} muted />
          <Points
            name={`operating point (β=${fmt(beta, 1)})`}
            x={[currentRate]}
            y={[currentDist]}
            size={9}
            emphasis
          />
        </Plot>
      </Plots>
    </Figure>
  )
}
