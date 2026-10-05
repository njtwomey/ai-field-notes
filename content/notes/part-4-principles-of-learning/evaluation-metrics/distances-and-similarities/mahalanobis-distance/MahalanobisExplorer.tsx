import { useCallback, useMemo, useState } from 'react'
import {
  ControlRow,
  Curve,
  Figure,
  Handle,
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

// Deterministic pseudo-random standard normal numbers using Box-Muller
const STANDARD_NORMAL_SAMPLES: [number, number][] = (() => {
  const samples: [number, number][] = []
  let seed = 42
  const rand = () => {
    seed = (seed * 16807) % 2147483647
    return (seed - 1) / 2147483646
  }
  for (let i = 0; i < 90; i++) {
    const u1 = Math.max(1e-6, rand())
    const u2 = rand()
    const r = Math.sqrt(-2.0 * Math.log(u1))
    const theta = 2.0 * Math.PI * u2
    samples.push([r * Math.cos(theta), r * Math.sin(theta)])
  }
  return samples
})()

/** The distribution's mean: fixed at the origin, so it is a constant, not state. */
const MU: readonly [number, number] = [0, 0]

export function MahalanobisExplorer() {
  const [sigmaX, setSigmaX] = useState(1.8)
  const [sigmaY, setSigmaY] = useState(0.9)
  const [rho, setRho] = useState(0.8)
  const [pointA, setPointA] = useState<[number, number]>([2.2, 1.0])
  const [pointB, setPointB] = useState<[number, number]>([0.2, 1.4])

  // Center is fixed at origin for canonical clarity

  // Covariance decomposition
  const { lambda1, lambda2, v1, v2, invCov, sqrtCov, invSqrtCov } = useMemo(() => {
    const sxx = sigmaX * sigmaX
    const syy = sigmaY * sigmaY
    const sxy = rho * sigmaX * sigmaY
    const det = sxx * syy - sxy * sxy
    const safeDet = Math.max(1e-6, det)

    // Eigenvalues of 2x2 symmetric matrix
    const trace = sxx + syy
    const disc = Math.sqrt(Math.max(0, (sxx - syy) ** 2 + 4 * sxy * sxy))
    const l1 = (trace + disc) / 2
    const l2 = Math.max(1e-6, (trace - disc) / 2)

    // Principal angle
    const angle = 0.5 * Math.atan2(2 * sxy, sxx - syy)
    const vec1: [number, number] = [Math.cos(angle), Math.sin(angle)]
    const vec2: [number, number] = [-Math.sin(angle), Math.cos(angle)]

    // Inverse covariance matrix
    const invSxx = syy / safeDet
    const invSyy = sxx / safeDet
    const invSxy = -sxy / safeDet

    // Matrix square root Σ^(1/2) = V diag(√l1, √l2) V^T
    const r1 = Math.sqrt(l1)
    const r2 = Math.sqrt(l2)
    const sqrtSxx = r1 * vec1[0] * vec1[0] + r2 * vec2[0] * vec2[0]
    const sqrtSxy = r1 * vec1[0] * vec1[1] + r2 * vec2[0] * vec2[1]
    const sqrtSyy = r1 * vec1[1] * vec1[1] + r2 * vec2[1] * vec2[1]

    // Whitening matrix Σ^(-1/2) = V diag(1/√l1, 1/√l2) V^T
    const ir1 = 1 / r1
    const ir2 = 1 / r2
    const winvSxx = ir1 * vec1[0] * vec1[0] + ir2 * vec2[0] * vec2[0]
    const winvSxy = ir1 * vec1[0] * vec1[1] + ir2 * vec2[0] * vec2[1]
    const winvSyy = ir1 * vec1[1] * vec1[1] + ir2 * vec2[1] * vec2[1]

    return {
      lambda1: l1,
      lambda2: l2,
      v1: vec1,
      v2: vec2,
      invCov: { sxx: invSxx, syy: invSyy, sxy: invSxy },
      sqrtCov: { sxx: sqrtSxx, sxy: sqrtSxy, syy: sqrtSyy },
      invSqrtCov: { sxx: winvSxx, sxy: winvSxy, syy: winvSyy },
    }
  }, [sigmaX, sigmaY, rho])

  // Distance calculator helper
  const calcDistances = useCallback(
    (pt: [number, number]) => {
      const dx = pt[0] - MU[0]
      const dy = pt[1] - MU[1]
      const euc = Math.hypot(dx, dy)
      const mahalanobisSq = dx * (invCov.sxx * dx + invCov.sxy * dy) + dy * (invCov.sxy * dx + invCov.syy * dy)
      const mah = Math.sqrt(Math.max(0, mahalanobisSq))

      // Whitened coordinate z = Σ^(-1/2) d
      const zx = invSqrtCov.sxx * dx + invSqrtCov.sxy * dy
      const zy = invSqrtCov.sxy * dx + invSqrtCov.syy * dy

      return { euc, mah, z: [zx, zy] as [number, number] }
    },
    [invCov, invSqrtCov],
  )

  const distA = useMemo(() => calcDistances(pointA), [pointA, calcDistances])
  const distB = useMemo(() => calcDistances(pointB), [pointB, calcDistances])

  // Inversion detection
  const hasInversion =
    (distA.euc > distB.euc && distA.mah < distB.mah) || (distA.euc < distB.euc && distA.mah > distB.mah)

  // Ellipse generation for data space (1, 2, 3 standard deviations)
  const ellipsePoints = useMemo(() => {
    const steps = 90
    const rings = [1, 2, 3]
    return rings.map((k) => {
      const xs = new Float64Array(steps)
      const ys = new Float64Array(steps)
      for (let i = 0; i < steps; i++) {
        const phi = (2 * Math.PI * i) / (steps - 1)
        const c1 = k * Math.sqrt(lambda1) * Math.cos(phi)
        const c2 = k * Math.sqrt(lambda2) * Math.sin(phi)
        xs[i] = MU[0] + c1 * v1[0] + c2 * v2[0]
        ys[i] = MU[1] + c1 * v1[1] + c2 * v2[1]
      }
      return { k, xs, ys }
    })
  }, [lambda1, lambda2, v1, v2])

  // Circular contours for whitened space (r = 1, 2, 3)
  const circlePoints = useMemo(() => {
    const steps = 90
    const rings = [1, 2, 3]
    return rings.map((k) => {
      const xs = new Float64Array(steps)
      const ys = new Float64Array(steps)
      for (let i = 0; i < steps; i++) {
        const phi = (2 * Math.PI * i) / (steps - 1)
        xs[i] = k * Math.cos(phi)
        ys[i] = k * Math.sin(phi)
      }
      return { k, xs, ys }
    })
  }, [])

  // Sample cloud in data space: x = μ + Σ^(1/2) z
  const dataCloud = useMemo(() => {
    const xs = new Float64Array(STANDARD_NORMAL_SAMPLES.length)
    const ys = new Float64Array(STANDARD_NORMAL_SAMPLES.length)
    for (let i = 0; i < STANDARD_NORMAL_SAMPLES.length; i++) {
      const [zx, zy] = STANDARD_NORMAL_SAMPLES[i]
      xs[i] = MU[0] + (sqrtCov.sxx * zx + sqrtCov.sxy * zy)
      ys[i] = MU[1] + (sqrtCov.sxy * zx + sqrtCov.syy * zy)
    }
    return { xs, ys }
  }, [sqrtCov])

  // Sample cloud in whitened space: z
  const whitenedCloud = useMemo(() => {
    const xs = new Float64Array(STANDARD_NORMAL_SAMPLES.length)
    const ys = new Float64Array(STANDARD_NORMAL_SAMPLES.length)
    for (let i = 0; i < STANDARD_NORMAL_SAMPLES.length; i++) {
      xs[i] = STANDARD_NORMAL_SAMPLES[i][0]
      ys[i] = STANDARD_NORMAL_SAMPLES[i][1]
    }
    return { xs, ys }
  }, [])

  // Axes setups
  const dataAxisX = useAxis({ range: [-4.2, 4.2], label: 'x₁' })
  const dataAxisY = useAxis({ range: [-4.2, 4.2], label: 'x₂' })
  const whiteAxisX = useAxis({ range: [-4.2, 4.2], label: 'z₁ (whitened)' })
  const whiteAxisY = useAxis({ range: [-4.2, 4.2], label: 'z₂ (whitened)' })

  return (
    <Figure
      title="Mahalanobis distance and statistical whitening"
      purpose="Visualise how Mahalanobis distance measures deviation in units of standard deviation along covariance axes, and how the whitening transformation z = Σ^(-1/2)(x - μ) maps ellipses to isotropic Euclidean circles."
      controls={
        <>
          <ControlRow>
            <Slider label="σ_x (scale 1)" value={sigmaX} min={0.5} max={2.5} step={0.1} onChange={setSigmaX} />
            <Slider label="σ_y (scale 2)" value={sigmaY} min={0.5} max={2.5} step={0.1} onChange={setSigmaY} />
            <Slider label="correlation ρ" value={rho} min={-0.92} max={0.92} step={0.04} onChange={setRho} />
          </ControlRow>
          <ControlRow>
            <Select
              label="preset geometry"
              options={[
                { value: 'correlated', label: 'High positive correlation (ρ = 0.85)' },
                { value: 'negative', label: 'Strong negative correlation (ρ = -0.80)' },
                { value: 'anisotropic', label: 'Uncorrelated anisotropic (σ_x ≫ σ_y)' },
                { value: 'isotropic', label: 'Standard isotropic (spherical)' },
              ]}
              value={
                rho > 0.6
                  ? 'correlated'
                  : rho < -0.6
                    ? 'negative'
                    : Math.abs(rho) < 0.1 && Math.abs(sigmaX - sigmaY) > 0.8
                      ? 'anisotropic'
                      : 'isotropic'
              }
              onChange={(val) => {
                if (val === 'correlated') {
                  setSigmaX(2.0)
                  setSigmaY(1.0)
                  setRho(0.85)
                  setPointA([2.3, 1.1])
                  setPointB([0.3, 1.4])
                } else if (val === 'negative') {
                  setSigmaX(1.8)
                  setSigmaY(1.2)
                  setRho(-0.8)
                  setPointA([2.0, -1.2])
                  setPointB([0.4, 1.4])
                } else if (val === 'anisotropic') {
                  setSigmaX(2.4)
                  setSigmaY(0.7)
                  setRho(0.0)
                  setPointA([2.2, 0.2])
                  setPointB([0.2, 1.2])
                } else {
                  setSigmaX(1.5)
                  setSigmaY(1.5)
                  setRho(0.0)
                  setPointA([1.8, 0.4])
                  setPointB([0.4, 1.5])
                }
              }}
            />
            <Readout label="eigenvalues (λ₁, λ₂)" value={`(${fmt(lambda1, 2)}, ${fmt(lambda2, 2)})`} />
            <Readout label="ranking status" value={hasInversion ? 'Inverted! d_E ≠ D_M order' : 'Consistent ranking'} />
          </ControlRow>
        </>
      }
      readouts={{
        'Point A measurements': (
          <>
            <Readout label="Euclidean distance d_E(μ, A)" value={fmt(distA.euc, 3)} />
            <Readout label="Mahalanobis distance D_M(μ, A)" value={fmt(distA.mah, 3)} />
            <Readout label="whitened coordinate z_A" value={`(${fmt(distA.z[0], 2)}, ${fmt(distA.z[1], 2)})`} />
          </>
        ),
        'Point B measurements': (
          <>
            <Readout label="Euclidean distance d_E(μ, B)" value={fmt(distB.euc, 3)} />
            <Readout label="Mahalanobis distance D_M(μ, B)" value={fmt(distB.mah, 3)} />
            <Readout label="whitened coordinate z_B" value={`(${fmt(distB.z[0], 2)}, ${fmt(distB.z[1], 2)})`} />
          </>
        ),
        'Nearest neighbour status': (
          <>
            <Readout label="Euclidean nearest" value={distA.euc < distB.euc ? 'Point A' : 'Point B'} />
            <Readout label="Mahalanobis nearest" value={distA.mah < distB.mah ? 'Point A' : 'Point B'} />
            <Readout
              label="geometric insight"
              value={
                hasInversion
                  ? 'Inversion: high variance axis dampens Mahalanobis distance'
                  : 'Consistent: both metrics agree on nearest point'
              }
            />
          </>
        ),
      }}
      caption="Left: Data space with covariance ellipse contours at 1σ, 2σ, and 3σ. Draggable test points A and B illustrate how Euclidean distance can severely misjudge likelihood under correlated or anisotropic variance. Right: Whitened space z = Σ^(-1/2)(x - μ). Here, covariance is rotated and normalised to the identity matrix, turning ellipses into standard concentric circles where Euclidean distance exactly equals Mahalanobis distance."
    >
      <Plots cols={2}>
        <Plot x={dataAxisX} y={dataAxisY} title="Original data space (correlated)">
          {/* Coordinate axes */}
          <Curve name="x₁ axis" x={[-4, 4]} y={[0, 0]} muted dashed thin />
          <Curve name="x₂ axis" x={[0, 0]} y={[-4, 4]} muted dashed thin />

          {/* Covariance ellipses */}
          {ellipsePoints.map((el) => (
            <Curve key={el.k} name={`${el.k}σ contour`} x={el.xs} y={el.ys} dashed={el.k > 1} thin />
          ))}

          {/* Principal eigenvector rays */}
          <Curve
            name="Major axis"
            x={[-2.5 * Math.sqrt(lambda1) * v1[0], 2.5 * Math.sqrt(lambda1) * v1[0]]}
            y={[-2.5 * Math.sqrt(lambda1) * v1[1], 2.5 * Math.sqrt(lambda1) * v1[1]]}
            muted
            thin
          />
          <Curve
            name="Minor axis"
            x={[-2.5 * Math.sqrt(lambda2) * v2[0], 2.5 * Math.sqrt(lambda2) * v2[0]]}
            y={[-2.5 * Math.sqrt(lambda2) * v2[1], 2.5 * Math.sqrt(lambda2) * v2[1]]}
            muted
            thin
          />

          {/* Sample scatter cloud */}
          <Points name="Sample cloud" x={dataCloud.xs} y={dataCloud.ys} size={4} muted />

          {/* Center μ */}
          <Points name="Mean μ" x={[MU[0]]} y={[MU[1]]} size={7} />

          {/* Vectors to A and B */}
          <Curve name="Chord to A" x={[MU[0], pointA[0]]} y={[MU[1], pointA[1]]} thin />
          <Curve name="Chord to B" x={[MU[0], pointB[0]]} y={[MU[1], pointB[1]]} thin dashed />

          {/* Point A */}
          <Points name="Point A" x={[pointA[0]]} y={[pointA[1]]} size={10} emphasis />
          <Handle
            kind="point"
            at={pointA}
            onDrag={([x, y]) => setPointA([x, y])}
            label={`A: d_E=${fmt(distA.euc, 2)}, D_M=${fmt(distA.mah, 2)}`}
          />

          {/* Point B */}
          <Points name="Point B" x={[pointB[0]]} y={[pointB[1]]} size={10} />
          <Handle
            kind="point"
            at={pointB}
            onDrag={([x, y]) => setPointB([x, y])}
            label={`B: d_E=${fmt(distB.euc, 2)}, D_M=${fmt(distB.mah, 2)}`}
          />
        </Plot>

        <Plot x={whiteAxisX} y={whiteAxisY} title="Whitened space (isotropic)">
          {/* Coordinate axes */}
          <Curve name="z₁ axis" x={[-4, 4]} y={[0, 0]} muted dashed thin />
          <Curve name="z₂ axis" x={[0, 0]} y={[-4, 4]} muted dashed thin />

          {/* Circular contours */}
          {circlePoints.map((c) => (
            <Curve key={c.k} name={`${c.k}σ circle`} x={c.xs} y={c.ys} dashed={c.k > 1} thin />
          ))}

          {/* Whitened scatter cloud */}
          <Points name="Whitened cloud" x={whitenedCloud.xs} y={whitenedCloud.ys} size={4} muted />

          {/* Center origin */}
          <Points name="Origin" x={[0]} y={[0]} size={7} />

          {/* Chords to whitened points */}
          <Curve name="Whitened chord A" x={[0, distA.z[0]]} y={[0, distA.z[1]]} thin />
          <Curve name="Whitened chord B" x={[0, distB.z[0]]} y={[0, distB.z[1]]} thin dashed />

          {/* Whitened points */}
          <Points name="Point A (whitened)" x={[distA.z[0]]} y={[distA.z[1]]} size={10} emphasis />
          <Points name="Point B (whitened)" x={[distB.z[0]]} y={[distB.z[1]]} size={10} />
        </Plot>
      </Plots>
    </Figure>
  )
}
