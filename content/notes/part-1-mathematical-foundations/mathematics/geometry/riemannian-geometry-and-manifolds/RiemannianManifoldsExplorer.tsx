import { useMemo, useState } from 'react'
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

export function RiemannianManifoldsExplorer() {
  const [latitude, setLatitude] = useState(0.3) // radians from equator
  const [longitude, setLongitude] = useState(0.0) // radians
  const [velocityAngle, setVelocityAngle] = useState(0.7) // direction in tangent space
  const [velocitySpeed, setVelocitySpeed] = useState(1.2) // speed = geodesic distance
  const [viewMode, setViewMode] = useState<'tangent' | 'retraction'>('tangent')

  // Base point p on unit sphere S² in 3D:
  const p3D = useMemo(() => {
    const x = Math.cos(latitude) * Math.cos(longitude)
    const y = Math.cos(latitude) * Math.sin(longitude)
    const z = Math.sin(latitude)
    return [x, y, z] as [number, number, number]
  }, [latitude, longitude])

  // Orthogonal basis for tangent space T_p S²:
  // e_east (along longitude), e_north (along latitude)
  const basis = useMemo(() => {
    const east = [-Math.sin(longitude), Math.cos(longitude), 0]
    const north = [-Math.sin(latitude) * Math.cos(longitude), -Math.sin(latitude) * Math.sin(longitude), Math.cos(latitude)]
    return { east, north }
  }, [latitude, longitude])

  // Tangent vector v in T_p S²:
  const tangentV = useMemo(() => {
    const vEast = velocitySpeed * Math.cos(velocityAngle)
    const vNorth = velocitySpeed * Math.sin(velocityAngle)
    const vx = vEast * basis.east[0] + vNorth * basis.north[0]
    const vy = vEast * basis.east[1] + vNorth * basis.north[1]
    const vz = vEast * basis.east[2] + vNorth * basis.north[2]
    return {
      vEast,
      vNorth,
      v3D: [vx, vy, vz] as [number, number, number],
      speed: velocitySpeed,
    }
  }, [velocityAngle, velocitySpeed, basis])

  // Geodesic curve via exponential map:
  // On unit sphere: exp_p(t * v) = cos(t * s) * p + sin(t * s) * (v / s)
  const geodesic = useMemo(() => {
    const steps = 40
    const s = tangentV.speed
    if (s < 1e-7) {
      return {
        path3D: [p3D],
        expP: p3D,
        euclidP: p3D,
        retractP: p3D,
      }
    }

    const unitV = [
      tangentV.v3D[0] / s,
      tangentV.v3D[1] / s,
      tangentV.v3D[2] / s,
    ]

    const path: [number, number, number][] = []
    for (let i = 0; i <= steps; i++) {
      const t = i / steps
      const cosT = Math.cos(t * s)
      const sinT = Math.sin(t * s)
      const gx = cosT * p3D[0] + sinT * unitV[0]
      const gy = cosT * p3D[1] + sinT * unitV[1]
      const gz = cosT * p3D[2] + sinT * unitV[2]
      path.push([gx, gy, gz])
    }

    const expP = path[steps]

    // Flat Euclidean step: p + v (violates constraint ||x|| = 1)
    const euclidP: [number, number, number] = [
      p3D[0] + tangentV.v3D[0],
      p3D[1] + tangentV.v3D[1],
      p3D[2] + tangentV.v3D[2],
    ]

    // Retraction map: R_p(v) = (p + v) / ||p + v||
    const euclidNorm = Math.sqrt(euclidP[0] ** 2 + euclidP[1] ** 2 + euclidP[2] ** 2)
    const retractP: [number, number, number] = [
      euclidP[0] / euclidNorm,
      euclidP[1] / euclidNorm,
      euclidP[2] / euclidNorm,
    ]

    return { path3D: path, expP, euclidP, retractP }
  }, [p3D, tangentV])

  // Orthographic 2D projection for plotting: (x, z)
  const projGeodesicX = useMemo(() => Float64Array.from(geodesic.path3D, (pt) => pt[0]), [geodesic])
  const projGeodesicZ = useMemo(() => Float64Array.from(geodesic.path3D, (pt) => pt[2]), [geodesic])

  // Sphere outline for 2D orthographic projection
  const sphereOutline = useMemo(() => {
    const steps = 90
    const xs = Float64Array.from({ length: steps }, (_, i) => Math.cos((2 * Math.PI * i) / (steps - 1)))
    const zs = Float64Array.from({ length: steps }, (_, i) => Math.sin((2 * Math.PI * i) / (steps - 1)))
    return { xs, zs }
  }, [])

  // Latitude circles
  const equator = useMemo(() => {
    const steps = 60
    const xs = Float64Array.from({ length: steps }, (_, i) => Math.cos((Math.PI * i) / (steps - 1)))
    const zs = new Float64Array(steps).fill(0)
    return { xs, zs }
  }, [])

  // Tangent plane view: 2D coordinates in T_p S² (vEast, vNorth)
  const tangentCircle = useMemo(() => {
    const steps = 60
    const xs = Float64Array.from({ length: steps }, (_, i) => Math.PI * Math.cos((2 * Math.PI * i) / (steps - 1)))
    const ys = Float64Array.from({ length: steps }, (_, i) => Math.PI * Math.sin((2 * Math.PI * i) / (steps - 1)))
    return { xs, ys }
  }, [])

  // Geodesic distance vs Euclidean cord distance
  const geodesicDist = velocitySpeed
  const euclidCordDist = Math.sqrt(
    (geodesic.expP[0] - p3D[0]) ** 2 +
      (geodesic.expP[1] - p3D[1]) ** 2 +
      (geodesic.expP[2] - p3D[2]) ** 2,
  )

  const euclidTangentDist = Math.sqrt(
    (geodesic.euclidP[0] - p3D[0]) ** 2 +
      (geodesic.euclidP[1] - p3D[1]) ** 2 +
      (geodesic.euclidP[2] - p3D[2]) ** 2,
  )

  const euclidConstraintViolation = Math.sqrt(
    geodesic.euclidP[0] ** 2 + geodesic.euclidP[1] ** 2 + geodesic.euclidP[2] ** 2,
  ) - 1.0

  const retractionError = Math.sqrt(
    (geodesic.retractP[0] - geodesic.expP[0]) ** 2 +
      (geodesic.retractP[1] - geodesic.expP[1]) ** 2 +
      (geodesic.retractP[2] - geodesic.expP[2]) ** 2,
  )

  // Axes
  const sphereAxisX = useAxis({ label: 'ambient x', range: [-1.4, 1.4] })
  const sphereAxisZ = useAxis({ label: 'ambient z (vertical axis)', range: [-1.4, 1.4] })

  const tangentAxisX = useAxis({ label: 'tangent velocity v_east', range: [-2.5, 2.5] })
  const tangentAxisY = useAxis({ label: 'tangent velocity v_north', range: [-2.5, 2.5] })

  return (
    <Figure
      title="Riemannian manifold: tangent space, geodesics, and exponential map"
      purpose="Visualizes how the exponential map exp_p(v) maps flat tangent vectors v ∈ T_p M onto intrinsic geodesic curves on the manifold, contrasting exact geodesics against flat Euclidean steps and first-order retractions."
      controls={
        <>
          <ControlRow label="Tangent vector in T_p S²">
            <Slider
              label="Vector speed ||v|| (geodesic arc length)"
              value={velocitySpeed}
              onChange={setVelocitySpeed}
              min={0.1}
              max={2.5}
              step={0.05}
            />
            <Slider
              label="Vector direction angle (rad)"
              value={velocityAngle}
              onChange={setVelocityAngle}
              min={0.0}
              max={Math.PI * 2}
              step={0.05}
            />
            <Select
              label="Comparison overlay"
              value={viewMode}
              onChange={(v) => setViewMode(v as 'tangent' | 'retraction')}
              options={[
                { value: 'tangent', label: 'Flat Euclidean step p + v (constraint violation)' },
                { value: 'retraction', label: 'Retraction R_p(v) = (p+v)/||p+v||' },
              ]}
            />
          </ControlRow>
          <ControlRow label="Base point position on sphere">
            <Slider
              label="Base latitude φ (rad)"
              value={latitude}
              onChange={setLatitude}
              min={-1.2}
              max={1.2}
              step={0.05}
            />
            <Slider
              label="Base longitude θ (rad)"
              value={longitude}
              onChange={setLongitude}
              min={-Math.PI}
              max={Math.PI}
              step={0.1}
            />
          </ControlRow>
        </>
      }
      readouts={{
        'geodesic metric quantities': (
          <>
            <Readout label="intrinsic geodesic distance d_M(p, exp(v))" value={`${fmt(geodesicDist, 3)} rad`} />
            <Readout label="Euclidean chord length in R³" value={fmt(euclidCordDist, 3)} />
            <Readout
              label="chord-to-arc contraction"
              value={`${fmt((1 - euclidCordDist / geodesicDist) * 100, 1)}%`}
            />
          </>
        ),
        'approximation errors': (
          <>
            <Readout label="Euclidean tangent step norm" value={fmt(euclidTangentDist, 3)} />
            <Readout
              label="Euclidean constraint violation (||p+v|| − 1)"
              value={fmt(euclidConstraintViolation, 4)}
            />
            <Readout
              label="retraction approximation error ||R_p(v) − exp_p(v)||"
              value={fmt(retractionError, 4)}
            />
          </>
        ),
      }}
      caption="Riemannian geometry on the sphere S². Left: 2D projection of the unit sphere. Point p (blue) serves as base point; the green curve shows the true geodesic exp_p(tv) swept by the tangent vector. The amber point denotes the unconstrained flat Euclidean step p + v (which escapes the sphere), while the purple point shows the retraction R_p(v) = (p+v)/||p+v|| commonly used in Riemannian SGD. Right: The flat tangent space T_p S² equipped with inner product g_p = I. The red circle denotes injectivity radius π."
    >
      <Plots cols={2}>
        <Plot x={sphereAxisX} y={sphereAxisZ} title="geodesic exp_p(v) on unit sphere S²">
          {/* Sphere circle and equator */}
          <Curve name="sphere boundary S²" x={sphereOutline.xs} y={sphereOutline.zs} muted dashed thin />
          <Curve name="equator" x={equator.xs} y={equator.zs} muted thin />
          {/* Geodesic path on sphere */}
          <Curve name="geodesic arc exp_p(tv)" x={projGeodesicX} y={projGeodesicZ} emphasis />
          {/* Base point p */}
          <Points name="base point p" x={[p3D[0]]} y={[p3D[2]]} size={9} slot={1} />
          {/* Destination exp_p(v) */}
          <Points name="exp_p(v)" x={[geodesic.expP[0]]} y={[geodesic.expP[2]]} size={8} emphasis />
          {viewMode === 'tangent' ? (
            <>
              {/* Flat Euclidean step */}
              <Curve
                name="tangent vector v in ambient R³"
                x={[p3D[0], geodesic.euclidP[0]]}
                y={[p3D[2], geodesic.euclidP[2]]}
                slot={2}
                dashed
              />
              <Points
                name="flat step p + v (violates constraint)"
                x={[geodesic.euclidP[0]]}
                y={[geodesic.euclidP[2]]}
                size={7}
                slot={2}
              />
            </>
          ) : (
            <>
              {/* Retraction step */}
              <Points
                name="retraction R_p(v)"
                x={[geodesic.retractP[0]]}
                y={[geodesic.retractP[2]]}
                size={8}
                slot={3}
              />
            </>
          )}
        </Plot>
        <Plot x={tangentAxisX} y={tangentAxisY} title="tangent space T_p S² (flat coordinate plane)">
          <Curve name="v_east axis" x={[-2.5, 2.5]} y={[0, 0]} muted dashed thin />
          <Curve name="v_north axis" x={[0, 0]} y={[-2.5, 2.5]} muted dashed thin />
          <Curve name="injectivity radius r = π" x={tangentCircle.xs} y={tangentCircle.ys} muted dashed thin />
          {/* Tangent vector arrow */}
          <Curve
            name="tangent vector v"
            x={[0, tangentV.vEast]}
            y={[0, tangentV.vNorth]}
            emphasis
          />
          <Handle
            kind="point"
            at={[tangentV.vEast, tangentV.vNorth]}
            onDrag={([ve, vn]) => {
              const speed = Math.max(0.1, Math.min(2.5, Math.sqrt(ve * ve + vn * vn)))
              const angle = (Math.atan2(vn, ve) + 2 * Math.PI) % (2 * Math.PI)
              setVelocitySpeed(speed)
              setVelocityAngle(angle)
            }}
            label={`v: ||v|| = ${fmt(tangentV.speed, 2)}`}
          />
        </Plot>
      </Plots>
    </Figure>
  )
}
