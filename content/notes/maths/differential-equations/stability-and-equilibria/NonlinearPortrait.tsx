import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  type Handle,
  type XYSeries,
} from '@/components/viz'
import { classify2, directionField, formatEig, integrate, type Mat2 } from '../_shared/ode'

type System = 'pendulum' | 'double-well' | 'predator-prey'
type Spec = {
  label: string
  x: [number, number]
  y: [number, number]
  start: [number, number]
  f: (x: number, y: number) => [number, number]
  fixed: [number, number][]
  jac: (x: number, y: number) => Mat2
  duration: number
}

function spec(system: System, gamma: number): Spec {
  if (system === 'pendulum')
    return {
      label: 'damped pendulum',
      x: [-7, 7],
      y: [-3.5, 3.5],
      start: [-6, 3],
      f: (x, y) => [y, -Math.sin(x) - gamma * y],
      fixed: [-2 * Math.PI, -Math.PI, 0, Math.PI, 2 * Math.PI].map((x) => [x, 0]),
      jac: (x) => [
        [0, 1],
        [-Math.cos(x), -gamma],
      ],
      duration: 30,
    }
  if (system === 'double-well')
    return {
      label: 'gradient flow on a double well',
      x: [-2, 2],
      y: [-1.5, 1.5],
      start: [0.05, 1.4],
      f: (x, y) => [x - x ** 3, -y],
      fixed: [
        [-1, 0],
        [0, 0],
        [1, 0],
      ],
      jac: (x) => [
        [1 - 3 * x * x, 0],
        [0, -1],
      ],
      duration: 12,
    }
  return {
    label: 'predator and prey',
    x: [0, 3.2],
    y: [0, 3.2],
    start: [0.5, 0.5],
    f: (x, y) => [x * (1 - y), y * (x - 1)],
    fixed: [
      [0, 0],
      [1, 1],
    ],
    jac: (x, y) => [
      [1 - y, -x],
      [y, x - 1],
    ],
    duration: 20,
  }
}

const SYSTEMS: { value: System; label: string }[] = [
  { value: 'pendulum', label: 'pendulum' },
  { value: 'double-well', label: 'double well' },
  { value: 'predator-prey', label: 'predator–prey' },
]

export function NonlinearPortrait() {
  const [system, setSystem] = useState<System>('pendulum')
  const [gamma, setGamma] = useState(0.5)
  const s = useMemo(() => spec(system, gamma), [system, gamma])
  const [start, setStart] = useState<[number, number]>(s.start)

  const field = useMemo(() => directionField(s.f, s.x, s.y, 17, 11), [s])
  const path = useMemo(
    () => integrate((_t, p) => s.f(p[0], p[1]), start, 0, s.duration, Math.round(s.duration * 30), 50),
    [s, start],
  )
  const kinds = useMemo(() => s.fixed.map(([x, y]) => classify2(s.jac(x, y))), [s])

  const series = useMemo<XYSeries[]>(
    () => [
      { name: 'trajectory', type: 'line', x: path.xs.map((p) => p[0]), y: path.xs.map((p) => p[1]), slot: 0 },
      {
        name: 'stable equilibria',
        type: 'scatter',
        x: s.fixed.filter((_, i) => isStable(kinds[i].kind)).map((p) => p[0]),
        y: s.fixed.filter((_, i) => isStable(kinds[i].kind)).map((p) => p[1]),
        slot: 1,
      },
      {
        name: 'unstable or neutral equilibria',
        type: 'scatter',
        x: s.fixed.filter((_, i) => !isStable(kinds[i].kind)).map((p) => p[0]),
        y: s.fixed.filter((_, i) => !isStable(kinds[i].kind)).map((p) => p[1]),
        slot: 2,
      },
      { name: 'start', type: 'scatter', x: [start[0]], y: [start[1]], emphasis: true },
    ],
    [path, s, kinds, start],
  )

  const handles = useMemo<Handle[]>(
    () => [
      {
        kind: 'point',
        at: start,
        label: 'start',
        onDrag: ([x, y]) => setStart([Math.min(s.x[1], Math.max(s.x[0], x)), Math.min(s.y[1], Math.max(s.y[0], y))]),
      },
    ],
    [start, s],
  )

  const choose = (v: System) => {
    setSystem(v)
    setStart(spec(v, gamma).start)
  }
  // Show each distinct equilibrium type once; the pendulum repeats its two types every 2π.
  const shown = s.fixed
    .map((p, i) => ({ p, k: kinds[i] }))
    .filter(({ p }) => system !== 'pendulum' || (p[0] >= 0 && p[0] <= Math.PI))

  return (
    <Interactive
      title="Equilibria of nonlinear systems"
      caption="Arrows show the vector field. Drag the black start point to launch a trajectory. Equilibria are marked on the plot and listed below, each classified by the eigenvalues of the Jacobian there. In the pendulum, lower the damping to zero and the stable spirals become centres; in the double well, starts on either side of the y-axis fall into different minima."
      controls={
        <>
          <ParamChoice label="system" value={system} onChange={choose} options={SYSTEMS} />
          {system === 'pendulum' && (
            <ParamSlider label="damping γ" value={gamma} onChange={setGamma} min={0} max={1.5} step={0.05} />
          )}
        </>
      }
      readout={shown.map(({ p, k }, i) => (
        <Readout
          key={i}
          label={`(${formatNumber(p[0])}, ${formatNumber(p[1])})`}
          value={`${k.kind}; λ = ${formatEig(k.values[0], formatNumber)}, ${formatEig(k.values[1], formatNumber)}`}
        />
      ))}
    >
      <XYChart
        height={340}
        series={series}
        segments={field}
        handles={handles}
        xRange={s.x}
        yRange={s.y}
        xLabel={system === 'pendulum' ? 'angle θ' : system === 'predator-prey' ? 'prey x' : 'x'}
        yLabel={system === 'pendulum' ? 'velocity ω' : system === 'predator-prey' ? 'predators y' : 'y'}
      />
    </Interactive>
  )
}

const isStable = (kind: string) => kind.startsWith('stable')
