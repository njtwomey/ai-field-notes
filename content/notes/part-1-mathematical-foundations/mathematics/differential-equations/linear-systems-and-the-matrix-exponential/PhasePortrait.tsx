import { useMemo } from 'react'
import {
  Choice,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Readout,
  Segments,
  seriesLayers,
  type SeriesSpec,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { classify2, directionField, expm2, formatEig, integrate, type Mat2 } from '../_shared/ode'

type Preset = 'node' | 'saddle' | 'spiral' | 'centre'
const PRESETS: Record<Preset, Mat2> = {
  node: [
    [-1, 0.5],
    [0, -2],
  ],
  saddle: [
    [1, 1],
    [4, 1],
  ],
  spiral: [
    [0, 1],
    [-1, -0.5],
  ],
  centre: [
    [0, 1],
    [-1, 0],
  ],
}
const R = 3
const RANGE: [number, number] = [-R, R]
const RING = Array.from({ length: 10 }, (_, k) => (2 * Math.PI * (k + 0.5)) / 10)
const SPEC = { min: -3, max: 3, step: 0.1 }

export function PhasePortrait() {
  const state = useFigureState({
    a: slider(SPEC.min, SPEC.max, 0, { step: SPEC.step, label: 'a₁₁' }),
    b: slider(SPEC.min, SPEC.max, 1, { step: SPEC.step, label: 'a₁₂' }),
    c: slider(SPEC.min, SPEC.max, -1, { step: SPEC.step, label: 'a₂₁' }),
    d: slider(SPEC.min, SPEC.max, -0.5, { step: SPEC.step, label: 'a₂₂' }),
    time: slider(0, 6, 1, { step: 0.1, label: 'time t' }),
    x0: slider(-R, R, 2, { step: 0.01, onChart: true }),
    y0: slider(-R, R, 0, { step: 0.01, onChart: true }),
  })
  const start = useMemo<[number, number]>(() => [state.x0, state.y0], [state.x0, state.y0])
  const m: Mat2 = useMemo(
    () => [
      [state.a, state.b],
      [state.c, state.d],
    ],
    [state.a, state.b, state.c, state.d],
  )
  const preset = (Object.keys(PRESETS) as Preset[]).find((k) => {
    const p = PRESETS[k]
    return p[0][0] === m[0][0] && p[0][1] === m[0][1] && p[1][0] === m[1][0] && p[1][1] === m[1][1]
  })
  const choose = (k: Preset) => {
    const p = PRESETS[k]
    state.set('a', p[0][0])
    state.set('b', p[0][1])
    state.set('c', p[1][0])
    state.set('d', p[1][1])
  }

  const info = useMemo(() => classify2(m), [m])
  const f = useMemo(
    () => (_t: number, x: number[]) => [m[0][0] * x[0] + m[0][1] * x[1], m[1][0] * x[0] + m[1][1] * x[1]],
    [m],
  )
  const field = useMemo(() => directionField((x, y) => f(0, [x, y]) as [number, number], RANGE, RANGE, 13, 13), [f])

  const background = useMemo<SeriesSpec[]>(() => {
    const out: SeriesSpec[] = RING.map((th) => {
      const path = integrate(f, [2.8 * Math.cos(th), 2.8 * Math.sin(th)], 0, 8, 240, 20)
      return {
        name: 'other trajectories',
        type: 'line',
        x: path.xs.map((p) => p[0]),
        y: path.xs.map((p) => p[1]),
        muted: true,
      }
    })
    // Real eigenvectors are invariant lines: a start on one stays on it.
    const { values } = info
    if (values[0][1] === 0 && Math.abs(info.det) > 1e-9) {
      values.forEach(([l], i) => {
        const row: [number, number] =
          Math.hypot(m[0][0] - l, m[0][1]) > 1e-9 ? [m[0][0] - l, m[0][1]] : [m[1][0], m[1][1] - l]
        const v = [-row[1], row[0]]
        const n = Math.hypot(v[0], v[1]) || 1
        out.push({
          name: `eigenvector λ${i === 0 ? '₁' : '₂'}`,
          type: 'line',
          x: [(-5 * v[0]) / n, (5 * v[0]) / n],
          y: [(-5 * v[1]) / n, (5 * v[1]) / n],
          dashed: true,
          slot: 2 + i,
        })
      })
    }
    return out
  }, [f, info, m])

  const trajectory = useMemo(() => integrate(f, start, 0, 6, 360, 20), [f, start])
  const E = useMemo(() => expm2(m, state.time), [m, state.time])
  const xt = useMemo<[number, number]>(
    () => [E[0][0] * start[0] + E[0][1] * start[1], E[1][0] * start[0] + E[1][1] * start[1]],
    [E, start],
  )

  const series = useMemo<SeriesSpec[]>(
    () => [
      ...background,
      {
        name: 'trajectory from x₀',
        type: 'line',
        x: trajectory.xs.map((p) => p[0]),
        y: trajectory.xs.map((p) => p[1]),
        slot: 0,
      },
      { name: 'x(t) = eᴬᵗ x₀', type: 'scatter', x: [xt[0]], y: [xt[1]], slot: 1 },
      { name: 'x₀', type: 'scatter', x: [start[0]], y: [start[1]], emphasis: true },
    ],
    [background, trajectory, xt, start],
  )
  const xAxis = useAxis({ label: 'x₁', range: RANGE })
  const yAxis = useAxis({ label: 'x₂', range: RANGE, equal: xAxis })
  return (
    <Figure
      title="Phase portrait of ẋ = Ax"
      state={state}
      caption="Arrows show the direction of Ax at each point; grey curves are trajectories from a ring of starts, and dashed lines are real eigenvectors. Drag the black point x₀. The coloured point is eᴬᵗx₀ at the chosen time. Change the entries of A, or pick a preset, and watch the portrait change type as the eigenvalues cross between real and complex or change sign."
      controls={
        <>
          <Choice
            label="preset"
            value={preset ?? ('' as Preset)}
            onChange={choose}
            options={(Object.keys(PRESETS) as Preset[]).map((k) => ({ value: k, label: k }))}
          />
        </>
      }
      readouts={
        <>
          <Readout label="type" value={info.kind} />
          <Readout label="trace" value={formatNumber(info.tr)} />
          <Readout label="det" value={formatNumber(info.det)} />
          <Readout label="λ₁" value={formatEig(info.values[0], formatNumber)} />
          <Readout label="λ₂" value={formatEig(info.values[1], formatNumber)} />
          <Readout
            label="eᴬᵗ"
            value={`[[${formatNumber(E[0][0])}, ${formatNumber(E[0][1])}], [${formatNumber(E[1][0])}, ${formatNumber(E[1][1])}]]`}
          />
        </>
      }
    >
      <div className="mx-auto w-full max-w-lg">
        <Plot x={xAxis} y={yAxis}>
          {seriesLayers(series)}
          <Segments segments={field} />
          <Handle {...state.handle(['x0', 'y0'], { label: 'x₀' })} />
        </Plot>
      </div>
    </Figure>
  )
}
