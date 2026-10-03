import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
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
  const a = useParam(0, SPEC)
  const b = useParam(1, SPEC)
  const c = useParam(-1, SPEC)
  const d = useParam(-0.5, SPEC)
  const time = useParam(1, { min: 0, max: 6, step: 0.1 })
  const [start, setStart] = useState<[number, number]>([2, 0])
  const m: Mat2 = useMemo(
    () => [
      [a.value, b.value],
      [c.value, d.value],
    ],
    [a.value, b.value, c.value, d.value],
  )
  const preset = (Object.keys(PRESETS) as Preset[]).find((k) => {
    const p = PRESETS[k]
    return p[0][0] === m[0][0] && p[0][1] === m[0][1] && p[1][0] === m[1][0] && p[1][1] === m[1][1]
  })
  const choose = (k: Preset) => {
    const p = PRESETS[k]
    a.set(p[0][0])
    b.set(p[0][1])
    c.set(p[1][0])
    d.set(p[1][1])
  }

  const info = useMemo(() => classify2(m), [m])
  const f = useMemo(
    () => (_t: number, x: number[]) => [m[0][0] * x[0] + m[0][1] * x[1], m[1][0] * x[0] + m[1][1] * x[1]],
    [m],
  )
  const field = useMemo(() => directionField((x, y) => f(0, [x, y]) as [number, number], RANGE, RANGE, 13, 13), [f])

  const background = useMemo<XYSeries[]>(() => {
    const out: XYSeries[] = RING.map((th) => {
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
  const E = useMemo(() => expm2(m, time.value), [m, time.value])
  const xt = useMemo<[number, number]>(
    () => [E[0][0] * start[0] + E[0][1] * start[1], E[1][0] * start[0] + E[1][1] * start[1]],
    [E, start],
  )

  const series = useMemo<XYSeries[]>(
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
  const handles = useMemo<Handle[]>(
    () => [
      {
        kind: 'point',
        at: start,
        label: 'x₀',
        onDrag: ([x, y]) => setStart([Math.max(-R, Math.min(R, x)), Math.max(-R, Math.min(R, y))]),
      },
    ],
    [start],
  )

  return (
    <Interactive
      title="Phase portrait of ẋ = Ax"
      caption="Arrows show the direction of Ax at each point; grey curves are trajectories from a ring of starts, and dashed lines are real eigenvectors. Drag the black point x₀. The coloured point is eᴬᵗx₀ at the chosen time. Change the entries of A, or pick a preset, and watch the portrait change type as the eigenvalues cross between real and complex or change sign."
      controls={
        <>
          <ParamChoice
            label="preset"
            value={preset ?? ('' as Preset)}
            onChange={choose}
            options={(Object.keys(PRESETS) as Preset[]).map((k) => ({ value: k, label: k }))}
          />
          <ParamSlider label="time t" param={time} withArrows />
          <ParamSlider label="a₁₁" param={a} />
          <ParamSlider label="a₁₂" param={b} />
          <ParamSlider label="a₂₁" param={c} />
          <ParamSlider label="a₂₂" param={d} />
        </>
      }
      readout={
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
        <XYChart
          series={series}
          segments={field}
          handles={handles}
          xRange={RANGE}
          yRange={RANGE}
          xLabel="x₁"
          yLabel="x₂"
          equalAspect
        />
      </div>
    </Interactive>
  )
}
