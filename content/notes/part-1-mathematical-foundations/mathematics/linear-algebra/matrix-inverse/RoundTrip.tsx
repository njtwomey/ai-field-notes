import { useEffect, useMemo, useRef, type ReactNode } from 'react'
import {
  Button,
  choice,
  Figure,
  float,
  formatNumber,
  Handle,
  Plot,
  Readout,
  type Segment,
  seriesLayers,
  type SeriesSpec,
  slider,
  useAxis,
  useFigureState,
  Vectors,
} from 'aifn-render'
import { normal, stream } from 'aifn/foundation/random'
import { apply2, svd2, type Mat2, type Vec2 } from 'aifn/numerics/linalg'

type Build = 'entries' | 'stretch'

const ENTRY_STEP = 0.05
const R = 3
const RANGE: [number, number] = [-R, R]
/** Singular values below this fraction of the largest count as zero: the matrix has no inverse. */
const RANK_TOL = 1e-6

const PARTS = ['stem', 'top bar', 'middle bar']

/** A filled letter F, centred on the origin, as points labelled by the part they belong to. */
const LETTER: { p: Vec2; part: number }[] = (() => {
  const rects: [number, number, number, number, number][] = [
    [-0.5, -0.2, -1.6, 1.6, 0],
    [-0.2, 0.6, 1.2, 1.6, 1],
    [-0.2, 0.4, 0.1, 0.5, 2],
  ]
  const out: { p: Vec2; part: number }[] = []
  for (const [x0, x1, y0, y1, part] of rects) {
    for (let x = x0; x <= x1 + 1e-9; x += 0.1) {
      for (let y = y0; y <= y1 + 1e-9; y += 0.1) out.push({ p: [x, y], part })
    }
  }
  return out
})()

/** A fixed standard-normal draw per point, scaled by the noise slider. */
const NOISE: Vec2[] = (() => {
  const g = stream(7)
  return LETTER.map((): Vec2 => [normal(g), normal(g)])
})()

const rotation = (deg: number): Mat2 => {
  const t = (deg * Math.PI) / 180
  return [
    [Math.cos(t), -Math.sin(t)],
    [Math.sin(t), Math.cos(t)],
  ]
}
const mul = (m: Mat2, n: Mat2): Mat2 => [
  [m[0][0] * n[0][0] + m[0][1] * n[1][0], m[0][0] * n[0][1] + m[0][1] * n[1][1]],
  [m[1][0] * n[0][0] + m[1][1] * n[1][0], m[1][0] * n[0][1] + m[1][1] * n[1][1]],
]

const PRESETS: { label: string; m: Mat2 }[] = [
  { label: 'Rotation', m: rotation(40) },
  {
    label: 'Shear',
    m: [
      [1, 1],
      [0, 1],
    ],
  },
  {
    label: 'Stretch',
    m: [
      [1.6, 0],
      [0, 0.6],
    ],
  },
  {
    label: 'Nearly singular',
    m: [
      [1, 1],
      [1, 1.1],
    ],
  },
  {
    label: 'Projection',
    m: [
      [1, 0],
      [0, 0],
    ],
  },
]

/**
 * Points x, their images y = A x (optionally with measurement noise), and the recovery A⁻¹ y. The recovery lands back
 * on x exactly when A is invertible and y is exact. A singular A has no inverse; the pseudoinverse A⁺ then gives the
 * closest recovery, which collapses onto a line: the lost direction cannot be restored.
 */
export function RoundTrip() {
  const byEntries = (v: Readonly<Record<string, unknown>>) => v.build === 'entries'
  const byStretch = (v: Readonly<Record<string, unknown>>) => v.build === 'stretch'
  const entry = (initial: number, label: string) => slider(-2, 2, initial, { step: ENTRY_STEP, label, when: byEntries })
  const state = useFigureState({
    build: choice<Build>(
      [
        { value: 'entries', label: 'entries' },
        { value: 'stretch', label: 'rotate · stretch · rotate' },
      ],
      'entries',
      { label: 'build A from' },
    ),
    a: entry(1.2, 'a'),
    b: entry(0.6, 'b'),
    c: entry(-0.3, 'c'),
    d: entry(0.9, 'd'),
    inner: slider(0, 180, 0, { step: 1, label: 'first rotation (°)', when: byStretch }),
    s1: slider(0.1, 2, 1.5, { step: 0.05, label: 'stretch σ₁', when: byStretch }),
    s2: slider(0, 2, 0.6, { step: 0.01, label: 'stretch σ₂', when: byStretch }),
    outer: slider(0, 180, 30, { step: 1, label: 'second rotation (°)', when: byStretch }),
    noise: float(0, { min: 0, max: 0.2, step: 0.01, label: 'noise on y' }),
  })
  const build = state.build

  const m: Mat2 = useMemo(() => {
    if (build === 'entries') {
      return [
        [state.a, state.b],
        [state.c, state.d],
      ]
    }
    const scale: Mat2 = [
      [state.s1, 0],
      [0, state.s2],
    ]
    const back = rotation(-state.inner)
    return mul(mul(rotation(state.outer), scale), back)
  }, [build, state.a, state.b, state.c, state.d, state.outer, state.inner, state.s1, state.s2])

  const setEntries = (next: Mat2) => {
    state.set('build', 'entries')
    state.set('a', next[0][0])
    state.set('b', next[0][1])
    state.set('c', next[1][0])
    state.set('d', next[1][1])
  }

  // On a switch to the stretch controls, start them from the current matrix's SVD so the picture does not jump.
  // Rotations cannot mirror, so a matrix with det A < 0 loses its reflection here.
  const previous = useRef(build)
  useEffect(() => {
    if (previous.current === build) return
    previous.current = build
    if (build !== 'stretch') return
    const { s, u, v } = svd2([
      [state.a, state.b],
      [state.c, state.d],
    ])
    const angle = (w: Vec2) => ((((Math.atan2(w[1], w[0]) * 180) / Math.PI) % 180) + 180) % 180
    state.set('s1', Math.max(s[0], 0.1))
    state.set('s2', s[1])
    state.set('outer', Math.round(angle(u[0])))
    state.set('inner', Math.round(angle(v[0])))
  }, [build, state])

  const r = useMemo(() => {
    const [[m11, m12], [m21, m22]] = m
    const { s, u, v } = svd2(m)
    const singular = s[1] <= RANK_TOL * Math.max(s[0], 1e-12)
    // Recovery map: A⁻¹ = V Σ⁻¹ Uᵀ when invertible, else A⁺ = V Σ⁺ Uᵀ with 1/0 replaced by 0.
    const inv = s.map((si) => (si > RANK_TOL * Math.max(s[0], 1e-12) ? 1 / si : 0))
    const recover = (y: Vec2): Vec2 => {
      const k1 = (u[0][0] * y[0] + u[0][1] * y[1]) * inv[0]
      const k2 = (u[1][0] * y[0] + u[1][1] * y[1]) * inv[1]
      return [v[0][0] * k1 + v[1][0] * k2, v[0][1] * k1 + v[1][1] * k2]
    }
    const images = LETTER.map(({ p }, i): Vec2 => {
      const y = apply2(m, p)
      return [y[0] + state.noise * NOISE[i][0], y[1] + state.noise * NOISE[i][1]]
    })
    const recovered = images.map(recover)
    const rms = Math.sqrt(
      recovered.reduce((acc, q, i) => acc + (q[0] - LETTER[i].p[0]) ** 2 + (q[1] - LETTER[i].p[1]) ** 2, 0) /
        LETTER.length,
    )
    return { s, singular, recover, images, recovered, rms, det: m11 * m22 - m12 * m21 }
  }, [m, state.noise])

  const cloud = (points: Vec2[]): SeriesSpec => ({
    name: 'points',
    type: 'scatter',
    x: points.map((p) => p[0]),
    y: points.map((p) => p[1]),
    group: LETTER.map((q) => q.part),
    groupNames: PARTS,
  })
  const arrows = (tips: Vec2[]): Segment[] => tips.map((t) => ({ from: [0, 0], to: t }))
  const columns: Vec2[] = [
    [m[0][0], m[1][0]],
    [m[0][1], m[1][1]],
  ]
  const handles: Handle[] = columns.map((col, j) => ({
    kind: 'point',
    at: col,
    label: `A e${j + 1}`,
    onDrag: ([x, y]) => {
      const next: Mat2 =
        j === 0
          ? [
              [x, m[0][1]],
              [y, m[1][1]],
            ]
          : [
              [m[0][0], x],
              [m[1][0], y],
            ]
      setEntries(next.map((row) => row.map((v) => Math.round(v / ENTRY_STEP) * ENTRY_STEP)) as Mat2)
    },
  }))

  const panel = (title: string, chart: ReactNode) => (
    <div className="min-w-0 space-y-1">
      <div className="text-center text-xs text-muted-foreground">{title}</div>
      {chart}
    </div>
  )

  const xAxis = useAxis({ range: RANGE })
  const yAxis = useAxis({ range: RANGE, equal: xAxis })
  const xAxis2 = useAxis({ range: RANGE })
  const yAxis2 = useAxis({ range: RANGE, equal: xAxis2 })
  const xAxis3 = useAxis({ range: RANGE })
  const yAxis3 = useAxis({ range: RANGE, equal: xAxis3 })
  return (
    <Figure
      title="Map, then invert"
      state={state}
      caption="The letter's points x are mapped to y = Ax, then mapped back by the inverse. With an invertible A the round trip returns every point exactly. Drag the tips of the columns Ae₁ and Ae₂ in the middle panel, or build A from a rotation, two stretches and a second rotation. Pull the second stretch to zero and A becomes singular: the letter collapses onto a line, no inverse exists, and the best recovery, the pseudoinverse, stays on a line. Add measurement noise to y to see a nearly singular A amplify small errors."
      controls={
        <>
          <div className="flex flex-wrap gap-1.5 self-end">
            {PRESETS.map((p) => (
              <Button variant="outline" size="sm" key={p.label} onClick={() => setEntries(p.m)}>
                {p.label}
              </Button>
            ))}
          </div>
        </>
      }
      readouts={
        <>
          <Readout label="det A" value={formatNumber(r.det)} />
          <Readout label="σ₁, σ₂" value={`${formatNumber(r.s[0])}, ${formatNumber(r.s[1])}`} />
          <Readout label="condition number" value={r.singular ? '∞' : formatNumber(r.s[0] / r.s[1])} />
          <Readout label="rank" value={r.singular ? (r.s[0] > RANK_TOL ? 1 : 0) : 2} />
          <Readout label="recovery error (rms)" value={formatNumber(r.rms)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-3">
        {panel(
          'x',
          <Plot x={xAxis} y={yAxis}>
            {seriesLayers([cloud(LETTER.map((q) => q.p))])}
            <Vectors
              vectors={arrows([
                [1, 0],
                [0, 1],
              ])}
            />
          </Plot>,
        )}
        {panel(
          state.noise > 0 ? 'y = Ax + noise' : 'y = Ax',
          <Plot x={xAxis2} y={yAxis2}>
            {seriesLayers([cloud(r.images)])}
            <Vectors vectors={arrows(columns)} />
            {(handles ?? []).map((h, i) => (
              <Handle key={i} {...h} />
            ))}
          </Plot>,
        )}
        {panel(
          r.singular ? 'no inverse: A⁺y, the best recovery' : 'A⁻¹y',
          <Plot x={xAxis3} y={yAxis3}>
            {seriesLayers([cloud(r.recovered)])}
            <Vectors vectors={arrows(columns.map(r.recover))} />
          </Plot>,
        )}
      </div>
    </Figure>
  )
}
