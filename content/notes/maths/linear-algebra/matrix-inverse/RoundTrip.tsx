import { useMemo, useState, type ReactNode } from 'react'
import {
  Interactive,
  ParamButton,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type ParamSpec,
  type Segment,
  type XYSeries,
} from '@/components/viz'
import { rng } from '@/lib/math'
import { apply, svd2, type Mat2, type Vec2 } from '@/lib/math/mat2'

type Build = 'entries' | 'stretch'

const ENTRY: ParamSpec = { min: -2, max: 2, step: 0.05 }
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
  const g = rng(7)
  return LETTER.map((): Vec2 => [g.normal(), g.normal()])
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
  const [build, setBuild] = useState<Build>('entries')
  const a = useParam(1.2, ENTRY)
  const b = useParam(0.6, ENTRY)
  const c = useParam(-0.3, ENTRY)
  const d = useParam(0.9, ENTRY)
  const outer = useParam(30, { min: 0, max: 180, step: 1 })
  const inner = useParam(0, { min: 0, max: 180, step: 1 })
  const s1 = useParam(1.5, { min: 0.1, max: 2, step: 0.05 })
  const s2 = useParam(0.6, { min: 0, max: 2, step: 0.01 })
  const noise = useParam(0, { min: 0, max: 0.2, step: 0.01 })

  const m: Mat2 = useMemo(() => {
    if (build === 'entries') {
      return [
        [a.value, b.value],
        [c.value, d.value],
      ]
    }
    const scale: Mat2 = [
      [s1.value, 0],
      [0, s2.value],
    ]
    const back = rotation(-inner.value)
    return mul(mul(rotation(outer.value), scale), back)
  }, [build, a.value, b.value, c.value, d.value, outer.value, inner.value, s1.value, s2.value])

  const setEntries = (next: Mat2) => {
    setBuild('entries')
    a.set(next[0][0])
    b.set(next[0][1])
    c.set(next[1][0])
    d.set(next[1][1])
  }

  const r = useMemo(() => {
    const [[m11, m12], [m21, m22]] = m
    const { s, u, v } = svd2(m11, m12, m21, m22)
    const singular = s[1] <= RANK_TOL * Math.max(s[0], 1e-12)
    // Recovery map: A⁻¹ = V Σ⁻¹ Uᵀ when invertible, else A⁺ = V Σ⁺ Uᵀ with 1/0 replaced by 0.
    const inv = s.map((si) => (si > RANK_TOL * Math.max(s[0], 1e-12) ? 1 / si : 0))
    const recover = (y: Vec2): Vec2 => {
      const k1 = (u[0][0] * y[0] + u[0][1] * y[1]) * inv[0]
      const k2 = (u[1][0] * y[0] + u[1][1] * y[1]) * inv[1]
      return [v[0][0] * k1 + v[1][0] * k2, v[0][1] * k1 + v[1][1] * k2]
    }
    const images = LETTER.map(({ p }, i): Vec2 => {
      const y = apply(m, p)
      return [y[0] + noise.value * NOISE[i][0], y[1] + noise.value * NOISE[i][1]]
    })
    const recovered = images.map(recover)
    const rms = Math.sqrt(
      recovered.reduce((acc, q, i) => acc + (q[0] - LETTER[i].p[0]) ** 2 + (q[1] - LETTER[i].p[1]) ** 2, 0) /
        LETTER.length,
    )
    return { s, singular, recover, images, recovered, rms, det: m11 * m22 - m12 * m21 }
  }, [m, noise.value])

  const cloud = (points: Vec2[]): XYSeries => ({
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
      setEntries(next.map((row) => row.map((v) => Math.round(v / ENTRY.step!) * ENTRY.step!)) as Mat2)
    },
  }))

  const panel = (title: string, chart: ReactNode) => (
    <div className="min-w-0 space-y-1">
      <div className="text-center text-xs text-muted-foreground">{title}</div>
      {chart}
    </div>
  )

  return (
    <Interactive
      title="Map, then invert"
      caption="The letter's points x are mapped to y = Ax, then mapped back by the inverse. With an invertible A the round trip returns every point exactly. Drag the tips of the columns Ae₁ and Ae₂ in the middle panel, or build A from a rotation, two stretches and a second rotation. Pull the second stretch to zero and A becomes singular: the letter collapses onto a line, no inverse exists, and the best recovery, the pseudoinverse, stays on a line. Add measurement noise to y to see a nearly singular A amplify small errors."
      controls={
        <>
          <ParamChoice
            label="build A from"
            value={build}
            onChange={(v) => {
              if (v === 'stretch') {
                // Start the stretch controls from the current matrix's SVD so the picture does not jump.
                // Rotations cannot mirror, so a matrix with det A < 0 loses its reflection here.
                const { s, u, v } = svd2(m[0][0], m[0][1], m[1][0], m[1][1])
                const angle = (w: Vec2) => ((((Math.atan2(w[1], w[0]) * 180) / Math.PI) % 180) + 180) % 180
                s1.set(Math.max(s[0], 0.1))
                s2.set(s[1])
                outer.set(Math.round(angle(u[0])))
                inner.set(Math.round(angle(v[0])))
              }
              setBuild(v)
            }}
            options={[
              { value: 'entries', label: 'entries' },
              { value: 'stretch', label: 'rotate · stretch · rotate' },
            ]}
          />
          {build === 'entries' ? (
            <>
              <ParamSlider label="a" param={a} />
              <ParamSlider label="b" param={b} />
              <ParamSlider label="c" param={c} />
              <ParamSlider label="d" param={d} />
            </>
          ) : (
            <>
              <ParamSlider label="first rotation (°)" param={inner} />
              <ParamSlider label="stretch σ₁" param={s1} />
              <ParamSlider label="stretch σ₂" param={s2} />
              <ParamSlider label="second rotation (°)" param={outer} />
            </>
          )}
          <ParamSlider label="noise on y" param={noise} />
          <div className="flex flex-wrap gap-1.5 self-end">
            {PRESETS.map((p) => (
              <ParamButton key={p.label} onClick={() => setEntries(p.m)}>
                {p.label}
              </ParamButton>
            ))}
          </div>
        </>
      }
      readout={
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
          <XYChart
            series={[cloud(LETTER.map((q) => q.p))]}
            vectors={arrows([
              [1, 0],
              [0, 1],
            ])}
            xRange={RANGE}
            yRange={RANGE}
            equalAspect
          />,
        )}
        {panel(
          noise.value > 0 ? 'y = Ax + noise' : 'y = Ax',
          <XYChart
            series={[cloud(r.images)]}
            vectors={arrows(columns)}
            xRange={RANGE}
            yRange={RANGE}
            equalAspect
            handles={handles}
          />,
        )}
        {panel(
          r.singular ? 'no inverse: A⁺y, the best recovery' : 'A⁻¹y',
          <XYChart
            series={[cloud(r.recovered)]}
            vectors={arrows(columns.map(r.recover))}
            xRange={RANGE}
            yRange={RANGE}
            equalAspect
          />,
        )}
      </div>
    </Interactive>
  )
}
