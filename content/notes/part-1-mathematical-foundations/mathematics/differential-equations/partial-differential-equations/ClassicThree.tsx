import { useMemo, useState } from 'react'
import {
  Heatmap,
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
import { linspace } from '@/lib/math'

type Equation = 'heat' | 'wave' | 'laplace'
const MODES = 80
const WIDTH = 0.06
const D = 0.02
const X = linspace(0, 1, 201)
const GRID = linspace(0, 1, 51)

const bump = (c: number) => (x: number) => Math.exp(-((x - c) ** 2) / (2 * WIDTH * WIDTH))

/** Sine-series coefficients bₙ = 2∫₀¹ g(x) sin(nπx) dx, by the trapezoid rule on a fine grid. */
function sineCoefficients(g: (x: number) => number): number[] {
  const xs = linspace(0, 1, 1001)
  const h = xs[1] - xs[0]
  return Array.from({ length: MODES }, (_, k) => {
    const n = k + 1
    let s = 0
    for (let i = 1; i < xs.length - 1; i++) s += g(xs[i]) * Math.sin(n * Math.PI * xs[i])
    return 2 * h * s
  })
}

/** sinh(nπy)/sinh(nπ), written with decaying exponentials so large n cannot overflow. */
const sinhRatio = (n: number, y: number) =>
  (Math.exp(n * Math.PI * (y - 1)) - Math.exp(-n * Math.PI * (y + 1))) / (1 - Math.exp(-2 * n * Math.PI))

export function ClassicThree() {
  const [equation, setEquation] = useState<Equation>('heat')
  const time = useParam(0.3, { min: 0, max: 2, step: 0.05 })
  const centre = useParam(0.35, { min: 0.1, max: 0.9, step: 0.01 })
  const b = useMemo(() => sineCoefficients(bump(centre.value)), [centre.value])

  const initial = useMemo(() => X.map(bump(centre.value)), [centre.value])
  const profile = useMemo(() => {
    if (equation === 'laplace') return []
    return X.map((x) =>
      b.reduce((s, bn, k) => {
        const n = k + 1
        const w = n * Math.PI
        const factor = equation === 'heat' ? Math.exp(-D * w * w * time.value) : Math.cos(w * time.value)
        return s + bn * factor * Math.sin(w * x)
      }, 0),
    )
  }, [b, equation, time.value])

  const laplace = useMemo(() => {
    if (equation !== 'laplace') return null
    // Rows are y values (z[i][j] at (x[j], y[i])); the top edge y = 1 carries the bump.
    const z = GRID.map((y) =>
      GRID.map((x) => b.reduce((s, bn, k) => s + bn * Math.sin((k + 1) * Math.PI * x) * sinhRatio(k + 1, y), 0)),
    )
    return z
  }, [b, equation])

  const series = useMemo<XYSeries[]>(
    () => [
      { name: 'initial shape', type: 'line', x: X, y: initial, muted: true, dashed: true },
      {
        name: equation === 'heat' ? 'temperature u(x, t)' : 'displacement u(x, t)',
        type: 'line',
        x: X,
        y: profile,
        slot: 0,
      },
    ],
    [initial, profile, equation],
  )
  const handles = useMemo<Handle[]>(
    () => [{ kind: 'x', at: centre.value, label: 'bump', onDrag: centre.set }],
    [centre],
  )

  const area = profile.length ? profile.reduce((s, v) => s + v, 0) * (X[1] - X[0]) : 0
  const captions: Record<Equation, string> = {
    heat: 'Heat equation uₜ = D uₓₓ on [0, 1] with the ends held at 0. The bump spreads out and flattens; sharp features vanish first. Nothing travels: the peak stays where it was and sinks.',
    wave: 'Wave equation uₜₜ = uₓₓ on [0, 1] with fixed ends and the string released from rest. The bump splits into two half-height copies that travel at speed 1, flip sign when they reflect off the ends, and recombine at t = 2. Nothing smooths.',
    laplace:
      'Laplace equation uₓₓ + u_yy = 0 on the unit square: the top edge holds the bump and the other edges are held at 0. There is no time; u is the steady temperature. The bump fades smoothly into the interior, and every interior value is the average of its surroundings.',
  }

  return (
    <Interactive
      title="The heat, wave and Laplace equations"
      caption={`${captions[equation]} Drag the vertical line or use the slider to move the bump.`}
      controls={
        <>
          <ParamChoice
            label="equation"
            value={equation}
            onChange={setEquation}
            options={[
              { value: 'heat', label: 'heat' },
              { value: 'wave', label: 'wave' },
              { value: 'laplace', label: 'Laplace' },
            ]}
          />
          <ParamSlider label="bump position" param={centre} />
          {equation !== 'laplace' && <ParamSlider label="time t" param={time} withArrows />}
        </>
      }
      readout={
        equation === 'laplace' ? (
          <Readout label="u at the centre (0.5, 0.5)" value={formatNumber(laplace ? laplace[25][25] : 0)} />
        ) : (
          <>
            <Readout label="max u" value={formatNumber(Math.max(...profile))} />
            <Readout label="∫ u dx" value={formatNumber(area)} />
          </>
        )
      }
    >
      {equation === 'laplace' && laplace ? (
        <Heatmap
          x={GRID}
          y={GRID}
          z={laplace}
          xLabel="x"
          yLabel="y"
          valueLabel="u"
          scale="sequential"
          range={[0, 1]}
          handles={handles}
          height={380}
        />
      ) : (
        <XYChart
          height={300}
          xLabel="x"
          yLabel="u"
          series={series}
          handles={handles}
          xRange={[0, 1]}
          yRange={[-1.1, 1.1]}
        />
      )}
    </Interactive>
  )
}
