import { useMemo } from 'react'
import {
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from 'aifn-render'
import { linspace } from '@/lib/math'

/** Closed outline of the rectangle [x0, x1] × [y0, y1]. */
function rect(name: string, x0: number, x1: number, y0: number, y1: number, style: Partial<XYSeries>): XYSeries {
  return { name, type: 'line', x: [x0, x1, x1, x0, x0], y: [y0, y0, y1, y1, y0], ...style }
}

const B = { min: 1, max: 12, step: 0.5 }
const C = { min: 1, max: 60, step: 1 }
// The largest completed square, at b and c both maximal, sets a fixed view so the picture does not rescale.
const VIEW: [number, number] = [0, Math.ceil(Math.sqrt(C.max + (B.max / 2) ** 2))]
const PARABOLA_X = linspace(-20, 10, 301)

/**
 * The geometric solution of x² + bx = c. Left: the square x² and two strips of width b/2 form an L-shaped gnomon of
 * area c; adding the missing corner (b/2)² completes a square of side √(c + b²/4), which gives x. Right: the parabola
 * x² + bx − c, whose positive root is the same x.
 */
export function SquareGeometry() {
  const b = useParam(10, B)
  const c = useParam(39, C)
  const h = b.value / 2
  const side = Math.sqrt(c.value + h * h)
  const x = side - h

  const pieces = useMemo(
    (): XYSeries[] => [
      rect('square x²', 0, x, 0, x, { slot: 0 }),
      rect('strips 2 · x · b/2', x, side, 0, x, { slot: 1 }),
      rect('strips 2 · x · b/2', 0, x, x, side, { slot: 1 }),
      rect('missing corner (b/2)²', x, side, x, side, { slot: 2, dashed: true }),
    ],
    [x, side],
  )

  const curve = useMemo((): XYSeries[] => {
    const roots = [x, -h - side]
    return [
      {
        name: 'x² + bx − c',
        type: 'line',
        x: PARABOLA_X,
        y: PARABOLA_X.map((t) => t * t + b.value * t - c.value),
        slot: 0,
      },
      { name: 'roots', type: 'scatter', x: roots, y: [0, 0], emphasis: true },
    ]
  }, [b.value, c.value, h, side, x])

  // The vertex (−b/2, −c − b²/4) is a location on the chart, so dragging it sets b and c: b = −2vₓ, c = −v_y − vₓ².
  const handles: Handle[] = [
    {
      kind: 'point',
      at: [-h, -c.value - h * h],
      label: 'vertex',
      onDrag: ([vx, vy]) => {
        b.set(-2 * vx)
        c.set(-vy - vx * vx)
      },
    },
  ]

  return (
    <Interactive
      title="Completing the square, as a construction"
      caption="Solve x² + bx = c. Left: the square of side x and two strips of width b/2 form an L-shaped gnomon of area c. The gnomon is a square of side x + b/2 missing its dashed corner of area (b/2)². Adding that corner to both sides gives (x + b/2)² = c + (b/2)², so the side of the completed square is known, and x is that side minus b/2. Right: the parabola x² + bx − c crosses zero at the same x, and at a negative root that has no geometric meaning. Drag the vertex, or use the sliders."
      controls={
        <>
          <ParamSlider label="b" param={b} />
          <ParamSlider label="c (area of the gnomon)" param={c} />
        </>
      }
      readout={
        <>
          <Readout label="gnomon area c" value={formatNumber(c.value)} />
          <Readout label="missing corner (b/2)²" value={formatNumber(h * h)} />
          <Readout label="completed square c + (b/2)²" value={formatNumber(c.value + h * h)} />
          <Readout label="its side x + b/2" value={formatNumber(side)} />
          <Readout label="x" value={formatNumber(x)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <div className="min-w-0 space-y-1">
          <div className="text-center text-xs text-muted-foreground">
            x² + {formatNumber(b.value)}x = {formatNumber(c.value)}
          </div>
          <XYChart series={pieces} xRange={VIEW} yRange={VIEW} equalAspect />
        </div>
        <div className="min-w-0 space-y-1">
          <div className="text-center text-xs text-muted-foreground">y = x² + bx − c</div>
          <XYChart series={curve} xRange={[-20, 10]} yRange={[-80, 40]} handles={handles} height={360} />
        </div>
      </div>
    </Interactive>
  )
}
