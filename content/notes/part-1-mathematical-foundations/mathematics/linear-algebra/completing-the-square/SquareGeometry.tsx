import { useMemo } from 'react'
import {
  Interactive,
  ParamSlider,
  Readout,
  Tex,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYRect,
  type XYSeries,
} from 'aifn-render'
import { linspace } from '@/lib/math'

const B = { min: 1, max: 12, step: 0.5 }
const C = { min: 1, max: 60, step: 1 }
// The largest completed square, at b and c both maximal, sets a fixed view so the picture does not rescale.
const VIEW: [number, number] = [0, Math.ceil(Math.sqrt(C.max + (B.max / 2) ** 2))]
const PARABOLA_X = linspace(-20, 10, 301)

/**
 * The geometric solution of x² + bx = c.
 * Left: The square is partitioned into 4 distinct quadrants:
 *   1. Bottom-left (Blue): square x²
 *   2. Bottom-right (Amber): strip x · (b/2)
 *   3. Top-left (Emerald): strip (b/2) · x
 *   4. Top-right (Rose, dashed): missing corner (b/2)² that completes the square.
 * Right: The parabola y = x² + bx − c with reactive title, vertex handle, and roots.
 */
export function SquareGeometry() {
  const b = useParam(10, B)
  const c = useParam(39, C)
  const h = b.value / 2
  const side = Math.sqrt(c.value + h * h)
  const x = side - h

  const bVal = b.value
  const cVal = c.value
  const bHalf = formatNumber(h)
  const bHalfSq = formatNumber(h * h)
  const totalArea = formatNumber(cVal + h * h)
  const sideStr = formatNumber(side)
  const xStr = formatNumber(x)
  const rootNeg = formatNumber(-h - side)

  // 4 distinct quadrants in Cartesian coordinates with matching colors
  const quadrants = useMemo(
    (): XYRect[] => [
      {
        name: 'Square x²',
        x0: 0,
        x1: x,
        y0: 0,
        y1: x,
        stroke: '#3b82f6',
        strokeWidth: 2,
        fill: 'rgba(59, 130, 246, 0.2)',
        label: 'x²',
        labelSub: `${formatNumber(x * x)}`,
        labelColor: '#3b82f6',
      },
      {
        name: 'Strip x · (b/2)',
        x0: x,
        x1: side,
        y0: 0,
        y1: x,
        stroke: '#f59e0b',
        strokeWidth: 2,
        fill: 'rgba(245, 158, 11, 0.2)',
        label: 'x · (b/2)',
        labelSub: `${formatNumber(x * h)}`,
        labelColor: '#f59e0b',
      },
      {
        name: 'Strip (b/2) · x',
        x0: 0,
        x1: x,
        y0: x,
        y1: side,
        stroke: '#10b981',
        strokeWidth: 2,
        fill: 'rgba(16, 185, 129, 0.2)',
        label: '(b/2) · x',
        labelSub: `${formatNumber(h * x)}`,
        labelColor: '#10b981',
      },
      {
        name: 'Missing corner (b/2)²',
        x0: x,
        x1: side,
        y0: x,
        y1: side,
        stroke: '#ec4899',
        strokeWidth: 2,
        fill: 'rgba(236, 72, 153, 0.2)',
        dashed: true,
        label: '(b/2)²',
        labelSub: `+${bHalfSq}`,
        labelColor: '#ec4899',
      },
    ],
    [x, side, h, bHalfSq],
  )

  // Legend dummy series so colors are explained in the chart legend
  const pieces = useMemo(
    (): XYSeries[] => [
      { name: 'Square x²', type: 'line', x: [], y: [], color: '#3b82f6' },
      { name: 'Strips x · (b/2)', type: 'line', x: [], y: [], color: '#f59e0b' },
      { name: 'Missing corner (b/2)²', type: 'line', x: [], y: [], color: '#ec4899', dashed: true },
    ],
    [],
  )

  const curve = useMemo((): XYSeries[] => {
    const roots = [x, -h - side]
    return [
      {
        name: 'Parabola',
        type: 'line',
        x: PARABOLA_X,
        y: PARABOLA_X.map((t) => t * t + bVal * t - cVal),
        color: '#3b82f6',
      },
      {
        name: 'Roots',
        type: 'scatter',
        x: roots,
        y: [0, 0],
        emphasis: true,
        pointColors: ['#10b981', '#6b7280'],
      },
    ]
  }, [bVal, cVal, h, side, x])

  // The vertex (−b/2, −c − b²/4) is a location on the chart, so dragging it sets b and c: b = −2vₓ, c = −v_y − vₓ².
  const handles: Handle[] = [
    {
      kind: 'point',
      at: [-h, -cVal - h * h],
      label: 'vertex',
      onDrag: ([vx, vy]) => {
        b.set(-2 * vx)
        c.set(-vy - vx * vx)
      },
    },
  ]

  const polySignB = bVal >= 0 ? `+ ${formatNumber(bVal)}x` : `- ${formatNumber(Math.abs(bVal))}x`
  const polySignC = cVal >= 0 ? `- ${formatNumber(cVal)}` : `+ ${formatNumber(Math.abs(cVal))}`
  const polyTex = String.raw`y = x^2 ${polySignB} ${polySignC}`
  const vertexTex = String.raw`\text{Vertex: } (${formatNumber(-h)}, ${formatNumber(-cVal - h * h)}) \quad \text{Roots: } x = ${xStr}, \; ${rootNeg}`

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
      equation={
        <div className="space-y-2.5 py-1 text-xs sm:text-sm">
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <span className="font-semibold uppercase tracking-wider text-[11px] text-muted-foreground">
              1. Equation &amp; Gnomon:
            </span>
            <Tex>
              {String.raw`\textcolor{#3b82f6}{x^2} + \textcolor{#f59e0b}{x \cdot ${bHalf}} + \textcolor{#10b981}{${bHalf} \cdot x} = ${formatNumber(cVal)} \quad \Longleftrightarrow \quad x^2 + ${formatNumber(bVal)}x = ${formatNumber(cVal)}`}
            </Tex>
          </div>
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <span className="font-semibold uppercase tracking-wider text-[11px] text-muted-foreground">
              2. Complete the square:
            </span>
            <Tex>
              {String.raw`\textcolor{#3b82f6}{x^2} + \textcolor{#f59e0b}{${bHalf}x} + \textcolor{#10b981}{${bHalf}x} + \textcolor{#ec4899}{${bHalfSq}} = ${formatNumber(cVal)} + \textcolor{#ec4899}{${bHalfSq}} = \mathbf{${totalArea}}`}
            </Tex>
          </div>
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <span className="font-semibold uppercase tracking-wider text-[11px] text-muted-foreground">
              3. Factor into a single square:
            </span>
            <Tex>
              {String.raw`\left(x + \textcolor{#ec4899}{${bHalf}}\right)^2 = \mathbf{${totalArea}} = \left(${sideStr}\right)^2`}
            </Tex>
          </div>
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <span className="font-semibold uppercase tracking-wider text-[11px] text-muted-foreground">
              4. Geometric solution:
            </span>
            <Tex>
              {String.raw`x + ${bHalf} = ${sideStr} \quad \Longrightarrow \quad \mathbf{x = ${sideStr} - ${bHalf} = ${xStr}} \quad \left(\text{algebraic root: } x = ${rootNeg}\right)`}
            </Tex>
          </div>
        </div>
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
      <div className="grid gap-6 md:grid-cols-2">
        <div className="min-w-0 space-y-2">
          <div className="rounded-md bg-muted/30 px-3 py-1.5 text-center">
            <div className="text-xs sm:text-sm font-medium">
              <Tex>{String.raw`\text{Completed square: } (x + ${bHalf})^2 = ${totalArea}`}</Tex>
            </div>
            <div className="text-[11px] text-muted-foreground">
              <Tex>{String.raw`\text{Side } = ${sideStr}, \quad \mathbf{x = ${xStr}}`}</Tex>
            </div>
          </div>
          <XYChart
            series={pieces}
            rects={quadrants}
            xRange={VIEW}
            yRange={VIEW}
            xLabel="x"
            yLabel="y"
            equalAspect
          />
        </div>
        <div className="min-w-0 space-y-2">
          <div className="rounded-md bg-muted/30 px-3 py-1.5 text-center">
            <div className="text-xs sm:text-sm font-medium">
              <Tex>{polyTex}</Tex>
            </div>
            <div className="text-[11px] text-muted-foreground">
              <Tex>{vertexTex}</Tex>
            </div>
          </div>
          <XYChart
            series={curve}
            xRange={[-20, 10]}
            yRange={[-80, 40]}
            xLabel="x"
            yLabel="y"
            handles={handles}
            height={360}
          />
        </div>
      </div>
    </Interactive>
  )
}
