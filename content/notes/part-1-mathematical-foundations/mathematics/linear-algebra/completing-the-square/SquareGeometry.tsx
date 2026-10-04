import { useMemo } from 'react'
import {
  Annotation,
  Curve,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Points,
  Readout,
  seriesColor,
  Shapes,
  slider,
  Tex,
  useAxis,
  useFigureState,
  useTheme,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn/foundation/tensor'

const B = { min: 1, max: 12, step: 0.5 }
const C = { min: 1, max: 60, step: 1 }
// The largest completed square, at b and c both maximal, sets a fixed view so the picture does not rescale.
const VIEW: [number, number] = [0, Math.ceil(Math.sqrt(C.max + (B.max / 2) ** 2))]
const PARABOLA_X = toFlat(linspace(-20, 10, 301))
/** Palette slots of the four pieces: the square x², the two strips x · (b/2), and the missing corner (b/2)². */
const SLOT = { square: 0, stripA: 1, stripB: 2, corner: 4 } as const

type Piece = {
  name: string
  x0: number
  x1: number
  y0: number
  y1: number
  slot: number
  label: string
  dashed?: boolean
}

/**
 * The geometric solution of x² + bx = c.
 * Left: the square is partitioned into four pieces: the square x², two strips x · (b/2), and the dashed missing corner
 * (b/2)² that completes the square.
 * Right: the parabola y = x² + bx − c with its vertex handle and roots.
 */
export function SquareGeometry() {
  const state = useFigureState({
    b: slider(B.min, B.max, 10, { step: B.step, label: 'b' }),
    c: slider(C.min, C.max, 39, { step: C.step, label: 'c (area of the gnomon)' }),
  })
  const { resolved: mode } = useTheme()
  const col = (slot: number) => seriesColor(mode, slot)
  const h = state.b / 2
  const side = Math.sqrt(state.c + h * h)
  const x = side - h

  const bVal = state.b
  const cVal = state.c
  const bHalf = formatNumber(h)
  const bHalfSq = formatNumber(h * h)
  const totalArea = formatNumber(cVal + h * h)
  const sideStr = formatNumber(side)
  const xStr = formatNumber(x)
  const rootNeg = formatNumber(-h - side)

  const pieces = useMemo(
    (): Piece[] => [
      { name: 'square x²', x0: 0, x1: x, y0: 0, y1: x, slot: SLOT.square, label: `x² = ${formatNumber(x * x)}` },
      {
        name: 'strip x · (b/2)',
        x0: x,
        x1: side,
        y0: 0,
        y1: x,
        slot: SLOT.stripA,
        label: `x · (b/2) = ${formatNumber(x * h)}`,
      },
      {
        name: 'strip (b/2) · x',
        x0: 0,
        x1: x,
        y0: x,
        y1: side,
        slot: SLOT.stripB,
        label: `(b/2) · x = ${formatNumber(h * x)}`,
      },
      {
        name: 'missing corner (b/2)²',
        x0: x,
        x1: side,
        y0: x,
        y1: side,
        slot: SLOT.corner,
        label: `(b/2)² = +${bHalfSq}`,
        dashed: true,
      },
    ],
    [x, side, h, bHalfSq],
  )
  const fills = useMemo(
    () =>
      pieces.map((p) => ({
        contours: [
          [
            [p.x0, p.y0],
            [p.x1, p.y0],
            [p.x1, p.y1],
            [p.x0, p.y1],
          ] as [number, number][],
        ],
        tone: p.slot,
        opacity: 0.2,
      })),
    [pieces],
  )

  const curve = useMemo(() => {
    const roots = [x, -h - side]
    return [
      {
        name: 'Parabola',
        x: PARABOLA_X,
        y: PARABOLA_X.map((t) => t * t + bVal * t - cVal),
        slot: SLOT.square,
      },
      {
        name: 'Roots',
        x: roots,
        y: [0, 0],
        emphasis: true,
      },
    ] as const
  }, [bVal, cVal, h, side, x])

  // The vertex (−b/2, −c − b²/4) is a location on the chart, so dragging it sets b and c: b = −2vₓ, c = −v_y − vₓ².

  const polySignB = bVal >= 0 ? `+ ${formatNumber(bVal)}x` : `- ${formatNumber(Math.abs(bVal))}x`
  const polySignC = cVal >= 0 ? `- ${formatNumber(cVal)}` : `+ ${formatNumber(Math.abs(cVal))}`
  const polyTex = String.raw`y = x^2 ${polySignB} ${polySignC}`
  const vertexTex = String.raw`\text{Vertex: } (${formatNumber(-h)}, ${formatNumber(-cVal - h * h)}) \quad \text{Roots: } x = ${xStr}, \; ${rootNeg}`

  const xAxis = useAxis({ label: 'x', range: [-20, 10] })
  const yAxis = useAxis({ label: 'y', range: [-80, 40] })
  const sqX = useAxis({ label: 'x', range: VIEW })
  const sqY = useAxis({ label: 'y', range: VIEW, equal: sqX })
  return (
    <Figure
      title="Completing the square, as a construction"
      caption="Solve x² + bx = c. Left: the square of side x and two strips of width b/2 form an L-shaped gnomon of area c. The gnomon is a square of side x + b/2 missing its dashed corner of area (b/2)². Adding that corner to both sides gives (x + b/2)² = c + (b/2)², so the side of the completed square is known, and x is that side minus b/2. Right: the parabola x² + bx − c crosses zero at the same x, and at a negative root that has no geometric meaning. Drag the vertex, or use the sliders."
      state={state}
      equation={
        <div className="space-y-2.5 py-1 text-xs sm:text-sm">
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <span className="text-[11px] font-semibold tracking-wider text-muted-foreground uppercase">
              1. Equation &amp; Gnomon:
            </span>
            <Tex>
              {String.raw`\textcolor{${col(SLOT.square)}}{x^2} + \textcolor{${col(SLOT.stripA)}}{x \cdot ${bHalf}} + \textcolor{${col(SLOT.stripB)}}{${bHalf} \cdot x} = ${formatNumber(cVal)} \quad \Longleftrightarrow \quad x^2 + ${formatNumber(bVal)}x = ${formatNumber(cVal)}`}
            </Tex>
          </div>
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <span className="text-[11px] font-semibold tracking-wider text-muted-foreground uppercase">
              2. Complete the square:
            </span>
            <Tex>
              {String.raw`\textcolor{${col(SLOT.square)}}{x^2} + \textcolor{${col(SLOT.stripA)}}{${bHalf}x} + \textcolor{${col(SLOT.stripB)}}{${bHalf}x} + \textcolor{${col(SLOT.corner)}}{${bHalfSq}} = ${formatNumber(cVal)} + \textcolor{${col(SLOT.corner)}}{${bHalfSq}} = \mathbf{${totalArea}}`}
            </Tex>
          </div>
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <span className="text-[11px] font-semibold tracking-wider text-muted-foreground uppercase">
              3. Factor into a single square:
            </span>
            <Tex>
              {String.raw`\left(x + \textcolor{${col(SLOT.corner)}}{${bHalf}}\right)^2 = \mathbf{${totalArea}} = \left(${sideStr}\right)^2`}
            </Tex>
          </div>
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <span className="text-[11px] font-semibold tracking-wider text-muted-foreground uppercase">
              4. Geometric solution:
            </span>
            <Tex>
              {String.raw`x + ${bHalf} = ${sideStr} \quad \Longrightarrow \quad \mathbf{x = ${sideStr} - ${bHalf} = ${xStr}} \quad \left(\text{algebraic root: } x = ${rootNeg}\right)`}
            </Tex>
          </div>
        </div>
      }
      readouts={
        <>
          <Readout label="gnomon area c" value={formatNumber(state.c)} />
          <Readout label="missing corner (b/2)²" value={formatNumber(h * h)} />
          <Readout label="completed square c + (b/2)²" value={formatNumber(state.c + h * h)} />
          <Readout label="its side x + b/2" value={formatNumber(side)} />
          <Readout label="x" value={formatNumber(x)} />
        </>
      }
    >
      <div className="grid gap-6 md:grid-cols-2">
        <div className="min-w-0 space-y-2">
          <div className="rounded-md bg-muted/30 px-3 py-1.5 text-center">
            <div className="text-xs font-medium sm:text-sm">
              <Tex>{String.raw`\text{Completed square: } (x + ${bHalf})^2 = ${totalArea}`}</Tex>
            </div>
            <div className="text-[11px] text-muted-foreground">
              <Tex>{String.raw`\text{Side } = ${sideStr}, \quad \mathbf{x = ${xStr}}`}</Tex>
            </div>
          </div>
          <Plot x={sqX} y={sqY} height={360}>
            <Shapes shapes={fills} />
            {pieces.map((p) => (
              <Curve
                key={p.name}
                name={p.name}
                x={[p.x0, p.x1, p.x1, p.x0, p.x0]}
                y={[p.y0, p.y0, p.y1, p.y1, p.y0]}
                slot={p.slot}
                dashed={p.dashed}
              />
            ))}
            {pieces.map((p) => (
              <Annotation key={p.name} at={[(p.x0 + p.x1) / 2, (p.y0 + p.y1) / 2]} text={p.label} slot={p.slot} />
            ))}
          </Plot>
        </div>
        <div className="min-w-0 space-y-2">
          <div className="rounded-md bg-muted/30 px-3 py-1.5 text-center">
            <div className="text-xs font-medium sm:text-sm">
              <Tex>{polyTex}</Tex>
            </div>
            <div className="text-[11px] text-muted-foreground">
              <Tex>{vertexTex}</Tex>
            </div>
          </div>
          <Plot x={xAxis} y={yAxis} height={360}>
            <Curve {...curve[0]} />
            <Points {...curve[1]} />
            <Handle
              kind="point"
              at={[-h, -cVal - h * h]}
              label="vertex"
              onDrag={([vx, vy]) => {
                state.set('b', -2 * vx)
                state.set('c', -vy - vx * vx)
              }}
            />
          </Plot>
        </div>
      </div>
    </Figure>
  )
}
