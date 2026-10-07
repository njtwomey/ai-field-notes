import { useContext, useMemo, useState, type ReactNode } from 'react'
import {
  Annotation,
  Bars,
  Curve,
  Figure,
  type FilledShape,
  FrameContext,
  Handle,
  Player,
  Plot,
  Points,
  Readout,
  Segments,
  setting,
  Shapes,
  slider,
  Tex,
  useAxis,
  useFigureState,
  type Vector,
  Vectors,
} from 'aifn-render'
import { mute, seriesColor, useTheme } from 'aifn-render/design'
import { castFan, rayOffsets, worldById } from '../_shared/world'
import { traceRay, type DdaTrace } from './dda'

const WORLD = worldById('pillar')
const DEG = Math.PI / 180
/** The screen of the worked example: 320 pixels wide, a 60° field of view, so f = 160 / tan 30° ≈ 277 pixels. */
const SCREEN_W = 320
const SCREEN_H = 200
const FOV = 60 * DEG
const FOCAL = SCREEN_W / 2 / Math.tan(FOV / 2)
/** The view panel draws 40 slices, each 8 pixels wide. */
const COLUMNS = 40
const OFFSETS = rayOffsets(COLUMNS, FOV)

const fmt = (v: number) => (Number.isFinite(v) ? v.toFixed(3) : '∞')
const cellName = ([cx, cy]: readonly [number, number]) => `(${cx}, ${cy})`
const square = (cx: number, cy: number): [number, number][] => [
  [cx, cy],
  [cx + 1, cy],
  [cx + 1, cy + 1],
  [cx, cy + 1],
]

/** Plots inside a Figure take the frame's height; this gives the plot inside it a fixed height instead. */
function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}

/** What frame k of the trace shows: before any step (k = 0) or just after step k. */
function frameAt(tr: DdaTrace, k: number) {
  const step = k > 0 ? tr.steps[k - 1] : undefined
  const before = k > 1 ? tr.steps[k - 2] : undefined
  return {
    step,
    t: step?.t ?? 0,
    cell: step?.cell ?? tr.cell,
    nextX: step?.nextX ?? tr.nextX,
    nextY: step?.nextY ?? tr.nextY,
    prevX: before?.nextX ?? tr.nextX,
    prevY: before?.nextY ?? tr.nextY,
    hit: step !== undefined && step.material !== 0,
  }
}

function describe(tr: DdaTrace, k: number): string {
  if (tr.inWall)
    return 'The camera stands inside a wall cell, so there is nothing to trace: drag it into an empty cell.'
  const f = frameAt(tr, k)
  if (!f.step)
    return `Step 0: the camera is in cell ${cellName(tr.cell)}. Along the ray, the first vertical grid line is tₓ = ${fmt(tr.nextX)} away and the first horizontal one t_y = ${fmt(tr.nextY)} away.`
  const s = f.step
  const vertical = s.crossed === 'vertical'
  const compare = vertical
    ? `tₓ = ${fmt(f.prevX)} < t_y = ${fmt(f.prevY)}`
    : `t_y = ${fmt(f.prevY)} ≤ tₓ = ${fmt(f.prevX)}`
  const line = vertical ? `vertical line x = ${s.line}` : `horizontal line y = ${s.line}`
  const grow = vertical
    ? `tₓ grows by δₓ = ${fmt(tr.deltaX)} to ${fmt(s.nextX)}`
    : `t_y grows by δ_y = ${fmt(tr.deltaY)} to ${fmt(s.nextY)}`
  const end = s.material !== 0 ? `, a wall: the ray stops at t = ${fmt(s.t)}.` : `; ${grow}.`
  return `Step ${k}: ${compare}, so the ray crosses the ${line} at t = ${fmt(s.t)} and enters cell ${cellName(s.cell)}${end}`
}

/** One ray traced through a grid map by the digital differential analyser, step by step, with the view it renders. */
export function DdaWalkthrough() {
  const state = useFigureState({
    x: slider(1.05, 6.95, 1.5, { step: 0.05, onChart: true }),
    y: slider(1.05, 6.95, 1.5, { step: 0.05, onChart: true }),
    heading: slider(-180, 180, 45, { step: 1, label: <Tex>{'\\text{heading } \\theta \\; (^\\circ)'}</Tex> }),
    offset: slider(-30, 30, -15, {
      step: 1,
      label: <Tex>{'\\text{ray angle from the heading } \\alpha \\; (^\\circ)'}</Tex>,
    }),
    fan: setting(false, { label: 'draw every column’s ray' }),
  })
  const [frame, setFrame] = useState(0)

  const angle = (state.heading + state.offset) * DEG
  const tr = useMemo(() => traceRay(WORLD, state.x, state.y, angle), [state.x, state.y, angle])
  const count = tr.steps.length + 1
  const k = Math.min(frame, count - 1)
  const f = frameAt(tr, k)
  const last = tr.steps.at(-1)

  const map = useMemo(() => {
    const walls: FilledShape[] = []
    for (let cy = 0; cy < WORLD.height; cy++)
      for (let cx = 0; cx < WORLD.width; cx++)
        if (WORLD.cells[cy * WORLD.width + cx] !== 0) walls.push({ contours: [square(cx, cy)], tone: 'muted' })
    const grid: { from: [number, number]; to: [number, number] }[] = []
    for (let i = 0; i <= WORLD.width; i++) grid.push({ from: [i, 0], to: [i, WORLD.height] })
    for (let j = 0; j <= WORLD.height; j++) grid.push({ from: [0, j], to: [WORLD.width, j] })
    return { walls, grid }
  }, [])

  // The whole ray, the crossings and the fan depend on the pose only; the frame picks how much of them to show.
  const ray = useMemo(() => {
    const end = tr.steps.at(-1)?.t ?? 0
    const point = (t: number) => [tr.x + t * tr.dx, tr.y + t * tr.dy]
    const crossings = tr.steps.map((s) => ({ crossed: s.crossed, p: point(s.t) }))
    return { full: { x: [tr.x, point(end)[0]], y: [tr.y, point(end)[1]] }, crossings }
  }, [tr])

  const shown = useMemo(() => {
    const now = frameAt(tr, k)
    const point = (t: number): [number, number] => [tr.x + t * tr.dx, tr.y + t * tr.dy]
    const before = ray.crossings.slice(0, k)
    const pick = (kind: 'vertical' | 'horizontal') => {
      const ps = before.filter((c) => c.crossed === kind)
      return { x: ps.map((c) => c.p[0]), y: ps.map((c) => c.p[1]) }
    }
    const end = point(now.t)
    const candidates = now.hit || tr.inWall ? null : { vertical: point(now.nextX), horizontal: point(now.nextY) }
    return {
      travelled: { x: [tr.x, end[0]], y: [tr.y, end[1]] },
      end,
      vertical: pick('vertical'),
      horizontal: pick('horizontal'),
      candidates,
      cell: [{ contours: [square(...now.cell)], tone: 0, opacity: 0.25 }] as FilledShape[],
    }
  }, [ray, tr, k])

  const view = useMemo(() => {
    const heading = state.heading * DEG
    const hits = castFan(WORLD, { x: state.x, y: state.y, theta: heading }, OFFSETS)
    const faces = {
      vertical: { x: [] as number[], h: [] as number[] },
      horizontal: { x: [] as number[], h: [] as number[] },
    }
    const rays: { from: [number, number]; to: [number, number] }[] = []
    hits.forEach((hit, i) => {
      const p = hit.distance * Math.cos(OFFSETS[i])
      const face = hit.side === 0 ? faces.vertical : faces.horizontal
      face.x.push(((i + 0.5) * SCREEN_W) / COLUMNS)
      face.h.push(FOCAL / p / 2)
      rays.push({ from: [state.x, state.y], to: [hit.x, hit.y] })
    })
    const below = (h: number[]) => h.map((v) => -v)
    return {
      rays,
      vertical: { x: faces.vertical.x, up: faces.vertical.h, down: below(faces.vertical.h) },
      horizontal: { x: faces.horizontal.x, up: faces.horizontal.h, down: below(faces.horizontal.h) },
      heading: [
        {
          from: [state.x, state.y],
          to: [state.x + Math.cos(heading), state.y + Math.sin(heading)],
          head: 8,
          width: 1.5,
        },
      ] as Vector[],
    }
  }, [state.x, state.y, state.heading])

  // The ray's place on the screen: c = tan α / tan(φ / 2) in (−1, 1), so pixel (W / 2)(1 + c).
  const column = (SCREEN_W / 2) * (1 + Math.tan(state.offset * DEG) / Math.tan(FOV / 2))
  const perp = last && last.material !== 0 ? last.t * Math.cos(state.offset * DEG) : NaN

  // Faces met across a horizontal grid line are drawn darker, as in Wolfenstein 3D.
  const mode = useTheme().resolved
  const shaded = mute(seriesColor(mode, 0), '#000000', 0.6)

  const xAxis = useAxis({ label: 'x', range: [0, WORLD.width], nice: false, integer: true })
  const yAxis = useAxis({
    label: 'y (downwards)',
    range: [0, WORLD.height],
    nice: false,
    integer: true,
    inverse: true,
    equal: xAxis,
  })
  const cAxis = useAxis({ label: 'screen column (pixels)', range: [0, SCREEN_W], nice: false })
  const hAxis = useAxis({ label: 'pixels from the horizon', range: [-SCREEN_H / 2, SCREEN_H / 2], nice: false })

  return (
    <Figure
      title="One ray through the grid, step by step"
      state={state}
      caption={`${describe(tr, k)} Left: the map from above, with y growing downwards as in the game's map and angles measured from the +x axis towards +y. Walls are grey; the camera is the large dot and the arrow is its heading. The ray (blue) is drawn up to the current crossing, and the dashed line is the rest of its way to the wall. Dots are the crossings so far, on vertical grid lines and on horizontal ones; the two diamonds are the next crossing of each kind, and the analyser always takes the nearer one. The shaded square is the current cell. Drag the camera, or set the heading and the ray's angle within the 60° view. Right: the view this camera renders on a 320-pixel screen, one slice per 8-pixel column of height f / p, with f ≈ 277 pixels; the dashed line marks this ray's column.`}
      controls={<Player value={k} onChange={setFrame} count={count} label="step" />}
      readouts={
        <>
          <Readout label="step" value={`${k} of ${count - 1}`} />
          <Readout label="cell" value={cellName(f.cell)} />
          <Readout label={<Tex>{'t'}</Tex>} value={fmt(f.t)} />
          <Readout label={<Tex>{'t_x \\text{ (next vertical line)}'}</Tex>} value={f.hit ? '·' : fmt(f.nextX)} />
          <Readout label={<Tex>{'t_y \\text{ (next horizontal line)}'}</Tex>} value={f.hit ? '·' : fmt(f.nextY)} />
          <Readout label={<Tex>{'p = t \\cos\\alpha'}</Tex>} value={f.hit ? fmt(perp) : '·'} />
          <Readout label={<Tex>{'f / p \\text{ (pixels)}'}</Tex>} value={f.hit ? fmt(FOCAL / perp) : '·'} />
        </>
      }
    >
      <div className="grid grid-cols-1 items-start gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} fitHeight>
          <Shapes shapes={map.walls} muted />
          <Segments segments={map.grid} muted width={0.75} />
          <Shapes shapes={shown.cell} slot={0} live />
          {state.fan && <Segments segments={view.rays} muted />}
          <Vectors vectors={view.heading} />
          <Curve x={ray.full.x} y={ray.full.y} muted dashed width={1} silent />
          <Curve name="ray" x={shown.travelled.x} y={shown.travelled.y} slot={0} live silent />
          <Points name="vertical-line crossings" x={shown.vertical.x} y={shown.vertical.y} slot={1} size={7} live />
          <Points
            name="horizontal-line crossings"
            x={shown.horizontal.x}
            y={shown.horizontal.y}
            slot={2}
            size={7}
            shape={1}
            live
          />
          {shown.candidates && (
            <Points
              name="next vertical-line crossing"
              x={[shown.candidates.vertical[0]]}
              y={[shown.candidates.vertical[1]]}
              slot={1}
              shape={3}
              size={12}
              live
            />
          )}
          {shown.candidates && (
            <Points
              name="next horizontal-line crossing"
              x={[shown.candidates.horizontal[0]]}
              y={[shown.candidates.horizontal[1]]}
              slot={2}
              shape={3}
              size={12}
              live
            />
          )}
          {f.hit && <Points name="hit" x={[shown.end[0]]} y={[shown.end[1]]} emphasis shape={4} size={12} live />}
          <Handle {...state.handle(['x', 'y'], { label: 'camera' })} />
        </Plot>
        <FixedHeight height={300}>
          <Plot x={cAxis} y={hAxis} legend={false}>
            <Bars
              name="face across a vertical line"
              x={view.vertical.x}
              y={view.vertical.up}
              width={SCREEN_W / COLUMNS}
              slot={0}
            />
            <Bars
              name="face across a vertical line"
              x={view.vertical.x}
              y={view.vertical.down}
              width={SCREEN_W / COLUMNS}
              slot={0}
            />
            <Bars
              name="face across a horizontal line"
              x={view.horizontal.x}
              y={view.horizontal.up}
              width={SCREEN_W / COLUMNS}
              color={shaded}
            />
            <Bars
              name="face across a horizontal line"
              x={view.horizontal.x}
              y={view.horizontal.down}
              width={SCREEN_W / COLUMNS}
              color={shaded}
            />
            <Annotation x={column} text="this ray" dashed live />
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
