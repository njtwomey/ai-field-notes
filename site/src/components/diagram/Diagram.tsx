import { useMemo, type ReactNode } from 'react'
import { useTheme } from '@/components/theme-provider'
import { MathText } from '@/components/content/MathText'
import { chrome, seriesColor, type Mode } from '@/components/viz/palette'
import type { DiagramEdge, DiagramGroup, DiagramNode, DiagramSpec, Direction, Side, Tone } from './types'

type Pt = { x: number; y: number }
type Box = { x0: number; y0: number; x1: number; y1: number }

const DEFAULT_SIZE: Record<NonNullable<DiagramNode['shape']>, [number, number]> = {
  box: [2.4, 0.8],
  pill: [2.4, 0.8],
  circle: [0.9, 0.9],
  latent: [0.9, 0.9],
  noise: [0.9, 0.9],
  op: [0.5, 0.5],
  factor: [0.26, 0.26],
  encoder: [1.6, 2],
  decoder: [1.6, 2],
  stack: [2.4, 0.8],
  dot: [0.14, 0.14],
  text: [1.8, 0.6],
}
const ROUND = new Set(['circle', 'latent', 'noise', 'op', 'dot'])
const ACCENT_SLOT = 0
/** Ratio of the narrow to the wide side of an encoder or decoder trapezoid. */
const TAPER = 0.45
const LABEL_H = 0.42

function size(n: DiagramNode): [number, number] {
  const [w, h] = DEFAULT_SIZE[n.shape ?? 'box']
  return [n.w ?? w, n.h ?? h]
}

function extent(n: DiagramNode): Box {
  const [w, h] = size(n)
  return { x0: n.x - w / 2, y0: n.y - h / 2, x1: n.x + w / 2, y1: n.y + h / 2 }
}

function toneColour(mode: Mode, tone: Tone | undefined): string {
  const c = chrome(mode)
  if (tone === undefined || tone === 'ink') return c.ink
  if (tone === 'neutral') return c.muted
  return seriesColor(mode, tone)
}

/** The point on a node's outline on the given side. */
function port(n: DiagramNode, side: Side): Pt {
  const [w, h] = size(n)
  const rx = w / 2
  const ry = h / 2
  switch (side) {
    case 'n':
      return { x: n.x, y: n.y - ry }
    case 's':
      return { x: n.x, y: n.y + ry }
    case 'e':
      return { x: n.x + rx, y: n.y }
    case 'w':
      return { x: n.x - rx, y: n.y }
  }
}

/** Where the ray from a node's centre towards `toward` leaves its outline (circles and rectangles). */
function boundary(n: DiagramNode, toward: Pt): Pt {
  const [w, h] = size(n)
  const dx = toward.x - n.x
  const dy = toward.y - n.y
  const len = Math.hypot(dx, dy) || 1
  if (ROUND.has(n.shape ?? 'box')) {
    const r = Math.min(w, h) / 2
    return { x: n.x + (dx / len) * r, y: n.y + (dy / len) * r }
  }
  const t = Math.min(dx === 0 ? Infinity : w / 2 / Math.abs(dx), dy === 0 ? Infinity : h / 2 / Math.abs(dy))
  return { x: n.x + dx * t, y: n.y + dy * t }
}

/** The side of `from` that faces `to`. */
function facing(from: Pt, to: Pt): Side {
  const dx = to.x - from.x
  const dy = to.y - from.y
  if (Math.abs(dx) >= Math.abs(dy)) return dx >= 0 ? 'e' : 'w'
  return dy >= 0 ? 's' : 'n'
}

const horizontal = (s: Side) => s === 'e' || s === 'w'

function parseEnd(ref: string): { id: string; side?: Side } {
  const [id, side] = ref.split(':')
  return { id, side: side as Side | undefined }
}

/** Corner points of an edge in grid units, from port to port. */
function routeEdge(e: DiagramEdge, byId: Map<string, DiagramNode>): Pt[] {
  const a = parseEnd(e.from)
  const b = parseEnd(e.to)
  const na = byId.get(a.id)
  const nb = byId.get(b.id)
  if (!na || !nb) throw new Error(`diagram edge ${e.from} → ${e.to}: unknown node`)
  const via = (e.via ?? []).map(([x, y]) => ({ x, y }))
  if ((e.route === 'straight' || e.route === 'curve') && !a.side && !b.side) {
    // Direct lines leave each node along the line itself, so diagonal edges meet circles cleanly.
    const firstStop = via[0] ?? nb
    const lastStop = via.at(-1) ?? na
    return [boundary(na, firstStop), ...via, boundary(nb, lastStop)]
  }
  const firstTarget = via[0] ?? nb
  const lastSource = via.at(-1) ?? na
  const sa = a.side ?? facing(na, firstTarget)
  const sb = b.side ?? facing(nb, lastSource)
  const p0 = port(na, sa)
  const p1 = port(nb, sb)
  if (e.route === 'straight' || e.route === 'curve') return [p0, ...via, p1]
  const pts: Pt[] = [p0]
  const stops = [...via, p1]
  let hFirst = horizontal(sa)
  for (let i = 0; i < stops.length; i++) {
    const prev = pts[pts.length - 1]
    const next = stops[i]
    const last = i === stops.length - 1
    if (prev.x !== next.x && prev.y !== next.y) {
      if (last && via.length === 0 && horizontal(sa) === horizontal(sb)) {
        // Same orientation at both ends: two elbows at the midpoint.
        if (horizontal(sa)) {
          const mx = (prev.x + next.x) / 2
          pts.push({ x: mx, y: prev.y }, { x: mx, y: next.y })
        } else {
          const my = (prev.y + next.y) / 2
          pts.push({ x: prev.x, y: my }, { x: next.x, y: my })
        }
      } else if (last) {
        // Arrive along the axis of the target side.
        pts.push(horizontal(sb) ? { x: prev.x, y: next.y } : { x: next.x, y: prev.y })
      } else {
        pts.push(hFirst ? { x: next.x, y: prev.y } : { x: prev.x, y: next.y })
      }
    }
    pts.push(next)
    hFirst = true
  }
  return pts.filter((p, i) => i === 0 || p.x !== pts[i - 1].x || p.y !== pts[i - 1].y)
}

/** An SVG path through the points with rounded corners of radius r (pixels). */
function roundedPath(pts: Pt[], r: number): string {
  let d = `M ${pts[0].x} ${pts[0].y}`
  for (let i = 1; i < pts.length - 1; i++) {
    const p = pts[i - 1]
    const c = pts[i]
    const n = pts[i + 1]
    const l1 = Math.hypot(c.x - p.x, c.y - p.y)
    const l2 = Math.hypot(n.x - c.x, n.y - c.y)
    const k = Math.min(r, l1 / 2, l2 / 2)
    const a = { x: c.x - ((c.x - p.x) / l1) * k, y: c.y - ((c.y - p.y) / l1) * k }
    const b = { x: c.x + ((n.x - c.x) / l2) * k, y: c.y + ((n.y - c.y) / l2) * k }
    d += ` L ${a.x} ${a.y} Q ${c.x} ${c.y} ${b.x} ${b.y}`
  }
  const last = pts[pts.length - 1]
  return `${d} L ${last.x} ${last.y}`
}

function arrowHead(tip: Pt, from: Pt, size: number): string {
  const len = Math.hypot(tip.x - from.x, tip.y - from.y) || 1
  const ux = (tip.x - from.x) / len
  const uy = (tip.y - from.y) / len
  const bx = tip.x - ux * size
  const by = tip.y - uy * size
  const px = -uy * size * 0.45
  const py = ux * size * 0.45
  return `${tip.x},${tip.y} ${bx + px},${by + py} ${bx - px},${by - py}`
}

function trapezoid(n: DiagramNode, u: number): string {
  const [w, h] = size(n)
  const cx = n.x * u
  const cy = n.y * u
  const hw = (w * u) / 2
  const hh = (h * u) / 2
  const dir: Direction = n.dir ?? 'right'
  const narrowAtEnd = n.shape === 'encoder'
  // Along a horizontal flow the height tapers; along a vertical flow the width does.
  const wide = 1
  const narrow = TAPER
  const [start, end] = narrowAtEnd ? [wide, narrow] : [narrow, wide]
  if (dir === 'right' || dir === 'left') {
    const [l, r] = dir === 'right' ? [start, end] : [end, start]
    return `${cx - hw},${cy - hh * l} ${cx + hw},${cy - hh * r} ${cx + hw},${cy + hh * r} ${cx - hw},${cy + hh * l}`
  }
  const [t, b] = dir === 'down' ? [start, end] : [end, start]
  return `${cx - hw * t},${cy - hh} ${cx + hw * t},${cy - hh} ${cx + hw * b},${cy + hh} ${cx - hw * b},${cy + hh}`
}

/** A label drawn as HTML over the SVG, positioned in the diagram's pixel coordinates. */
type Overlay = {
  key: string
  box: Box
  /** Font size in diagram pixels; it scales with the rendered diagram. */
  size: number
  align: 'start' | 'center' | 'end'
  content: ReactNode
}

function labelLines(text: string, colour: string): ReactNode {
  return (
    <div className="leading-tight" style={{ color: colour }}>
      {text.split('\n').map((l, i) => (
        <div key={i} className="whitespace-nowrap">
          <MathText text={l} />
        </div>
      ))}
    </div>
  )
}

function groupBox(g: DiagramGroup, byId: Map<string, DiagramNode>): Box {
  if (g.rect) return { x0: g.rect.x, y0: g.rect.y, x1: g.rect.x + g.rect.w, y1: g.rect.y + g.rect.h }
  const pad = g.pad ?? 0.35
  const boxes = (g.around ?? []).map((id) => {
    const n = byId.get(id)
    if (!n) throw new Error(`diagram group ${g.id}: unknown node ${id}`)
    return extent(n)
  })
  const top = g.label && (g.labelAt ?? 'top-left').startsWith('top') ? LABEL_H : 0
  const bottom = g.label && (g.labelAt ?? 'top-left').startsWith('bottom') ? LABEL_H : 0
  return {
    x0: Math.min(...boxes.map((b) => b.x0)) - pad,
    y0: Math.min(...boxes.map((b) => b.y0)) - pad - top,
    x1: Math.max(...boxes.map((b) => b.x1)) + pad,
    y1: Math.max(...boxes.map((b) => b.y1)) + pad + bottom,
  }
}

/**
 * Renders a hand-specified diagram as SVG. Colours come from the data palette and the chrome tokens, so the diagram
 * follows the light and dark themes. Labels are HTML laid over the SVG, positioned in percentages of the diagram and
 * sized in container units, so `$…$` maths renders with KaTeX and stays aligned at any scale in every browser
 * (SVG `foreignObject` is misplaced by WebKit when the SVG is scaled).
 */
export function Diagram({
  spec,
  ariaLabel,
  className,
  onNodeClick,
}: {
  spec: DiagramSpec
  ariaLabel: string
  className?: string
  /** Makes nodes clickable, e.g. to choose which node a figure is about. */
  onNodeClick?: (id: string) => void
}) {
  const { resolved: mode } = useTheme()
  const u = spec.unit ?? 40
  const layout = useMemo(() => {
    const byId = new Map(spec.nodes.map((n) => [n.id, n]))
    const groups = (spec.groups ?? []).map((g) => ({ g, box: groupBox(g, byId) }))
    const edges = (spec.edges ?? []).map((e) => ({ e, pts: routeEdge(e, byId) }))
    const all: Box[] = [
      ...spec.nodes.map(extent),
      ...groups.map((g) => g.box),
      ...edges.flatMap(({ pts }) => pts.map((p) => ({ x0: p.x, y0: p.y, x1: p.x, y1: p.y }))),
    ]
    const margin = 0.3
    const view = {
      x0: Math.min(...all.map((b) => b.x0)) - margin,
      y0: Math.min(...all.map((b) => b.y0)) - margin,
      x1: Math.max(...all.map((b) => b.x1)) + margin,
      y1: Math.max(...all.map((b) => b.y1)) + margin,
    }
    return { byId, groups, edges, view }
  }, [spec])
  const c = chrome(mode)
  const { view } = layout
  const width = (view.x1 - view.x0) * u
  const height = (view.y1 - view.y0) * u
  const scaled = (b: Box): Box => ({ x0: b.x0 * u, y0: b.y0 * u, x1: b.x1 * u, y1: b.y1 * u })
  const overlays: Overlay[] = []
  const vx = view.x0 * u
  const vy = view.y0 * u
  const pct = (v: number) => `${v * 100}%`

  const svg = (
    <svg
      viewBox={`${view.x0 * u} ${view.y0 * u} ${width} ${height}`}
      role="img"
      aria-label={ariaLabel}
      style={{ width: '100%', height: 'auto', display: 'block' }}
    >
      {layout.groups.map(({ g, box }) => {
        const colour = toneColour(mode, g.tone ?? 'neutral')
        const b = scaled(box)
        const at = g.labelAt ?? 'top-left'
        const right = at.endsWith('right')
        // At least 6 units wide, so a label longer than a narrow group still shows in full.
        const span = Math.max(b.x1 - b.x0 - 16, 6 * u)
        const lx = right ? b.x1 - 8 - span : b.x0 + 8
        const labelBox: Box = at.startsWith('top')
          ? { x0: lx, y0: b.y0 + 2, x1: lx + span, y1: b.y0 + LABEL_H * u }
          : { x0: lx, y0: b.y1 - LABEL_H * u, x1: lx + span, y1: b.y1 - 2 }
        if (g.label)
          overlays.push({
            key: `g-${g.id}`,
            box: labelBox,
            size: 11,
            align: right ? 'end' : 'start',
            content: (
              <div
                className="font-sans font-medium tracking-wide whitespace-nowrap uppercase"
                style={{ color: colour }}
              >
                <MathText text={g.label} />
              </div>
            ),
          })
        return (
          <g key={g.id}>
            <rect
              x={b.x0}
              y={b.y0}
              width={b.x1 - b.x0}
              height={b.y1 - b.y0}
              rx={10}
              fill={colour}
              fillOpacity={0.06}
              stroke={colour}
              strokeOpacity={0.55}
              strokeDasharray={g.dashed ? '5 4' : undefined}
            />
          </g>
        )
      })}

      {layout.edges.map(({ e, pts }, i) => {
        const colour = e.highlight
          ? seriesColor(mode, ACCENT_SLOT)
          : e.tone === undefined
            ? c.inkSecondary
            : toneColour(mode, e.tone)
        let px = pts.map((p) => ({ x: p.x * u, y: p.y * u }))
        let control: Pt | undefined
        if (e.route === 'curve' && px.length === 2) {
          // A quadratic curve bowed to the left of the direction of travel; the ends re-aim at the control point.
          const [p, q] = px
          const bend = (e.bend ?? 0.6) * u
          const len = Math.hypot(q.x - p.x, q.y - p.y) || 1
          control = {
            x: (p.x + q.x) / 2 + (-(q.y - p.y) / len) * bend,
            y: (p.y + q.y) / 2 + ((q.x - p.x) / len) * bend,
          }
          const a = layout.byId.get(parseEnd(e.from).id)!
          const b = layout.byId.get(parseEnd(e.to).id)!
          const ctrl = { x: control.x / u, y: control.y / u }
          const s0 = boundary(a, ctrl)
          const s1 = boundary(b, ctrl)
          px = [
            { x: s0.x * u, y: s0.y * u },
            { x: s1.x * u, y: s1.y * u },
          ]
        }
        const before = (k: 'start' | 'end') => control ?? (k === 'end' ? px[px.length - 2] : px[1])
        const arrow = e.arrow ?? 'end'
        // Pull the line back from the tip so the stroke does not poke through the arrowhead.
        const head = 7
        const trimmed = px.map((p) => ({ ...p }))
        const shorten = (tip: Pt, from: Pt) => {
          const len = Math.hypot(tip.x - from.x, tip.y - from.y) || 1
          tip.x -= ((tip.x - from.x) / len) * head * 0.8
          tip.y -= ((tip.y - from.y) / len) * head * 0.8
        }
        if (arrow === 'end' || arrow === 'both') shorten(trimmed[trimmed.length - 1], before('end'))
        if (arrow === 'start' || arrow === 'both') shorten(trimmed[0], before('start'))
        let labelAt: (Box & { start?: boolean }) | undefined
        if (e.label) {
          let best = 0
          for (let k = 1; k < px.length; k++) {
            const len = Math.hypot(px[k].x - px[k - 1].x, px[k].y - px[k - 1].y)
            if (len > best) {
              best = len
              const mx = (px[k].x + px[k - 1].x) / 2
              const my = (px[k].y + px[k - 1].y) / 2
              // Beside the segment, not on it: above a horizontal run, to the right of a vertical one.
              labelAt =
                Math.abs(px[k].y - px[k - 1].y) < 1
                  ? { x0: mx - 3 * u, y0: my - 0.62 * u, x1: mx + 3 * u, y1: my - 0.04 * u }
                  : { x0: mx + 0.12 * u, y0: my - 0.3 * u, x1: mx + 6 * u, y1: my + 0.3 * u, start: true }
            }
          }
        }
        if (e.label && labelAt)
          overlays.push({
            key: `e-${i}`,
            box: labelAt,
            size: 11,
            align: labelAt.start ? 'start' : 'center',
            content: (
              <span className="rounded px-1" style={{ background: c.surface, color: c.inkSecondary }}>
                <MathText text={e.label} />
              </span>
            ),
          })
        return (
          <g key={i}>
            <path
              d={
                control
                  ? `M ${trimmed[0].x} ${trimmed[0].y} Q ${control.x} ${control.y} ${trimmed[1].x} ${trimmed[1].y}`
                  : roundedPath(trimmed, 8)
              }
              fill="none"
              stroke={colour}
              strokeWidth={e.highlight ? 2.5 : 1.5}
              strokeDasharray={e.dashed ? '5 4' : undefined}
            />
            {(arrow === 'end' || arrow === 'both') && (
              <polygon points={arrowHead(px[px.length - 1], before('end'), head)} fill={colour} />
            )}
            {(arrow === 'start' || arrow === 'both') && (
              <polygon points={arrowHead(px[0], before('start'), head)} fill={colour} />
            )}
          </g>
        )
      })}

      {spec.nodes.map((n) => {
        const shape = n.shape ?? 'box'
        const colour = n.highlight
          ? seriesColor(mode, ACCENT_SLOT)
          : toneColour(mode, n.tone ?? (shape === 'op' || shape === 'text' || shape === 'factor' ? 'ink' : 'neutral'))
        const b = scaled(extent(n))
        const w = b.x1 - b.x0
        const h = b.y1 - b.y0
        const dash = n.dashed || shape === 'noise' ? '4 3' : undefined
        const tint = {
          fill: colour,
          fillOpacity: n.filled ? 0.35 : 0.14,
          stroke: colour,
          strokeWidth: n.highlight ? 2.5 : 1.5,
          strokeDasharray: dash,
        }
        const labelColour = n.highlight || (shape === 'text' && n.tone !== undefined) ? colour : c.ink
        const side = n.labelSide ?? (shape === 'factor' || shape === 'dot' ? 'n' : undefined)
        // A label beside the shape gets a box the size of a text node on that side.
        const outside = (s: Side): Box => {
          const lw = 1.8 * u
          const lh = 0.5 * u
          const gap = 4
          if (s === 'n') return { x0: n.x * u - lw / 2, y0: b.y0 - gap - lh, x1: n.x * u + lw / 2, y1: b.y0 - gap }
          if (s === 's') return { x0: n.x * u - lw / 2, y0: b.y1 + gap, x1: n.x * u + lw / 2, y1: b.y1 + gap + lh }
          if (s === 'e') return { x0: b.x1 + gap, y0: n.y * u - lh / 2, x1: b.x1 + gap + lw, y1: n.y * u + lh / 2 }
          return { x0: b.x0 - gap - lw, y0: n.y * u - lh / 2, x1: b.x0 - gap, y1: n.y * u + lh / 2 }
        }
        if (n.label)
          overlays.push({
            key: `n-${n.id}`,
            box: side ? outside(side) : b,
            size: n.small ? 11 : 13,
            align: side === 'e' ? 'start' : side === 'w' ? 'end' : 'center',
            content: labelLines(n.label, labelColour),
          })
        let body = null
        if (shape === 'factor') {
          body = <rect x={b.x0} y={b.y0} width={w} height={h} fill={colour} />
        } else if (shape === 'box' || shape === 'pill') {
          body = <rect x={b.x0} y={b.y0} width={w} height={h} rx={shape === 'pill' ? h / 2 : 6} {...tint} />
        } else if (shape === 'stack') {
          body = (
            <>
              <rect
                x={b.x0 + 8}
                y={b.y0 - 8}
                width={w}
                height={h}
                rx={6}
                {...tint}
                fillOpacity={0.06}
                strokeOpacity={0.4}
              />
              <rect
                x={b.x0 + 4}
                y={b.y0 - 4}
                width={w}
                height={h}
                rx={6}
                {...tint}
                fillOpacity={0.09}
                strokeOpacity={0.7}
              />
              <rect x={b.x0} y={b.y0} width={w} height={h} rx={6} {...tint} fill={c.surface} fillOpacity={1} />
              <rect x={b.x0} y={b.y0} width={w} height={h} rx={6} {...tint} />
            </>
          )
        } else if (shape === 'encoder' || shape === 'decoder') {
          body = <polygon points={trapezoid(n, u)} {...tint} strokeLinejoin="round" />
        } else if (ROUND.has(shape)) {
          const r = Math.min(w, h) / 2
          if (shape === 'dot') body = <circle cx={n.x * u} cy={n.y * u} r={r} fill={colour} />
          else if (shape === 'op')
            body = <circle cx={n.x * u} cy={n.y * u} r={r} fill={c.surface} stroke={colour} strokeWidth={1.5} />
          else
            body = (
              <>
                <circle cx={n.x * u} cy={n.y * u} r={r} {...tint} />
                {shape === 'latent' && (
                  <circle cx={n.x * u} cy={n.y * u} r={r - 4} fill="none" stroke={colour} strokeWidth={1} />
                )}
              </>
            )
        }
        return (
          <g
            key={n.id}
            onClick={onNodeClick ? () => onNodeClick(n.id) : undefined}
            style={onNodeClick ? { cursor: 'pointer' } : undefined}
          >
            {body}
          </g>
        )
      })}
    </svg>
  )

  return (
    <div
      className={className}
      style={{
        position: 'relative',
        width: '100%',
        maxWidth: width * (spec.maxScale ?? 1.4),
        margin: '0 auto',
        containerType: 'inline-size',
      }}
    >
      {svg}
      {overlays.map((o) => (
        <div
          key={o.key}
          className="font-prose"
          style={{
            position: 'absolute',
            left: pct((o.box.x0 - vx) / width),
            top: pct((o.box.y0 - vy) / height),
            width: pct((o.box.x1 - o.box.x0) / width),
            height: pct((o.box.y1 - o.box.y0) / height),
            // Container units: the font scales with the rendered width, exactly as the SVG does.
            fontSize: `${(o.size / width) * 100}cqw`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: o.align === 'start' ? 'flex-start' : o.align === 'end' ? 'flex-end' : 'center',
            textAlign: o.align === 'start' ? 'left' : o.align === 'end' ? 'right' : 'center',
            pointerEvents: 'none',
          }}
        >
          {o.content}
        </div>
      ))}
    </div>
  )
}
