/**
 * Closed paths for the epicycle drawings, computed in the browser: SVG elements are flattened to polylines, several
 * strokes are joined into one closed path, and the path is resampled uniformly by arc length.
 */
import { toFlat } from 'aifn-compute/foundation/tensor'
import { contourLines } from 'aifn-compute/numerics/geometry'

export type Pt = [number, number]
export type Stroke = { points: Pt[]; closed: boolean }
/** One SVG element: a tag (`path`, `circle`, `ellipse`, `line`, `rect`, `polyline`, `polygon`) and its attributes. */
export type SvgElement = [string, Record<string, string>]

const dist = (a: Pt, b: Pt) => Math.hypot(a[0] - b[0], a[1] - b[1])

// --------------------------------------------------------------------------------------------------------------------
// SVG flattening
// --------------------------------------------------------------------------------------------------------------------

const ARITY: Record<string, number> = { M: 2, L: 2, H: 1, V: 1, C: 6, S: 4, Q: 4, T: 2, A: 7, Z: 0 }
const NUMBER = /^[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?/

/** Commands with their argument groups. Arc flags may be written without separators (`a2 2 0 012 2`). */
function* tokens(d: string): Generator<[string, number[]]> {
  const parts = d.match(/[MmLlHhVvCcSsQqTtAaZz][^MmLlHhVvCcSsQqTtAaZz]*/g) ?? []
  for (const part of parts) {
    const cmd = part[0]
    const up = cmd.toUpperCase()
    const arity = ARITY[up]
    if (arity === 0) {
      yield [cmd, []]
      continue
    }
    const body = part.slice(1)
    const args: number[] = []
    let i = 0
    while (i < body.length) {
      if (' ,\t\n'.includes(body[i])) {
        i++
        continue
      }
      if (up === 'A' && (args.length % 7 === 3 || args.length % 7 === 4)) {
        args.push(Number(body[i++]))
        continue
      }
      const m = NUMBER.exec(body.slice(i))
      if (!m) throw new Error(`bad number in SVG path at '${body.slice(i)}'`)
      args.push(Number(m[0]))
      i += m[0].length
    }
    for (let k = 0; k < args.length; k += arity) {
      // Extra pairs after a moveto are implicit linetos.
      const c = k > 0 && up === 'M' ? (cmd === 'm' ? 'l' : 'L') : cmd
      yield [c, args.slice(k, k + arity)]
    }
  }
}

function cubic(p0: Pt, p1: Pt, p2: Pt, p3: Pt, k = 64): Pt[] {
  const out: Pt[] = []
  for (let i = 1; i <= k; i++) {
    const t = i / k
    const a = (1 - t) ** 3
    const b = 3 * (1 - t) ** 2 * t
    const c = 3 * (1 - t) * t ** 2
    const d = t ** 3
    out.push([a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0], a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1]])
  }
  return out
}

/** Endpoint-to-centre conversion of an SVG elliptical arc (SVG 1.1, appendix F.6.5), then sampled. */
function arc(p0: Pt, rx: number, ry: number, phi: number, large: boolean, sweep: boolean, p1: Pt): Pt[] {
  if (rx === 0 || ry === 0 || dist(p0, p1) < 1e-12) return [p1]
  rx = Math.abs(rx)
  ry = Math.abs(ry)
  const c = Math.cos((phi * Math.PI) / 180)
  const s = Math.sin((phi * Math.PI) / 180)
  const dx = (p0[0] - p1[0]) / 2
  const dy = (p0[1] - p1[1]) / 2
  const x1 = c * dx + s * dy
  const y1 = -s * dx + c * dy
  const lam = x1 ** 2 / rx ** 2 + y1 ** 2 / ry ** 2
  if (lam > 1) {
    rx *= Math.sqrt(lam)
    ry *= Math.sqrt(lam)
  }
  const num = rx ** 2 * ry ** 2 - rx ** 2 * y1 ** 2 - ry ** 2 * x1 ** 2
  const den = rx ** 2 * y1 ** 2 + ry ** 2 * x1 ** 2
  const coef = Math.sqrt(Math.max(num / den, 0)) * (large === sweep ? -1 : 1)
  const cx1 = (coef * rx * y1) / ry
  const cy1 = (-coef * ry * x1) / rx
  const cx = c * cx1 - s * cy1 + (p0[0] + p1[0]) / 2
  const cy = s * cx1 + c * cy1 + (p0[1] + p1[1]) / 2
  const a0 = Math.atan2((y1 - cy1) / ry, (x1 - cx1) / rx)
  let da = Math.atan2((-y1 - cy1) / ry, (-x1 - cx1) / rx) - a0
  if (sweep && da < 0) da += 2 * Math.PI
  else if (!sweep && da > 0) da -= 2 * Math.PI
  const k = Math.max(8, Math.floor((Math.abs(da) / (2 * Math.PI)) * 256))
  const out: Pt[] = []
  for (let i = 1; i <= k; i++) {
    const a = a0 + (da * i) / k
    const ex = rx * Math.cos(a)
    const ey = ry * Math.sin(a)
    out.push([c * ex - s * ey + cx, s * ex + c * ey + cy])
  }
  return out
}

/** Flatten an SVG path into its subpaths. */
export function flattenPath(d: string): Stroke[] {
  const strokes: Stroke[] = []
  let pts: Pt[] = []
  let cur: Pt = [0, 0]
  let start: Pt = [0, 0]
  let ctrl: Pt | null = null // the last control point, reflected by S and T
  let prev = ''
  const flush = (closed: boolean) => {
    if (pts.length > 1) strokes.push({ points: pts, closed })
  }
  for (const [cmd, a] of tokens(d)) {
    const up = cmd.toUpperCase()
    const ox = cmd === up ? 0 : cur[0]
    const oy = cmd === up ? 0 : cur[1]
    const at = (i: number): Pt => [ox + a[i], oy + a[i + 1]]
    const reflect = (kinds: string): Pt =>
      ctrl && kinds.includes(prev) ? [2 * cur[0] - ctrl[0], 2 * cur[1] - ctrl[1]] : cur
    if (up === 'M') {
      flush(false)
      cur = start = at(0)
      pts = [cur]
    } else if (up === 'Z') {
      flush(true)
      cur = start
      pts = [cur]
    } else if (up === 'L') {
      cur = at(0)
      pts.push(cur)
    } else if (up === 'H') {
      cur = [ox + a[0], cur[1]]
      pts.push(cur)
    } else if (up === 'V') {
      cur = [cur[0], oy + a[0]]
      pts.push(cur)
    } else if (up === 'C' || up === 'S') {
      const c1 = up === 'C' ? at(0) : reflect('CS')
      const c2 = up === 'C' ? at(2) : at(0)
      const end = up === 'C' ? at(4) : at(2)
      pts.push(...cubic(cur, c1, c2, end))
      ctrl = c2
      cur = end
    } else if (up === 'Q' || up === 'T') {
      const q = up === 'Q' ? at(0) : reflect('QT')
      const end = up === 'Q' ? at(2) : at(0)
      const lift = (p: Pt): Pt => [p[0] + (2 / 3) * (q[0] - p[0]), p[1] + (2 / 3) * (q[1] - p[1])]
      pts.push(...cubic(cur, lift(cur), lift(end), end))
      ctrl = q
      cur = end
    } else if (up === 'A') {
      const end = at(5)
      pts.push(...arc(cur, a[0], a[1], a[2], a[3] !== 0, a[4] !== 0, end))
      cur = end
    }
    prev = up
  }
  flush(false)
  return strokes
}

function ellipse(cx: number, cy: number, rx: number, ry: number): Pt[] {
  return Array.from({ length: 256 }, (_, i) => {
    const a = (2 * Math.PI * i) / 256
    return [cx + rx * Math.cos(a), cy + ry * Math.sin(a)] as Pt
  })
}

/** Flatten SVG elements into strokes. */
export function flattenSvg(elements: SvgElement[]): Stroke[] {
  return elements.flatMap(([tag, attrs]): Stroke[] => {
    const f = (k: string, fallback = 0) => (k in attrs ? Number(attrs[k]) : fallback)
    switch (tag) {
      case 'path':
        return flattenPath(attrs.d)
      case 'circle':
        return [{ points: ellipse(f('cx'), f('cy'), f('r'), f('r')), closed: true }]
      case 'ellipse':
        return [{ points: ellipse(f('cx'), f('cy'), f('rx'), f('ry')), closed: true }]
      case 'line':
        return [
          {
            points: [
              [f('x1'), f('y1')],
              [f('x2'), f('y2')],
            ],
            closed: false,
          },
        ]
      case 'rect': {
        const [x, y, w, h] = [f('x'), f('y'), f('width'), f('height')]
        const r = f('rx')
        return flattenPath(
          `M${x + r} ${y}H${x + w - r}A${r} ${r} 0 0 1 ${x + w} ${y + r}V${y + h - r}` +
            `A${r} ${r} 0 0 1 ${x + w - r} ${y + h}H${x + r}A${r} ${r} 0 0 1 ${x} ${y + h - r}V${y + r}` +
            `A${r} ${r} 0 0 1 ${x + r} ${y}Z`,
        )
      }
      case 'polyline':
      case 'polygon': {
        const v = attrs.points
          .split(/[\s,]+/)
          .filter(Boolean)
          .map(Number)
        const points = Array.from({ length: v.length / 2 }, (_, i) => [v[2 * i], v[2 * i + 1]] as Pt)
        return [{ points, closed: tag === 'polygon' }]
      }
      default:
        throw new Error(`unsupported SVG element '${tag}'`)
    }
  })
}

// --------------------------------------------------------------------------------------------------------------------
// One closed path
// --------------------------------------------------------------------------------------------------------------------

const pathLength = (p: Pt[]) => p.reduce((s, q, i) => (i ? s + dist(p[i - 1], q) : 0), 0)

/** A closed loop through a stroke: a closed contour as it is, an open stroke traced out and back. */
const loop = ({ points, closed }: Stroke): Pt[] => (closed ? points : [...points, ...points.slice(1, -1).reverse()])

/**
 * Join strokes into one closed path. The longest loop is the trunk. Each remaining loop is attached, nearest first, by
 * a bridge from the closest point of the path so far to the closest point of the loop; the path runs once round the
 * loop and returns along the same bridge, so each join adds only a doubled straight segment to the drawing.
 */
export function stitch(strokes: Stroke[]): Pt[] {
  const loops = strokes.filter((s) => s.points.length > 0).map(loop)
  if (!loops.length) return []
  const lengths = loops.map(pathLength)
  let path = loops.splice(lengths.indexOf(Math.max(...lengths)), 1)[0]
  while (loops.length) {
    let best = { d: Infinity, i: 0, j: 0, k: 0 }
    loops.forEach((l, k) => {
      for (let i = 0; i < path.length; i++)
        for (let j = 0; j < l.length; j++) {
          const d = (path[i][0] - l[j][0]) ** 2 + (path[i][1] - l[j][1]) ** 2
          if (d < best.d) best = { d, i, j, k }
        }
    })
    const l = loops.splice(best.k, 1)[0]
    const tour = [...l.slice(best.j), ...l.slice(0, best.j), l[best.j]]
    path = [...path.slice(0, best.i + 1), ...tour, ...path.slice(best.i)]
  }
  return path
}

/** `n` points spaced uniformly by arc length along the closed path through `points`. */
export function resample(points: Pt[], n: number): Pt[] {
  const p = points.filter((q, i) => i === 0 || dist(points[i - 1], q) > 1e-12)
  if (p.length < 2) return Array.from({ length: n }, () => [...(p[0] ?? [0, 0])] as Pt)
  const ring = [...p, p[0]]
  const s = [0]
  for (let i = 1; i < ring.length; i++) s.push(s[i - 1] + dist(ring[i - 1], ring[i]))
  const total = s[s.length - 1]
  const out: Pt[] = []
  let j = 0
  for (let i = 0; i < n; i++) {
    const t = (total * i) / n
    while (s[j + 1] < t) j++
    const u = s[j + 1] > s[j] ? (t - s[j]) / (s[j + 1] - s[j]) : 0
    out.push([ring[j][0] + u * (ring[j + 1][0] - ring[j][0]), ring[j][1] + u * (ring[j + 1][1] - ring[j][1])])
  }
  return out
}

/** Centre on the centroid, scale to unit maximum radius, and flip y when the source has y pointing down. */
export function normalise(points: Pt[], flipY: boolean): Pt[] {
  const sy = flipY ? -1 : 1
  const mx = points.reduce((s, p) => s + p[0], 0) / points.length
  const my = points.reduce((s, p) => s + sy * p[1], 0) / points.length
  const centred = points.map(([x, y]) => [x - mx, sy * y - my] as Pt)
  const r = Math.max(...centred.map(([x, y]) => Math.hypot(x, y))) || 1
  return centred.map(([x, y]) => [x / r, y / r])
}

/** A path from SVG elements (y down), as `n` normalised points. */
export const svgPath = (elements: SvgElement[], n: number): Pt[] =>
  normalise(resample(stitch(flattenSvg(elements)), n), true)

/** A freehand stroke, closed by the straight segment from its end back to its start, as `n` normalised points. */
export const freehandPath = (points: Pt[], n: number, flipY = false): Pt[] => normalise(resample(points, n), flipY)

// --------------------------------------------------------------------------------------------------------------------
// Glyphs
// --------------------------------------------------------------------------------------------------------------------

const GLYPH_GRID = 512

/**
 * The outline of one character in a CSS font family and style (`''`, `'italic'`, `'bold'`), as `n` normalised points. The character is drawn on an offscreen canvas;
 * marching squares trace the half-coverage level of its anti-aliased mask, and the outer contours and counters are
 * stitched into one path. The font must already be loaded (see `loadFont`), or the browser substitutes a fallback.
 */
export function glyphPath(char: string, family: string, n: number, style = ''): Pt[] {
  const size = GLYPH_GRID
  const canvas = document.createElement('canvas')
  canvas.width = canvas.height = size
  const ctx = canvas.getContext('2d', { willReadFrequently: true })
  if (!ctx) return []
  ctx.font = `${style} ${Math.round(size * 0.62)}px ${family}`
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillText(char, size / 2, size / 2)
  const alpha = ctx.getImageData(0, 0, size, size).data
  // A one-pixel empty border keeps every contour closed.
  const z = Array.from({ length: size }, (_, r) =>
    Array.from({ length: size }, (_, c) =>
      r === 0 || c === 0 || r === size - 1 || c === size - 1 ? 0 : alpha[4 * (r * size + c) + 3] / 255,
    ),
  )
  const axis = Array.from({ length: size }, (_, i) => i)
  const strokes: Stroke[] = contourLines(axis, axis, z, 0.5).map((line) => {
    const v = toFlat(line)
    const points = Array.from({ length: v.length / 2 - 1 }, (_, i) => [v[2 * i], v[2 * i + 1]] as Pt)
    return { points, closed: true }
  })
  const path = stitch(strokes.filter((s) => s.points.length > 2))
  return path.length ? normalise(resample(path, n), true) : []
}

/** Ask the browser to load a font family in a style (`''`, `'italic'`, `'bold'`) before drawing glyphs with it. */
export const loadFont = (family: string, style = ''): Promise<unknown> =>
  typeof document !== 'undefined' && document.fonts ? document.fonts.load(`${style} 32px ${family}`) : Promise.resolve()
