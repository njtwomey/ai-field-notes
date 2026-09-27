import { useMemo } from 'react'
import { useTheme } from '@/components/theme-provider'
import { EChart, type EChartClick } from './EChart'
import { chrome, seriesColor } from './palette'

/**
 * A node of a small hand-placed graph. `x` and `y` are layout units (y grows downwards); the diagram scales them to
 * fit and keeps their aspect ratio. Random variables are circles, filled when `observed`; factors are small squares;
 * clusters (junction-tree cliques) are rounded boxes sized to their label; `text` draws the label alone (e.g. "⋯").
 *
 * Labels take a small TeX-like markup: `_x` or `_{..}` for a subscript, `^x` or `^{..}` for a superscript and
 * `\mathbf{..}` for bold. Single letters are set in the italic maths face and words upright, as in LaTeX.
 */
export type GraphNode = {
  id: string
  label?: string
  x: number
  y: number
  kind?: 'variable' | 'observed' | 'factor' | 'cluster' | 'text'
  /** Where the label sits. Defaults to inside, or above for factors. */
  labelPosition?: 'inside' | 'top' | 'bottom' | 'left' | 'right'
}

/** `directed` defaults to true. `highlight` draws the edge in the accent colour; `dashed` marks an added edge. */
export type GraphEdge = {
  source: string
  target: string
  directed?: boolean
  label?: string
  dashed?: boolean
  highlight?: boolean
  /** Bend the edge, e.g. 0.3, so that it clears a node lying on the straight line. */
  curveness?: number
}

/** A plate (repeated block) drawn as a rectangle from (x0, y0) to (x1, y1), labelled in its bottom-right corner. */
export type GraphPlate = { label: string; x0: number; y0: number; x1: number; y1: number }

const MAIN = 'KaTeX_Main, serif'
const MATH = 'KaTeX_Math, KaTeX_Main, serif'
const BASE_SIZE = 15
const SCRIPT_SIZE = 10.5

type Style = 'up' | 'it' | 'bf' | 'bi'
type Level = 'n' | 'b' | 'p'
type Run = { text: string; level: Level; bold: boolean }

const LATIN = /[A-Za-z]/
const GREEK_LOWER = /[α-ω]/
const GREEK_UPPER = /[Α-Ω]/

/** Split label markup into runs of plain, subscript (b) and superscript (p) text, with a bold flag. */
function parse(label: string): Run[] {
  const runs: Run[] = []
  let i = 0
  const group = (bold: boolean, level: Level) => {
    // Read one char or a {...} group after _ ^ or \mathbf.
    if (label[i] === '{') {
      const end = label.indexOf('}', i)
      const inner = label.slice(i + 1, end < 0 ? undefined : end)
      i = end < 0 ? label.length : end + 1
      for (const r of parse(inner)) runs.push({ ...r, level: r.level === 'n' ? level : r.level, bold: r.bold || bold })
    } else if (i < label.length) {
      runs.push({ text: label[i], level, bold })
      i++
    }
  }
  while (i < label.length) {
    const ch = label[i]
    if (ch === '_' || ch === '^') {
      i++
      group(false, ch === '_' ? 'b' : 'p')
    } else if (label.startsWith('\\mathbf', i)) {
      i += '\\mathbf'.length
      group(true, 'n')
    } else {
      runs.push({ text: ch, level: 'n', bold: false })
      i++
    }
  }
  return runs
}

/** Style of one character: single letters italic, words (2+ Latin letters at base level) upright, like LaTeX. */
function styleOf(ch: string, bold: boolean, inWord: boolean): Style {
  if (bold) return GREEK_LOWER.test(ch) ? 'bi' : 'bf'
  if (inWord || GREEK_UPPER.test(ch)) return 'up'
  return LATIN.test(ch) || GREEK_LOWER.test(ch) ? 'it' : 'up'
}

/** ECharts rich-text markup for a label, and its approximate width in pixels. */
function richLabel(label: string): { text: string; width: number } {
  const runs = parse(label)
  // Consecutive characters in the same level and face become one token, so the browser kerns them together.
  const tokens: { tag: string; text: string }[] = []
  let width = 0
  for (let k = 0; k < runs.length; k++) {
    const r = runs[k]
    const neighbour = (d: number) => runs[k + d]?.level === 'n' && LATIN.test(runs[k + d]?.text ?? '')
    const inWord = r.level === 'n' && LATIN.test(r.text) && (neighbour(-1) || neighbour(1))
    const tag = `${r.level}${styleOf(r.text, r.bold, inWord)}`
    // A hyphen in an index (n-1) is a minus sign, as in TeX.
    const ch = (r.level !== 'n' && r.text === '-' ? '\u2212' : r.text).replace(/[{}|]/g, '')
    const last = tokens.at(-1)
    if (last?.tag === tag) last.text += ch
    else tokens.push({ tag, text: ch })
    width += (r.level === 'n' ? BASE_SIZE : SCRIPT_SIZE) * (r.text === ' ' ? 0.3 : 0.55)
  }
  return { text: tokens.map((t) => `{${t.tag}|${t.text}}`).join(''), width }
}

/** One rich style per level and face. Sub- and superscripts sit at the bottom or top of the line box. */
function richStyles(color: string) {
  const faces: Record<Style, object> = {
    up: { fontFamily: MAIN },
    it: { fontFamily: MATH, fontStyle: 'italic' },
    bf: { fontFamily: MAIN, fontWeight: 'bold' },
    bi: { fontFamily: MATH, fontStyle: 'italic', fontWeight: 'bold' },
  }
  const levels: Record<Level, object> = {
    n: { fontSize: BASE_SIZE, lineHeight: 20 },
    b: { fontSize: SCRIPT_SIZE, lineHeight: 10, verticalAlign: 'bottom' },
    p: { fontSize: SCRIPT_SIZE, lineHeight: 10, verticalAlign: 'top' },
  }
  const out: Record<string, object> = {}
  for (const [l, ls] of Object.entries(levels))
    for (const [s, fs] of Object.entries(faces)) out[`${l}${s}`] = { color, ...ls, ...fs }
  return out
}

const plateId = (i: number, corner: string) => `__plate${i}${corner}`

/**
 * A fixed-layout diagram of a graphical model, factor graph, junction tree or computation graph. Nodes in `highlight`
 * take the first palette colour, which is how a figure marks the variables a sentence is about (a Markov blanket, a
 * query). `onNodeClick` makes nodes clickable, e.g. to choose which node a figure is about.
 */
export function GraphDiagram({
  nodes,
  edges,
  plates = [],
  highlight = [],
  height = 220,
  ariaLabel,
  onNodeClick,
}: {
  nodes: GraphNode[]
  edges: GraphEdge[]
  plates?: GraphPlate[]
  highlight?: string[]
  height?: number
  ariaLabel?: string
  onNodeClick?: (id: string) => void
}) {
  const { resolved: mode } = useTheme()
  const clickable = onNodeClick !== undefined
  // Value dependencies: callers usually pass inline arrays, so key the memo on their contents.
  const key = JSON.stringify([nodes, edges, plates, highlight])
  const option = useMemo(() => {
    const [n, e, p, h] = JSON.parse(key) as [GraphNode[], GraphEdge[], GraphPlate[], string[]]
    const c = chrome(mode)
    const accent = seriesColor(mode, 0)
    const marked = new Set(h)
    const data: object[] = n.map((node) => {
      const kind = node.kind ?? 'variable'
      const colour = marked.has(node.id) ? accent : c.ink
      const { text, width } = richLabel(node.label ?? node.id)
      const size =
        kind === 'factor' ? 12 : kind === 'cluster' ? [width + 22, 30] : kind === 'text' ? 0 : Math.max(34, width + 14)
      const position = node.labelPosition ?? (kind === 'factor' ? 'top' : 'inside')
      return {
        id: node.id,
        name: node.id,
        x: node.x,
        y: node.y,
        symbol: kind === 'factor' ? 'rect' : kind === 'cluster' ? 'roundRect' : 'circle',
        symbolSize: size,
        itemStyle: {
          color: kind === 'factor' ? colour : kind === 'observed' ? c.grid : c.surface,
          borderColor: colour,
          borderWidth: kind === 'text' ? 0 : marked.has(node.id) ? 2.5 : 1.5,
        },
        label: {
          show: node.label !== '' && !(kind === 'factor' && node.label === undefined),
          position,
          distance: 4,
          formatter: text,
          rich: richStyles(colour),
        },
      }
    })
    const links: object[] = e.map((edge) => ({
      source: edge.source,
      target: edge.target,
      symbol: edge.directed === false ? ['none', 'none'] : ['none', 'arrow'],
      lineStyle: {
        color: edge.highlight ? accent : c.inkSecondary,
        type: edge.dashed ? 'dashed' : 'solid',
        width: edge.highlight ? 2 : 1.5,
        curveness: edge.curveness ?? 0,
      },
      label: edge.label
        ? {
            show: true,
            formatter: richLabel(edge.label).text,
            rich: richStyles(edge.highlight ? accent : c.inkSecondary),
            backgroundColor: c.surface,
            padding: [0, 3],
          }
        : { show: false },
    }))
    // ECharts pads a zero extent by ±1 unit, which squeezes a single row (or column) into a narrow box. Invisible
    // anchors a quarter unit either side give flat layouts a sensible scale instead.
    const xs = n.map((node) => node.x)
    const ys = n.map((node) => node.y)
    const flat = (v: number[]) => v.length > 0 && Math.max(...v) === Math.min(...v)
    const anchors: [number, number][] = []
    if (flat(ys)) anchors.push([xs[0], ys[0] - 0.25], [xs[0], ys[0] + 0.25])
    if (flat(xs)) anchors.push([xs[0] - 0.25, ys[0]], [xs[0] + 0.25, ys[0]])
    anchors.forEach(([x, y], i) =>
      data.push({
        id: `__anchor${i}`,
        name: `__anchor${i}`,
        x,
        y,
        symbol: 'none',
        symbolSize: 0,
        label: { show: false },
      }),
    )
    // A plate is four invisible corner nodes joined by straight edges, so it shares the nodes' coordinate system.
    p.forEach((plate, i) => {
      const corners = {
        a: [plate.x0, plate.y0],
        b: [plate.x1, plate.y0],
        c: [plate.x1, plate.y1],
        d: [plate.x0, plate.y1],
      }
      for (const [corner, [x, y]] of Object.entries(corners))
        data.push({
          id: plateId(i, corner),
          name: plateId(i, corner),
          x,
          y,
          symbol: 'none',
          symbolSize: 0,
          silent: true,
          label:
            corner === 'c'
              ? {
                  show: true,
                  position: [-6, -6],
                  align: 'right',
                  verticalAlign: 'bottom',
                  formatter: richLabel(plate.label).text,
                  rich: richStyles(c.inkSecondary),
                }
              : { show: false },
        })
      for (const [s, t] of ['ab', 'bc', 'cd', 'da'])
        links.push({
          source: plateId(i, s),
          target: plateId(i, t),
          symbol: ['none', 'none'],
          lineStyle: { color: c.muted, width: 1.2, type: 'solid' },
        })
    })
    return {
      tooltip: { show: false },
      series: [
        {
          type: 'graph',
          layout: 'none',
          // Keep the layout's shape; without this ECharts stretches it (and every circle) to fill the box.
          preserveAspect: true,
          silent: !clickable,
          cursor: clickable ? 'pointer' : 'default',
          animation: false,
          left: 28,
          right: 28,
          top: 26,
          bottom: 26,
          emphasis: { disabled: true },
          label: { show: true, color: c.ink },
          lineStyle: { color: c.inkSecondary, width: 1.5, opacity: 1 },
          edgeSymbolSize: 8,
          data,
          links,
        },
      ],
    }
  }, [key, mode, clickable])
  const onClick = useMemo(
    () =>
      onNodeClick &&
      ((event: EChartClick) => {
        const id = (event.data as { id?: string } | undefined)?.id
        if (event.dataType === 'node' && id && !id.startsWith('__')) onNodeClick(id)
      }),
    [onNodeClick],
  )
  return <EChart option={option} height={height} cartesian={false} ariaLabel={ariaLabel} onClick={onClick} />
}
