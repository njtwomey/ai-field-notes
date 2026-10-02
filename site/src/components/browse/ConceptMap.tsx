import { useMemo } from 'react'
import { useNavigate } from 'react-router'
import { useTheme } from '@/components/theme-provider'
import { EChart, type EChartClick, chrome, MARKER_SHAPES, seriesColor } from 'aifn-render'
import { kindLabels, notes, noteUrl, taxonomy, topicOf, type NoteMeta } from '@/lib/content'

type Edge = { source: string; target: string; kind: 'requires' | 'part of' | 'related' }

function edges(): Edge[] {
  const seen = new Set<string>()
  const out: Edge[] = []
  const add = (source: string, target: string, kind: Edge['kind']) => {
    // Related links are symmetric; draw each pair once.
    const key = kind === 'related' ? [source, target].sort().join('|') : `${source}>${target}`
    if (seen.has(key)) return
    seen.add(key)
    out.push({ source, target, kind })
  }
  for (const n of notes) {
    n.requires.forEach((t) => add(n.slug, t, 'requires'))
    n.partOf.forEach((t) => add(n.slug, t, 'part of'))
    ;[...n.related, ...n.linked].forEach((t) => add(n.slug, t, 'related'))
  }
  return out
}

/**
 * Every note as a node, coloured and shaped by topic, linked by its relations. Notes outside `highlight` are dimmed.
 * Clicking a node opens the note.
 */
export function ConceptMap({ highlight, height = 560 }: { highlight: Set<string>; height?: number }) {
  const { resolved: mode } = useTheme()
  const navigate = useNavigate()

  const option = useMemo(() => {
    const c = chrome(mode)
    const links = edges()
    const degree = new Map<string, number>()
    for (const e of links) {
      degree.set(e.source, (degree.get(e.source) ?? 0) + 1)
      degree.set(e.target, (degree.get(e.target) ?? 0) + 1)
    }
    const topicIndex = new Map(taxonomy.map((t, i) => [t.path, i]))
    const dim = (slug: string) => highlight.size > 0 && !highlight.has(slug)
    return {
      tooltip: {
        formatter: (p: {
          dataType: string
          data: { note?: NoteMeta; source?: string; target?: string; kind?: string }
        }) =>
          p.dataType === 'node' && p.data.note
            ? `<b>${p.data.note.title}</b><br/><span style="opacity:.7">${kindLabels[p.data.note.kind]} · ${
                topicOf(p.data.note).title
              }</span>`
            : `${p.data.source} ${p.data.kind} ${p.data.target}`,
      },
      legend: { data: taxonomy.map((t) => t.title), bottom: 0, top: 'auto', left: 'center' },
      series: [
        {
          type: 'graph',
          layout: 'force',
          roam: true,
          draggable: true,
          force: { repulsion: 900, edgeLength: [110, 200], gravity: 0.04, friction: 0.2 },
          labelLayout: { hideOverlap: true },
          categories: taxonomy.map((t, i) => ({
            name: t.title,
            symbol: MARKER_SHAPES[i % MARKER_SHAPES.length],
            itemStyle: { color: seriesColor(mode, i % 8) },
          })),
          label: { show: true, position: 'right', fontSize: 12, color: c.ink },
          emphasis: { focus: 'adjacency', lineStyle: { width: 2 } },
          lineStyle: { color: c.muted, opacity: 0.6, curveness: 0.1 },
          edgeSymbol: ['none', 'arrow'],
          edgeSymbolSize: 7,
          data: notes.map((n) => ({
            id: n.slug,
            name: n.title,
            note: n,
            category: topicIndex.get(topicOf(n).path),
            symbolSize: 12 + 3 * Math.min(degree.get(n.slug) ?? 0, 5),
            itemStyle: { opacity: dim(n.slug) ? 0.15 : 1, borderColor: c.surface, borderWidth: 1 },
            label: { opacity: dim(n.slug) ? 0.25 : 1 },
          })),
          links: links.map((e) => ({
            source: e.source,
            target: e.target,
            kind: e.kind,
            symbol: e.kind === 'related' ? ['none', 'none'] : ['none', 'arrow'],
            lineStyle: {
              type: e.kind === 'requires' ? 'solid' : e.kind === 'part of' ? 'dashed' : 'dotted',
              opacity: dim(e.source) || dim(e.target) ? 0.1 : 0.6,
            },
          })),
        },
      ],
    }
  }, [mode, highlight])

  const onClick = (e: EChartClick) => {
    if (e.dataType === 'node' && e.data && typeof e.data === 'object' && 'id' in e.data) {
      navigate(noteUrl(String((e.data as { id: string }).id)))
    }
  }

  return (
    <EChart
      option={option}
      height={height}
      cartesian={false}
      onClick={onClick}
      ariaLabel="Map of all notes, linked by their relations"
    />
  )
}
