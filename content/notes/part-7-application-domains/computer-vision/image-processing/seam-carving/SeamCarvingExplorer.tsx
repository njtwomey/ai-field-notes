import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import {
  choice,
  Figure,
  float,
  formatNumber,
  Curve,
  Handle,
  Pixels,
  Plot,
  type PlotPointer,
  Readout,
  setting,
  useAxis,
  useFigureState,
} from 'aifn-render'
import {
  H,
  MAX_INSERT,
  MAX_REMOVE,
  W,
  boxMask,
  carve,
  scaled,
  seamThrough,
  testImage,
  viewAt,
  type Box,
  type BoxMode,
  type Energy,
  type ImageId,
  type Sequence,
  type View,
} from './seam'

const EXTENT = W + MAX_INSERT
const ROWS = Array.from({ length: H }, (_, r) => r)
const DEFAULT_BOX: Box = { r0: 38, r1: 72, c0: 49, c1: 71 }

type Settings = { image: ImageId; energy: Energy; mode: BoxMode; box: Box }
const keyOf = (s: Settings) =>
  `${s.image}|${s.energy}|${s.mode}|${s.mode === 'off' ? '' : [s.box.r0, s.box.r1, s.box.c0, s.box.c1].join(',')}`

// Removal sequences by settings; each takes tens of milliseconds, so revisiting a setting is free.
const cache = new Map<string, Sequence>()
function sequenceFor(s: Settings): Sequence {
  const key = keyOf(s)
  let seq = cache.get(key)
  if (!seq) {
    if (cache.size > 24) cache.delete(cache.keys().next().value!)
    seq = carve(testImage(s.image), boxMask(s.box, s.mode), s.energy)
    cache.set(key, seq)
  }
  return seq
}

/** The column in the current image of the first pixel that came from original column `c` or later, in row `r`. */
function currentColumn(v: View, r: number, c: number): number {
  const row = v.src.subarray(r * v.width, (r + 1) * v.width)
  let x = 0
  while (x < v.width && row[x] < c) x++
  return x
}
/** The original column of the pixel at current column `x` in row `r`. */
const originalColumn = (v: View, r: number, x: number) =>
  v.src[r * v.width + Math.min(Math.max(Math.round(x), 0), v.width - 1)]
const clamp = (v: number, lo: number, hi: number) => Math.min(Math.max(v, lo), hi)

const seamLine = (name: string, seam: ArrayLike<number>, extra: Partial<ImageLine> = {}): ImageLine => ({
  name,
  x: Array.from(seam),
  y: ROWS,
  ...extra,
})

/** A path over an image in pixel coordinates (x = column, y = row), e.g. a seam. */
type ImageLine = { name: string; x: number[]; y: number[]; slot?: number; emphasis?: boolean; dashed?: boolean }

type ImageViewProps = {
  width: number
  rgb?: ArrayLike<number>
  values?: ArrayLike<number>
  scale?: 'sequential' | 'diverging'
  range?: [number, number]
  lines?: ImageLine[]
  handles?: Handle[]
  onPointer?: (e: PlotPointer) => void
  ariaLabel: string
}

/**
 * An image of H rows on pixel axes, row 0 at the top, with paths and handles over it. Every panel spans EXTENT
 * columns, so images of different widths share one pixel size.
 */
function ImageView({ width, rgb, values, scale, range, lines, handles, onPointer, ariaLabel }: ImageViewProps) {
  const x = useAxis({ range: [-0.5, EXTENT - 0.5], nice: false })
  const y = useAxis({ range: [-0.5, H - 0.5], nice: false, inverse: true, equal: x })
  return (
    <Plot x={x} y={y} bare fitHeight onPointer={onPointer} ariaLabel={ariaLabel}>
      <Pixels width={width} height={H} rgb={rgb} values={values} scale={scale} range={range} />
      {lines?.map((l) => (
        <Curve key={l.name} name={l.name} x={l.x} y={l.y} slot={l.slot} emphasis={l.emphasis} dashed={l.dashed} live />
      ))}
      {handles?.map((h, i) => (
        <Handle key={i} {...h} />
      ))}
    </Plot>
  )
}

function Panel({ title, children }: { title: ReactNode; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-xs text-muted-foreground">{title}</span>
      {children}
    </div>
  )
}

/** Seam carving on generated images, with the energy map and the cumulative cost M of the image being carved. */
export function SeamCarvingExplorer() {
  const state = useFigureState({
    width: float(90, {
      min: W - MAX_REMOVE,
      max: W + MAX_INSERT,
      step: 1,
      label: 'width',
      format: (v) =>
        v < W ? `${v} px (${W - v} removed)` : v > W ? `${v} px (${v - W} inserted)` : `${v} px (original)`,
    }),
    image: choice<ImageId>(
      [
        { value: 'landscape', label: 'landscape' },
        { value: 'shapes', label: 'shapes' },
        { value: 'checkerboard', label: 'checkerboard' },
        { value: 'text', label: 'text' },
        { value: 'line', label: 'straight line' },
      ],
      'landscape',
      { label: 'image' },
    ),
    energy: choice<Energy>(
      [
        { value: 'backward', label: 'backward' },
        { value: 'forward', label: 'forward' },
      ],
      'backward',
      { label: 'energy' },
    ),
    mode: choice<BoxMode>(
      [
        { value: 'off', label: 'off' },
        { value: 'protect', label: 'protect' },
        { value: 'remove', label: 'remove' },
      ],
      'off',
      { label: 'box' },
    ),
    compare: setting(true, 'compare with plain scaling'),
  })
  // The box as drawn while dragging, and the box the removal sequence uses, committed when the drag ends.
  const [box, setBox] = useState<Box>(DEFAULT_BOX)
  const [committed, setCommitted] = useState<Box>(DEFAULT_BOX)
  const draft = useRef(box)
  const [hover, setHover] = useState<{ r: number; c: number; panel: 'image' | 'energy' | 'cost' } | null>(null)

  const wanted: Settings = { image: state.image, energy: state.energy, mode: state.mode, box: committed }
  const key = keyOf(wanted)
  const [ready, setReady] = useState(() => ({ key, seq: sequenceFor(wanted) }))
  const busy = ready.key !== key
  useEffect(() => {
    if (!busy) return
    const settings: Settings = { image: state.image, energy: state.energy, mode: state.mode, box: committed }
    // Wait a frame so "recomputing…" can paint before the carving blocks the main thread.
    const id = setTimeout(() => setReady({ key, seq: sequenceFor(settings) }), 16)
    return () => clearTimeout(id)
  }, [busy, key, state.image, state.energy, state.mode, committed])

  const seq = ready.seq
  const w = state.width
  const view = useMemo(() => viewAt(seq, w), [seq, w])
  const plain = useMemo(() => (state.compare ? scaled(seq.rgb, w) : null), [state.compare, seq, w])

  const at = hover && hover.r < H && hover.c < view.width ? hover : null
  const [atR, atC] = at ? [at.r, at.c] : [-1, -1]
  const through = useMemo(() => (atR >= 0 ? seamThrough(view, atR, atC) : null), [view, atR, atC])

  // Where the box sits in the current image: measured along its middle row.
  const mid = Math.round((box.r0 + box.r1) / 2)
  const left = currentColumn(view, mid, box.c0) - 0.5
  const right = currentColumn(view, mid, box.c1 + 1) - 0.5
  const top = box.r0 - 0.5
  const bottom = box.r1 + 0.5

  const lines = useMemo(() => {
    const out: ImageLine[] = [seamLine('next seam', view.seam, { slot: 1 })]
    if (through) out.push(seamLine('seam through pixel', through.seam, { emphasis: true, dashed: true }))
    return out
  }, [view, through])
  const imageLines = useMemo(
    () =>
      state.mode === 'off'
        ? lines
        : [
            ...lines,
            {
              name: 'box',
              x: [left, right, right, left, left],
              y: [top, top, bottom, bottom, top],
              slot: state.mode === 'protect' ? 2 : 7,
            },
          ],
    [lines, state.mode, left, right, top, bottom],
  )

  const setDraft = (next: Box) => {
    draft.current = next
    setBox(next)
  }
  const commit = () => setCommitted(draft.current)

  const widthHandle: Handle = {
    kind: 'x',
    at: w - 0.5,
    label: 'width',
    onDrag: (x) => state.set('width', Math.round(x + 0.5)),
  }
  const handles: Handle[] =
    state.mode === 'off'
      ? [widthHandle]
      : [
          widthHandle,
          {
            kind: 'point',
            at: [(left + right) / 2, (top + bottom) / 2],
            onDrag: ([x, y]) => {
              const b = draft.current
              const hw = (b.c1 - b.c0) / 2
              const hh = (b.r1 - b.r0) / 2
              const r = clamp(Math.round(y), 0, H - 1)
              const c0 = clamp(Math.round(originalColumn(view, r, x) - hw), 0, W - 1 - (b.c1 - b.c0))
              const r0 = clamp(Math.round(y - hh), 0, H - 1 - (b.r1 - b.r0))
              setDraft({ r0, r1: r0 + b.r1 - b.r0, c0, c1: c0 + b.c1 - b.c0 })
            },
            onRelease: commit,
          },
          {
            kind: 'point',
            at: [right, bottom],
            onDrag: ([x, y]) => {
              const b = draft.current
              const r = clamp(Math.round(y), 0, H - 1)
              const c1 = clamp(originalColumn(view, r, x - 0.5), b.c0 + 3, W - 1)
              const r1 = clamp(Math.round(y - 0.5), b.r0 + 3, H - 1)
              setDraft({ ...b, r1, c1 })
            },
            onRelease: commit,
          },
        ]

  const pointer = (panel: 'image' | 'energy' | 'cost') => (e: PlotPointer) => {
    if (e.type === 'leave') return setHover(null)
    const c = Math.round(e.point[0])
    const r = Math.round(e.point[1])
    setHover(c >= 0 && c < view.width && r >= 0 && r < H ? { r, c, panel } : null)
  }

  const k = w - W
  const M = view.fields.M
  const parents =
    at && at.r > 0
      ? [-1, 0, 1].map((d) => {
          const c = at.c + d
          const chosen = view.fields.parent[at.r * view.width + at.c] === d
          const value = c >= 0 && c < view.width ? formatNumber(M[(at.r - 1) * view.width + c]) : '–'
          return (
            <span key={d} className={chosen ? 'font-semibold underline' : 'opacity-70'}>
              {value}
              {d < 1 ? ' · ' : ''}
            </span>
          )
        })
      : null

  return (
    <Figure
      title="Seam carving an image"
      state={state}
      caption={
        <>
          Drag the dashed line at the image's right edge, or use the slider, to set the width. Left of the original 128
          pixels removes seams; right of it inserts them. The orange path is the next seam to remove, drawn over the
          image and over the cumulative cost M. Plain scaling to the same width squashes every object; carving keeps the
          trees and house intact and removes sky and grass. With the box on, drag its centre to move it and its corner
          to resize it: protect adds energy so seams avoid it, remove subtracts energy so seams pass through it and
          delete the object. The checkerboard, text and line images have no empty region to remove, so their structure
          breaks and the line bends. Hover a pixel to see the cheapest seam through it and, on M, the three parents it
          chose between.
        </>
      }
      readouts={
        <>
          <Readout label="width" value={`${w} px`} />
          <Readout label={k < 0 ? 'seams removed' : 'seams inserted'} value={String(Math.abs(k))} />
          <Readout label="next seam's energy" value={formatNumber(view.cost)} />
          {at && <Readout label="pixel (row, column)" value={`(${at.r}, ${at.c})`} />}
          {at && <Readout label="M" value={formatNumber(M[at.r * view.width + at.c])} />}
          {at && at.panel === 'cost' && parents && <Readout label="parents M" value={parents} />}
          {through && <Readout label="cheapest seam through it" value={formatNumber(through.cost)} />}
          {busy && <span className="text-foreground">recomputing…</span>}
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Panel title="Seam-carved image">
          <ImageView
            width={view.width}
            rgb={view.rgb}
            lines={imageLines}
            handles={handles}
            onPointer={pointer('image')}
            ariaLabel="Seam-carved image with the next seam and a draggable right edge"
          />
        </Panel>
        {plain && (
          <Panel title="Plain scaling to the same width">
            <ImageView
              width={w}
              rgb={plain}
              handles={[widthHandle]}
              ariaLabel="The original image scaled to the same width"
            />
          </Panel>
        )}
        <Panel
          title={
            state.energy === 'backward' ? (
              'Energy e (gradient magnitude)'
            ) : (
              <>
                Forward cost of a straight-down step, C<sub>U</sub>
              </>
            )
          }
        >
          <ImageView
            width={view.width}
            values={view.shown}
            scale={state.mode === 'remove' ? 'diverging' : 'sequential'}
            range={seq.shownRange}
            lines={lines}
            onPointer={pointer('energy')}
            ariaLabel="Energy map of the current image"
          />
        </Panel>
        <Panel title="Cumulative minimum cost M">
          <ImageView
            width={view.width}
            values={M}
            range={seq.costRange}
            lines={lines}
            onPointer={pointer('cost')}
            ariaLabel="Cumulative minimum cost M with the next seam"
          />
        </Panel>
      </div>
    </Figure>
  )
}
