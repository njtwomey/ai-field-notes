import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import {
  ImagePlot,
  Interactive,
  ParamChoice,
  ParamSlider,
  ParamSwitch,
  Readout,
  formatNumber,
  useParam,
  type Handle,
  type ImagePlotLine,
  type PlotPointer,
} from '@/components/viz'
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

const seamLine = (name: string, seam: ArrayLike<number>, extra: Partial<ImagePlotLine> = {}): ImagePlotLine => ({
  name,
  x: Array.from(seam),
  y: ROWS,
  ...extra,
})

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
  const [image, setImage] = useState<ImageId>('landscape')
  const [energy, setEnergy] = useState<Energy>('backward')
  const [mode, setMode] = useState<BoxMode>('off')
  const [compare, setCompare] = useState(true)
  const width = useParam(90, { min: W - MAX_REMOVE, max: W + MAX_INSERT, step: 1 })
  // The box as drawn while dragging, and the box the removal sequence uses, committed when the drag ends.
  const [box, setBox] = useState<Box>(DEFAULT_BOX)
  const [committed, setCommitted] = useState<Box>(DEFAULT_BOX)
  const draft = useRef(box)
  const [hover, setHover] = useState<{ r: number; c: number; panel: 'image' | 'energy' | 'cost' } | null>(null)

  const wanted: Settings = { image, energy, mode, box: committed }
  const key = keyOf(wanted)
  const [ready, setReady] = useState(() => ({ key, seq: sequenceFor(wanted) }))
  const busy = ready.key !== key
  useEffect(() => {
    if (!busy) return
    const settings: Settings = { image, energy, mode, box: committed }
    // Wait a frame so "recomputing…" can paint before the carving blocks the main thread.
    const id = setTimeout(() => setReady({ key, seq: sequenceFor(settings) }), 16)
    return () => clearTimeout(id)
  }, [busy, key, image, energy, mode, committed])

  const seq = ready.seq
  const w = width.value
  const view = useMemo(() => viewAt(seq, w), [seq, w])
  const plain = useMemo(() => (compare ? scaled(seq.rgb, w) : null), [compare, seq, w])

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
    const out: ImagePlotLine[] = [seamLine('next seam', view.seam, { slot: 1 })]
    if (through) out.push(seamLine('seam through pixel', through.seam, { emphasis: true, dashed: true }))
    return out
  }, [view, through])
  const imageLines = useMemo(
    () =>
      mode === 'off'
        ? lines
        : [
            ...lines,
            {
              name: 'box',
              x: [left, right, right, left, left],
              y: [top, top, bottom, bottom, top],
              slot: mode === 'protect' ? 2 : 7,
            },
          ],
    [lines, mode, left, right, top, bottom],
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
    onDrag: (x) => width.set(Math.round(x + 0.5)),
  }
  const handles: Handle[] =
    mode === 'off'
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
    <Interactive
      title="Seam carving an image"
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
      controls={
        <>
          <ParamSlider
            label="width"
            param={width}
            withArrows
            format={(v) =>
              v < W ? `${v} px (${W - v} removed)` : v > W ? `${v} px (${v - W} inserted)` : `${v} px (original)`
            }
          />
          <ParamChoice
            label="image"
            value={image}
            onChange={setImage}
            options={[
              { value: 'landscape', label: 'landscape' },
              { value: 'shapes', label: 'shapes' },
              { value: 'checkerboard', label: 'checkerboard' },
              { value: 'text', label: 'text' },
              { value: 'line', label: 'straight line' },
            ]}
          />
          <ParamChoice
            label="energy"
            value={energy}
            onChange={setEnergy}
            options={[
              { value: 'backward', label: 'backward' },
              { value: 'forward', label: 'forward' },
            ]}
          />
          <ParamChoice
            label="box"
            value={mode}
            onChange={setMode}
            options={[
              { value: 'off', label: 'off' },
              { value: 'protect', label: 'protect' },
              { value: 'remove', label: 'remove' },
            ]}
          />
          <ParamSwitch label="compare with plain scaling" checked={compare} onChange={setCompare} />
        </>
      }
      readout={
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
          <ImagePlot
            width={view.width}
            height={H}
            rgb={view.rgb}
            xExtent={EXTENT}
            lines={imageLines}
            handles={handles}
            onPointer={pointer('image')}
            ariaLabel="Seam-carved image with the next seam and a draggable right edge"
          />
        </Panel>
        {plain && (
          <Panel title="Plain scaling to the same width">
            <ImagePlot
              width={w}
              height={H}
              rgb={plain}
              xExtent={EXTENT}
              handles={[widthHandle]}
              ariaLabel="The original image scaled to the same width"
            />
          </Panel>
        )}
        <Panel
          title={
            energy === 'backward' ? (
              'Energy e (gradient magnitude)'
            ) : (
              <>
                Forward cost of a straight-down step, C<sub>U</sub>
              </>
            )
          }
        >
          <ImagePlot
            width={view.width}
            height={H}
            values={view.shown}
            scale={mode === 'remove' ? 'diverging' : 'sequential'}
            range={seq.shownRange}
            xExtent={EXTENT}
            lines={lines}
            onPointer={pointer('energy')}
            ariaLabel="Energy map of the current image"
          />
        </Panel>
        <Panel title="Cumulative minimum cost M">
          <ImagePlot
            width={view.width}
            height={H}
            values={M}
            range={seq.costRange}
            xExtent={EXTENT}
            lines={lines}
            onPointer={pointer('cost')}
            ariaLabel="Cumulative minimum cost M with the next seam"
          />
        </Panel>
      </div>
    </Interactive>
  )
}
