import { useContext, useEffect, useState, type ReactNode } from 'react'
import { Contours, Curve, FrameContext, Handle, Pixels, Plot, useAxis } from 'aifn-render'
import { trainSmallCnn, type CnnParams, type TrainSetup } from './small-cnn'

/** Plots inside a Figure take the frame's height; this gives the plots inside it a fixed height instead. */
export function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}

/** A training run in progress or finished: the minibatch loss of each step so far, and the parameters at the end. */
export type TrainingProgress = { losses: number[]; params: CnnParams | null; done: boolean }

/** Milliseconds of work per slice, and the most training steps per slice (the clock may be frozen in a test render). */
const SLICE_MS = 25
const SLICE_STEPS = 4

/** Finished runs by setup, so two figures (or a revisit) with the same recipe train once per page load. */
const finished = new Map<string, TrainingProgress>()

/**
 * Trains the chapter's small network on a setup in short slices between frames and returns the progress so far. A
 * new setup restarts the run; a setup trained before returns at once.
 */
export function useTrainedCnn(setup: TrainSetup): TrainingProgress {
  const key = JSON.stringify(setup)
  const [state, setState] = useState<{ key: string; progress: TrainingProgress }>(() => ({
    key,
    progress: finished.get(key) ?? { losses: [], params: null, done: false },
  }))
  useEffect(() => {
    const cached = finished.get(key)
    if (cached) {
      setState({ key, progress: cached })
      return
    }
    const it = trainSmallCnn(JSON.parse(key) as TrainSetup)
    const losses: number[] = []
    let params: CnnParams | null = null
    let timer = 0
    const slice = () => {
      const start = performance.now()
      let done = false
      for (let k = 0; k < SLICE_STEPS && performance.now() - start < SLICE_MS; k++) {
        const next = it.next()
        if (next.done) {
          done = true
          break
        }
        losses.push(next.value.loss)
        if (next.value.params) {
          params = next.value.params
          done = true
          break
        }
      }
      const progress = { losses: [...losses], params, done }
      if (done) finished.set(key, progress)
      setState({ key, progress })
      if (!done) timer = window.setTimeout(slice, 0)
    }
    timer = window.setTimeout(slice, 0)
    return () => window.clearTimeout(timer)
  }, [key])
  return state.key === key ? state.progress : (finished.get(key) ?? { losses: [], params: null, done: false })
}

/** An outline drawn on an image tile: a closed polyline in (column, row) pixel coordinates. */
export type TileOutline = { name: string; x: number[]; y: number[]; slot?: number; emphasis?: boolean }

/** A draggable point on an image tile, in (column, row) pixel coordinates. */
export type TileHandle = { at: [number, number]; onDrag: (p: [number, number]) => void; label: string }

/**
 * A small greyscale or heat-map image on pixel axes, row 0 at the top, square pixels, with optional outlines, contour
 * lines of a second field and handles.
 */
export function ImageTile({
  values,
  width,
  height,
  scale = 'sequential',
  range,
  title,
  outlines,
  contour,
  handles,
  ariaLabel,
}: {
  values: ArrayLike<number>
  width: number
  height: number
  scale?: 'sequential' | 'diverging'
  range?: [number, number]
  title?: string
  outlines?: TileOutline[]
  contour?: { field: ArrayLike<number>; levels: number[] }
  handles?: TileHandle[]
  ariaLabel: string
}) {
  const x = useAxis({ range: [-0.5, width - 0.5], nice: false, zoom: false })
  const y = useAxis({ range: [-0.5, height - 0.5], nice: false, inverse: true, equal: x, zoom: false })
  const cols = Array.from({ length: width }, (_, j) => j)
  const rows = Array.from({ length: height }, (_, i) => i)
  return (
    <Plot x={x} y={y} bare fitHeight title={title} ariaLabel={ariaLabel} legend={false}>
      <Pixels width={width} height={height} values={values} scale={scale} range={range} />
      {contour && (
        <Contours
          x={cols}
          y={rows}
          z={rows.map((i) => cols.map((j) => contour.field[i * width + j]))}
          levels={contour.levels}
          labels={false}
          width={2}
          emphasis
        />
      )}
      {outlines?.map((o) => (
        <Curve key={o.name} name={o.name} x={o.x} y={o.y} slot={o.slot} emphasis={o.emphasis} width={2} live />
      ))}
      {handles?.map((h) => (
        <Handle key={h.label} kind="point" at={h.at} onDrag={h.onDrag} label={h.label} />
      ))}
    </Plot>
  )
}

/** The outline of the pixel block rows r0..r1, columns c0..c1 (inclusive), clipped to the image. */
export function boxOutline(r0: number, r1: number, c0: number, c1: number, size: number) {
  const a = Math.max(0, r0) - 0.5
  const b = Math.min(size - 1, r1) + 0.5
  const c = Math.max(0, c0) - 0.5
  const d = Math.min(size - 1, c1) + 0.5
  return { x: [c, d, d, c, c], y: [a, a, b, b, a] }
}
