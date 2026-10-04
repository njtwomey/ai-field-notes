import { useMemo } from 'react'
import { choice, Figure, int, Plot, Raster, Readout, useAxis, useFigureState } from 'aifn-render'
import { AXIS, SIZE, type Image } from '../_shared/image'
import { stream, uniform } from 'aifn/foundation/random'

type Op = 'erode' | 'dilate' | 'open' | 'close' | 'open-close' | 'gradient'
type Shape = 'square' | 'disk'

/** Binary shapes with a one-pixel line, a narrow gap and salt-and-pepper noise (2% of pixels flipped). */
const INPUT: Image = (() => {
  const g = stream(11)
  return Array.from({ length: SIZE }, (_, r) =>
    Array.from({ length: SIZE }, (_, c) => {
      let v = 0
      if (r >= 8 && r <= 27 && c >= 6 && c <= 25) v = 1
      // A two-pixel gap splits the rectangle.
      if (c >= 15 && c <= 16 && r >= 8 && r <= 27) v = 0
      if ((r - 46) ** 2 + (c - 18) ** 2 <= 12 ** 2) v = 1
      if (r >= 40 && r <= 52 && c >= 38 && c <= 58) v = 1
      // A one-pixel line.
      if (r === 20 && c >= 32 && c <= 58) v = 1
      return uniform(g) < 0.02 ? 1 - v : v
    }),
  )
})()

function offsets(shape: Shape, radius: number): [number, number][] {
  const out: [number, number][] = []
  for (let dr = -radius; dr <= radius; dr++)
    for (let dc = -radius; dc <= radius; dc++)
      if (shape === 'square' || dr * dr + dc * dc <= radius * radius + 0.5) out.push([dr, dc])
  return out
}

/** Minimum (erosion) or maximum (dilation) over the structuring element; outside pixels are ignored. */
function rank(img: Image, se: [number, number][], useMax: boolean): Image {
  return img.map((row, r) =>
    row.map((_, c) => {
      let best = useMax ? 0 : 1
      for (const [dr, dc] of se) {
        const rr = r + dr
        const cc = c + dc
        if (rr < 0 || rr >= SIZE || cc < 0 || cc >= SIZE) continue
        const v = img[rr][cc]
        best = useMax ? Math.max(best, v) : Math.min(best, v)
      }
      return best
    }),
  )
}

function apply(op: Op, se: [number, number][]): Image {
  const erode = (x: Image) => rank(x, se, false)
  const dilate = (x: Image) => rank(x, se, true)
  switch (op) {
    case 'erode':
      return erode(INPUT)
    case 'dilate':
      return dilate(INPUT)
    case 'open':
      return dilate(erode(INPUT))
    case 'close':
      return erode(dilate(INPUT))
    case 'open-close':
      return erode(dilate(dilate(erode(INPUT))))
    case 'gradient': {
      const d = dilate(INPUT)
      const e = erode(INPUT)
      return d.map((row, r) => row.map((v, c) => v - e[r][c]))
    }
  }
}

const area = (img: Image) => img.reduce((a, row) => a + row.reduce((s, v) => s + v, 0), 0)

/** Binary erosion, dilation, opening and closing of a noisy image with a square or disk structuring element. */
export function MorphologyExplorer() {
  const state = useFigureState({
    op: choice<Op>(
      [
        { value: 'erode', label: 'erode' },
        { value: 'dilate', label: 'dilate' },
        { value: 'open', label: 'open' },
        { value: 'close', label: 'close' },
        { value: 'open-close', label: 'open, close' },
        { value: 'gradient', label: 'gradient' },
      ],
      'open',
      { label: 'operation' },
    ),
    shape: choice<Shape>(
      [
        { value: 'square', label: 'square' },
        { value: 'disk', label: 'disk' },
      ],
      'square',
      { label: 'structuring element' },
    ),
    radius: int(1, { min: 1, max: 3, step: 1, label: 'radius', format: (v) => `${v} (${2 * v + 1} × ${2 * v + 1})` }),
  })
  const se = useMemo(() => offsets(state.shape, state.radius), [state.shape, state.radius])
  const out = useMemo(() => apply(state.op, se), [state.op, se])

  const xAxis = useAxis({ label: 'column' })
  const yAxis = useAxis({ label: 'row' })
  const xAxis2 = useAxis({ label: 'column' })
  const yAxis2 = useAxis({ label: 'row' })
  return (
    <Figure
      title="Erosion, dilation, opening and closing"
      state={state}
      caption="Left: a binary image with a one-pixel line, a two-pixel gap and salt-and-pepper noise. Right: the result. Erosion shrinks shapes and deletes anything thinner than the structuring element; dilation grows them and fills small holes. Opening (erode, then dilate) removes specks and thin lines but restores the shapes that survive. Closing (dilate, then erode) fills holes and gaps. The gradient, dilation minus erosion, outlines the shapes."

      readouts={
        <>
          <Readout label="element size" value={`${se.length} pixels`} />
          <Readout label="foreground before" value={String(area(INPUT))} />
          <Readout label="foreground after" value={String(area(out))} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={320} ariaLabel={'Binary input image'}>
          <Raster x={AXIS} y={AXIS} z={INPUT} range={[0, 1]} valueLabel={'pixel'} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={320} ariaLabel={'Result of the morphological operation'}>
          <Raster x={AXIS} y={AXIS} z={out} range={[0, 1]} valueLabel={'pixel'} />
        </Plot>
      </div>
    </Figure>
  )
}
