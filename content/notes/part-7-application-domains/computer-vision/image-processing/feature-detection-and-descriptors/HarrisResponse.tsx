import { useMemo } from 'react'
import { choice, Figure, float, Plot, Points, Raster, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { AXIS, SIZE, gaussianBlur, gradients, testImage, type Image } from '../_shared/image'

type Method = 'harris' | 'shi-tomasi'

const INPUT = testImage(0.02)
// Gradients at a fixed derivative scale of 1 pixel; the integration scale is the slider.
const { gx: GX, gy: GY } = gradients(INPUT, 1)
const products = (f: (a: number, b: number) => number): Image => GX.map((row, r) => row.map((v, c) => f(v, GY[r][c])))
const XX = products((a) => a * a)
const YY = products((_, b) => b * b)
const XY = products((a, b) => a * b)

/** Structure-tensor entries summed over a Gaussian window of width `sigma`. */
function structureTensor(sigma: number) {
  return { a: gaussianBlur(XX, sigma), b: gaussianBlur(XY, sigma), c: gaussianBlur(YY, sigma) }
}

function response(t: { a: Image; b: Image; c: Image }, method: Method, k: number): Image {
  return t.a.map((row, r) =>
    row.map((a, col) => {
      const b = t.b[r][col]
      const c = t.c[r][col]
      const det = a * c - b * b
      const tr = a + c
      if (method === 'harris') return det - k * tr * tr
      // Smaller eigenvalue of [[a, b], [b, c]].
      return tr / 2 - Math.sqrt(((a - c) / 2) ** 2 + b * b)
    }),
  )
}

/** Pixels that are the maximum of their 3 × 3 neighbourhood and exceed `threshold`. */
function peaks(R: Image, threshold: number): { x: number[]; y: number[] } {
  const x: number[] = []
  const y: number[] = []
  for (let r = 1; r < SIZE - 1; r++)
    for (let c = 1; c < SIZE - 1; c++) {
      const v = R[r][c]
      if (v <= threshold) continue
      let isMax = true
      for (let dr = -1; dr <= 1 && isMax; dr++)
        for (let dc = -1; dc <= 1; dc++) if ((dr || dc) && R[r + dr][c + dc] >= v) isMax = false
      if (isMax) {
        x.push(c)
        y.push(r)
      }
    }
  return { x, y }
}

/** Harris (or Shi–Tomasi) corner response on a test image, with detected corners overlaid. */
export function HarrisResponse() {
  const state = useFigureState({
    method: choice<Method>(
      [
        { value: 'harris', label: 'Harris' },
        { value: 'shi-tomasi', label: 'Shi–Tomasi' },
      ],
      'harris',
      { label: 'response' },
    ),
    k: float(0.05, {
      min: 0.01,
      max: 0.24,
      step: 0.01,
      label: 'Harris k',
      format: (v) => v.toFixed(2),
      when: (v) => v.method === 'harris',
    }),
    sigma: float(1.5, { min: 0.7, max: 3, step: 0.1, label: 'window σ (pixels)', format: (v) => v.toFixed(1) }),
    frac: slider(0.01, 0.3, 0.05, {
      step: 0.01,
      label: 'threshold (fraction of max |R|)',
      format: (v) => v.toFixed(2),
    }),
  })

  const tensor = useMemo(() => structureTensor(state.sigma), [state.sigma])
  const R = useMemo(() => response(tensor, state.method, state.k), [tensor, state.method, state.k])
  const bound = useMemo(() => R.reduce((m, row) => row.reduce((mm, v) => Math.max(mm, Math.abs(v)), m), 1e-12), [R])
  const corners = useMemo(() => peaks(R, state.frac * bound), [R, state.frac, bound])
  const edgeCount = useMemo(
    () => R.reduce((n, row) => n + row.filter((v) => v < -state.frac * bound).length, 0),
    [R, state.frac, bound],
  )
  const overlay = [{ name: 'corners', x: corners.x, y: corners.y, emphasis: true }] as const

  const xAxis = useAxis({ label: 'column' })
  const yAxis = useAxis({ label: 'row' })
  const xAxis2 = useAxis({ label: 'column' })
  const yAxis2 = useAxis({ label: 'row' })
  return (
    <Figure
      title="Corner response from the structure tensor"
      state={state}
      caption="Left: the test image. Right: the corner response at every pixel, with detected corners (local maxima above the threshold) as diamonds. Harris's R = det M − k (tr M)² is positive at corners, negative along edges (one large eigenvalue) and near zero on flat regions. Raising k makes the detector stricter: points where one eigenvalue dominates turn negative. Shi–Tomasi uses the smaller eigenvalue, which is never negative. A wider integration window σ blurs the response and merges nearby corners."

      readouts={
        <>
          <Readout label="corners detected" value={String(corners.x.length)} />
          {state.method === 'harris' && (
            <>
              <Readout label="edge pixels (R below −threshold)" value={String(edgeCount)} />
              <Readout label="largest eigenvalue ratio accepted" value={maxRatio(state.k)} />
            </>
          )}
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={320} ariaLabel={'Test image with detected corners'}>
          <Raster x={AXIS} y={AXIS} z={INPUT} range={[0, 1]} valueLabel={'intensity'} />
          <Points {...overlay[0]} live />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={320} ariaLabel={'Corner response'}>
          <Raster
            x={AXIS}
            y={AXIS}
            z={R}
            scale={state.method === 'harris' ? 'diverging' : 'sequential'}
            range={state.method === 'harris' ? [-bound, bound] : [0, bound]}
            valueLabel={'response'}
          />
          <Points {...overlay[0]} live />
        </Plot>
      </div>
    </Figure>
  )
}

/** R > 0 exactly when the eigenvalue ratio ρ satisfies ρ / (1 + ρ)² > k; return the larger root. */
function maxRatio(k: number): string {
  const q = 1 / k - 2
  const rho = (q + Math.sqrt(q * q - 4)) / 2
  return Number.isFinite(rho) ? rho.toFixed(1) : '—'
}
