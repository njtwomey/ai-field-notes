import { useMemo } from 'react'
import {
  choice,
  Figure,
  formatNumber,
  Plot,
  Raster,
  Readout,
  setting,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import {
  AXIS,
  SOBEL_X,
  SOBEL_Y,
  correlate,
  extent,
  gaussian1d,
  logKernel,
  mapImage,
  separable,
  testImage,
  type Border,
  type Image,
} from '../_shared/image'

type KernelName = 'box' | 'gaussian' | 'sobel-x' | 'sobel-y' | 'log' | 'sharpen'

const NOISY = testImage(0.08)
const CLEAN = testImage(0)

type Result = { out: Image; size: number; separable: boolean; signed: boolean }

function apply(img: Image, kernel: KernelName, boxSize: number, sigma: number, border: Border): Result {
  switch (kernel) {
    case 'box': {
      const k = new Array<number>(boxSize).fill(1 / boxSize)
      return { out: separable(img, k, k, border), size: boxSize, separable: true, signed: false }
    }
    case 'gaussian': {
      const g = gaussian1d(sigma)
      return { out: separable(img, g, g, border), size: g.length, separable: true, signed: false }
    }
    case 'sobel-x':
      return { out: correlate(img, SOBEL_X, border), size: 3, separable: true, signed: true }
    case 'sobel-y':
      return { out: correlate(img, SOBEL_Y, border), size: 3, separable: true, signed: true }
    case 'log': {
      const k = logKernel(sigma)
      return { out: correlate(img, k, border), size: k.length, separable: false, signed: true }
    }
    case 'sharpen': {
      // Unsharp masking: add back the detail that a Gaussian blur removes, with gain 1.5.
      const g = gaussian1d(sigma)
      const blurred = separable(img, g, g, border)
      return {
        out: mapImage(img, (v, r, c) => v + 1.5 * (v - blurred[r][c])),
        size: g.length,
        separable: true,
        signed: false,
      }
    }
  }
}

/** A 64 × 64 test image and the result of filtering it with a chosen kernel and border rule. */
export function ImageFilterExplorer() {
  const state = useFigureState({
    kernel: choice<KernelName>(
      [
        { value: 'box', label: 'box' },
        { value: 'gaussian', label: 'Gaussian' },
        { value: 'sobel-x', label: 'Sobel x' },
        { value: 'sobel-y', label: 'Sobel y' },
        { value: 'log', label: 'LoG' },
        { value: 'sharpen', label: 'sharpen' },
      ],
      'gaussian',
      { label: 'kernel' },
    ),
    border: choice<Border>(
      [
        { value: 'zero', label: 'zero' },
        { value: 'replicate', label: 'replicate' },
        { value: 'reflect', label: 'reflect' },
      ],
      'zero',
      { label: 'border' },
    ),
    noisy: setting(true, 'add noise (σ = 0.08)'),
    boxSize: slider(3, 11, 5, {
      step: 2,
      label: 'box size k',
      format: (v) => `${v} × ${v}`,
      when: (v) => v.kernel === 'box',
    }),
    sigma: slider(0.5, 3, 1.5, {
      step: 0.1,
      label: 'Gaussian σ (pixels)',
      format: (v) => v.toFixed(1),
      when: (v) => v.kernel === 'gaussian' || v.kernel === 'log' || v.kernel === 'sharpen',
    }),
  })
  const { kernel, border, noisy, boxSize, sigma } = state
  const input = noisy ? NOISY : CLEAN

  const res = useMemo(() => apply(input, kernel, boxSize, sigma, border), [input, kernel, boxSize, sigma, border])
  const [lo, hi] = useMemo(() => extent(res.out), [res])
  const bound = Math.max(Math.abs(lo), Math.abs(hi), 1e-9)
  const outRange: [number, number] = res.signed ? [-bound, bound] : [Math.min(0, lo), Math.max(1, hi)]
  const k = res.size

  const xAxis = useAxis({ label: 'column' })
  const yAxis = useAxis({ label: 'row' })
  const xAxis2 = useAxis({ label: 'column' })
  const yAxis2 = useAxis({ label: 'row' })
  return (
    <Figure
      title="Filtering an image with a kernel"
      caption="Left: the input. Right: the filtered image. Box and Gaussian kernels average and blur; the Gaussian has no ringing and is isotropic. Sobel kernels respond to horizontal and vertical intensity changes with a sign. The Laplacian of Gaussian (LoG) is zero on flat regions and changes sign across an edge. Unsharp masking adds back what a Gaussian blur removes. Switch the border rule to zero padding and watch a dark frame appear at the image boundary."
      state={state}
      readouts={
        <>
          <Readout label="kernel size" value={`${k} × ${k}`} />
          <Readout label="multiply-adds per pixel, direct" value={String(k * k)} />
          <Readout label="separable" value={res.separable ? `yes: ${2 * k} per pixel` : 'no'} />
          <Readout label="output range" value={`${formatNumber(lo)} to ${formatNumber(hi)}`} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={320} ariaLabel={'Input test image'}>
          <Raster x={AXIS} y={AXIS} z={input} range={[0, 1]} valueLabel={'intensity'} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={320} ariaLabel={'Filtered image'}>
          <Raster
            x={AXIS}
            y={AXIS}
            z={res.out}
            scale={res.signed ? 'diverging' : 'sequential'}
            range={outRange}
            valueLabel={'output'}
          />
        </Plot>
      </div>
    </Figure>
  )
}
