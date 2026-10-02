import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from 'aifn-render'
import { FILTERS, wavedec, waverec, type Family } from '../_shared/wavelets'

type Shape = 'steps' | 'chirp'

const N = 512
const T = Array.from({ length: N }, (_, n) => n / N)

function makeSignal(shape: Shape): Float64Array {
  const x = new Float64Array(N)
  for (let n = 0; n < N; n++) {
    const t = T[n]
    x[n] =
      shape === 'steps'
        ? (t > 0.25 ? 1 : 0) - (t > 0.6 ? 1.5 : 0) + 0.8 * t + 0.25 * Math.sin(2 * Math.PI * 24 * t)
        : Math.sin(2 * Math.PI * (2 * t + 30 * t * t)) * (1 - 0.5 * t) + 0.5 * Math.cos(2 * Math.PI * 1.5 * t)
  }
  return x
}

/**
 * A J-level orthogonal wavelet decomposition splits a signal into a coarse approximation A_J and details D_1 … D_J,
 * each projected back to full length. They sum exactly to the signal, and their energies add.
 */
export function MraComponents() {
  const [shape, setShape] = useState<Shape>('steps')
  const [family, setFamily] = useState<Family>('db4')
  const levels = useParam(4, { min: 1, max: 5, step: 1 })

  const r = useMemo(() => {
    const x = makeSignal(shape)
    const h = FILTERS[family]
    const { approx, details } = wavedec(x, h, levels.value)
    const zero = (v: Float64Array) => new Float64Array(v.length)
    // Each component: reconstruct with every other coefficient set zeroed.
    const a = waverec(approx, details.map(zero), h)
    const ds = details.map((_, j) =>
      waverec(
        zero(approx),
        details.map((d, i) => (i === j ? d : zero(d))),
        h,
      ),
    )
    const energy = (v: ArrayLike<number>) => Array.from(v).reduce((s, u) => s + u * u, 0)
    const total = energy(x)
    const coeffEnergy = [energy(approx), ...details.map(energy)]
    const sum = Float64Array.from(x, (_, n) => a[n] + ds.reduce((s, d) => s + d[n], 0))
    const error = Math.max(...Array.from(x, (v, n) => Math.abs(v - sum[n])))
    return { x, a, ds, total, coeffEnergy, error }
  }, [shape, family, levels.value])

  // Stack the components with vertical offsets so each has its own band.
  const gap = 2.2
  const J = levels.value
  const series: XYSeries[] = [
    { name: 'signal', type: 'line', x: T, y: Array.from(r.x), slot: 0 },
    { name: `A${J}`, type: 'line', x: T, y: Array.from(r.a, (v) => v - gap), slot: 1 },
    ...r.ds
      .map((d, j) => ({ d, j }))
      .reverse()
      .map(({ d, j }, i): XYSeries => ({
        name: `D${j + 1}`,
        type: 'line',
        x: T,
        y: Array.from(d, (v) => v - gap * (i + 2)),
        slot: 2 + (J - 1 - j),
      })),
  ]

  return (
    <Interactive
      title="A signal as approximation plus details"
      caption="Top: the signal. Below, offset for legibility: the level-J approximation A_J, then the details D_J (coarsest) down to D_1 (finest), each reconstructed to full length with every other coefficient zeroed. The components add up to the signal exactly, and, because the transform is orthogonal, the coefficient energies add up to the signal's energy. Steps and kinks put detail energy at every level near the discontinuity; smooth parts go to the approximation."
      controls={
        <>
          <ParamChoice
            label="signal"
            value={shape}
            onChange={setShape}
            options={[
              { value: 'steps', label: 'steps and a tone' },
              { value: 'chirp', label: 'chirp' },
            ]}
          />
          <ParamChoice
            label="wavelet"
            value={family}
            onChange={setFamily}
            options={[
              { value: 'haar', label: 'Haar' },
              { value: 'db2', label: 'db2' },
              { value: 'db4', label: 'db4' },
            ]}
          />
          <ParamSlider label="levels J" param={levels} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
        <>
          <Readout
            label="energy share A, then D_J … D_1"
            value={[r.coeffEnergy[0], ...r.coeffEnergy.slice(1).reverse()]
              .map((e) => `${formatNumber((100 * e) / r.total)}%`)
              .join(', ')}
          />
          <Readout label="max reconstruction error" value={r.error.toExponential(1)} />
        </>
      }
    >
      <XYChart series={series} xLabel="t" yLabel="components (offset)" xRange={[0, 1]} height={420} />
    </Interactive>
  )
}
