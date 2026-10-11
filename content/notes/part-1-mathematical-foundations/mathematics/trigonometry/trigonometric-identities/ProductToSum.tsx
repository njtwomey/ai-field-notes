import { useMemo } from 'react'
import { choice, Curve, Figure, formatNumber, Plot, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { grid, TAU } from '../_shared/geometry'

const T = grid(0, 2 * TAU, 1201)

/**
 * Product-to-sum and sum-to-product. A product of two sinusoids is a sum of two cosines at the difference and sum
 * frequencies; a sum of two sinusoids is a fast oscillation inside a slow envelope (beats).
 */
export function ProductToSum() {
  const state = useFigureState({
    mode: choice(
      [
        { value: 'product', label: 'product sin(mt) sin(nt)' },
        { value: 'sum', label: 'sum sin(mt) + sin(nt)' },
      ],
      'product',
      { label: 'identity' },
    ),
    m: slider(0.5, 12, 4, { step: 0.5, label: 'frequency m' }),
    n: slider(0.5, 12, 3, { step: 0.5, label: 'frequency n' }),
  })
  const { mode, m, n } = state

  const curves = useMemo(() => {
    if (mode === 'product') {
      return {
        main: T.map((t) => Math.sin(m * t) * Math.sin(n * t)),
        a: T.map((t) => 0.5 * Math.cos((m - n) * t)),
        b: T.map((t) => -0.5 * Math.cos((m + n) * t)),
      }
    }
    return {
      main: T.map((t) => Math.sin(m * t) + Math.sin(n * t)),
      a: T.map((t) => 2 * Math.cos(((m - n) * t) / 2)),
      b: T.map((t) => -2 * Math.cos(((m - n) * t) / 2)),
    }
  }, [mode, m, n])

  const x = useAxis({ label: 't', range: [0, 2 * TAU] })
  const y = useAxis({ label: 'y', range: mode === 'product' ? [-1.1, 1.1] : [-2.2, 2.2] })
  const product = mode === 'product'
  return (
    <Figure
      title="Products and sums of sinusoids"
      state={state}
      caption={
        product
          ? 'The black curve is sin(mt) sin(nt). The dashed curves are ½ cos((m − n)t) and −½ cos((m + n)t); at every t they add up to the black curve. Multiplying two sinusoids creates the difference and sum frequencies and removes the originals, which is how a radio mixer shifts a signal in frequency. With m = n the slow term is the constant ½, the average of sin²(mt).'
          : 'The black curve is sin(mt) + sin(nt). It equals 2 sin((m + n)t/2) cos((m − n)t/2): a fast oscillation at the mean frequency inside the slow envelope ±2 cos((m − n)t/2) (dashed). Two close frequencies produce beats, swells that repeat every 2π/|m − n| in t. With m = n the envelope is flat.'
      }
      readouts={
        product ? (
          <>
            <Readout label="difference frequency |m − n|" value={formatNumber(Math.abs(m - n))} />
            <Readout label="sum frequency m + n" value={formatNumber(m + n)} />
          </>
        ) : (
          <>
            <Readout label="carrier frequency (m + n)/2" value={formatNumber((m + n) / 2)} />
            <Readout label="beat period 2π/|m − n|" value={m === n ? 'none' : formatNumber(TAU / Math.abs(m - n))} />
          </>
        )
      }
    >
      <Plot x={x} y={y}>
        <Curve name={product ? '½ cos((m − n)t)' : 'envelope'} x={T} y={curves.a} slot={0} dashed />
        <Curve name={product ? '−½ cos((m + n)t)' : 'envelope'} x={T} y={curves.b} slot={product ? 1 : 0} dashed />
        <Curve name={product ? 'sin(mt) sin(nt)' : 'sin(mt) + sin(nt)'} x={T} y={curves.main} emphasis />
      </Plot>
    </Figure>
  )
}
