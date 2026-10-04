import { useMemo } from 'react'
import { Curve, Figure, formatNumber, Handle, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'

const EXPONENTS = Array.from({ length: 21 }, (_, i) => i + 2)

/** Complex multiplications for a length-N DFT: N² computed directly, (N/2) log₂ N with a radix-2 FFT. */
export function OperationCount() {
  const state = useFigureState({
    p: int(10, { min: 2, max: 22, step: 1, label: 'log₂ N', format: (v) => String(v) }),
  })
  const n = 2 ** state.p

  const series = useMemo(
    () =>
      [
        { name: 'direct DFT: N²', x: EXPONENTS, y: EXPONENTS.map((e) => 2 * e), slot: 0 },
        {
          name: 'radix-2 FFT: (N/2) log₂ N',
          x: EXPONENTS,
          y: EXPONENTS.map((e) => e - 1 + Math.log2(e)),
          slot: 1,
        },
      ] as const,
    [],
  )
  const direct = n * n
  const fast = (n / 2) * state.p

  const xAxis = useAxis({ label: 'log₂ N', range: [2, 22] })
  const yAxis = useAxis({ label: 'log₂ (multiplications)', hold: 'union' })
  return (
    <Figure
      title="How much the FFT saves"
      state={state}
      caption="Complex multiplications for a length-N transform, on a log₂ scale on both axes. The direct DFT grows as N², the radix-2 FFT as (N/2) log₂ N. The gap is the speed-up, and it widens without bound. Drag N or use the slider."

      readouts={
        <>
          <Readout label="N" value={n.toLocaleString()} />
          <Readout label="direct" value={direct.toExponential(2)} />
          <Readout label="FFT" value={fast.toExponential(2)} />
          <Readout label="speed-up" value={`${formatNumber(direct / fast)}×`} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={260}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Handle kind="x" at={state.p} label="N" onDrag={(v) => state.set('p', Math.round(v))} />
      </Plot>
    </Figure>
  )
}
