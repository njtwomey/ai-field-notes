import { useMemo } from 'react'
import { Curve, Figure, formatNumber, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'

const OMEGA = toFlat(linspace(-3 * Math.PI, 3 * Math.PI, 1801))

/**
 * The DTFT of a length-L rectangular pulse, |sin(ωL/2) / sin(ω/2)|, over three periods. It repeats every 2π, and its
 * main lobe narrows as the pulse lengthens.
 */
export function PulseSpectrum() {
  const state = useFigureState({
    length: int(8, { min: 1, max: 32, step: 1, label: 'pulse length L', format: (v) => String(v) }),
  })

  const magnitude = useMemo(() => {
    const L = state.length
    return OMEGA.map((w) => {
      const s = Math.sin(w / 2)
      // At multiples of 2π the ratio tends to L.
      return Math.abs(s) < 1e-9 ? L : Math.abs(Math.sin((w * L) / 2) / s)
    })
  }, [state.length])

  const series = [{ name: '|X(e^{iω})|', x: OMEGA.map((w) => w / Math.PI), y: magnitude, slot: 0 }] as const

  const xAxis = useAxis({ label: 'ω / π', range: [-3, 3] })
  const yAxis = useAxis({ label: 'magnitude', hold: 'union' })
  return (
    <Figure
      title="The DTFT of a pulse"
      state={state}
      caption="A rectangular pulse of L ones has DTFT magnitude |sin(ωL/2) / sin(ω/2)|, the Dirichlet kernel. The pattern repeats every 2π, because e^{iωn} does. The peak height equals L, and the first nulls sit at ω = ±2π/L: a longer pulse gives a narrower spectrum."

      readouts={
        <>
          <Readout label="peak |X(e^{i0})| = L" value={state.length} />
          <Readout label="first null at ω/π" value={formatNumber(2 / state.length)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={260}>
        <Curve {...series[0]} />
      </Plot>
    </Figure>
  )
}
