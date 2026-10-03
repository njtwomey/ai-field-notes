import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from 'aifn-render'
import { linspace } from '@/lib/math'

const OMEGA = linspace(-3 * Math.PI, 3 * Math.PI, 1801)

/**
 * The DTFT of a length-L rectangular pulse, |sin(ωL/2) / sin(ω/2)|, over three periods. It repeats every 2π, and its
 * main lobe narrows as the pulse lengthens.
 */
export function PulseSpectrum() {
  const length = useParam(8, { min: 1, max: 32, step: 1 })

  const magnitude = useMemo(() => {
    const L = length.value
    return OMEGA.map((w) => {
      const s = Math.sin(w / 2)
      // At multiples of 2π the ratio tends to L.
      return Math.abs(s) < 1e-9 ? L : Math.abs(Math.sin((w * L) / 2) / s)
    })
  }, [length.value])

  const series: XYSeries[] = [
    { name: '|X(e^{iω})|', type: 'line', x: OMEGA.map((w) => w / Math.PI), y: magnitude, slot: 0 },
  ]

  return (
    <Interactive
      title="The DTFT of a pulse"
      caption="A rectangular pulse of L ones has DTFT magnitude |sin(ωL/2) / sin(ω/2)|, the Dirichlet kernel. The pattern repeats every 2π, because e^{iωn} does. The peak height equals L, and the first nulls sit at ω = ±2π/L: a longer pulse gives a narrower spectrum."
      controls={<ParamSlider label="pulse length L" param={length} format={(v) => String(v)} withArrows />}
      readout={
        <>
          <Readout label="peak |X(e^{i0})| = L" value={length.value} />
          <Readout label="first null at ω/π" value={formatNumber(2 / length.value)} />
        </>
      }
    >
      <XYChart series={series} xLabel="ω / π" yLabel="magnitude" xRange={[-3, 3]} height={260} />
    </Interactive>
  )
}
