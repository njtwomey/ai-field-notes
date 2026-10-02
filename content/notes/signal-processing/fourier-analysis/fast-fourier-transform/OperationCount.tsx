import { useMemo } from 'react'
import {
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from 'aifn-render'

const EXPONENTS = Array.from({ length: 21 }, (_, i) => i + 2)

/** Complex multiplications for a length-N DFT: N² computed directly, (N/2) log₂ N with a radix-2 FFT. */
export function OperationCount() {
  const p = useParam(10, { min: 2, max: 22, step: 1 })
  const n = 2 ** p.value

  const series = useMemo(
    (): XYSeries[] => [
      { name: 'direct DFT: N²', type: 'line', x: EXPONENTS, y: EXPONENTS.map((e) => 2 * e), slot: 0 },
      {
        name: 'radix-2 FFT: (N/2) log₂ N',
        type: 'line',
        x: EXPONENTS,
        y: EXPONENTS.map((e) => e - 1 + Math.log2(e)),
        slot: 1,
      },
    ],
    [],
  )
  const handles: Handle[] = [{ kind: 'x', at: p.value, label: 'N', onDrag: (v) => p.set(Math.round(v)) }]
  const direct = n * n
  const fast = (n / 2) * p.value

  return (
    <Interactive
      title="How much the FFT saves"
      caption="Complex multiplications for a length-N transform, on a log₂ scale on both axes. The direct DFT grows as N², the radix-2 FFT as (N/2) log₂ N. The gap is the speed-up, and it widens without bound. Drag N or use the slider."
      controls={<ParamSlider label="log₂ N" param={p} format={(v) => String(v)} withArrows />}
      readout={
        <>
          <Readout label="N" value={n.toLocaleString()} />
          <Readout label="direct" value={direct.toExponential(2)} />
          <Readout label="FFT" value={fast.toExponential(2)} />
          <Readout label="speed-up" value={`${formatNumber(direct / fast)}×`} />
        </>
      }
    >
      <XYChart
        series={series}
        xLabel="log₂ N"
        yLabel="log₂ (multiplications)"
        xRange={[2, 22]}
        handles={handles}
        height={260}
      />
    </Interactive>
  )
}
