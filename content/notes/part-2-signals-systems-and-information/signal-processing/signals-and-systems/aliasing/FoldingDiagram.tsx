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
import { linspace } from '@/lib/math'

const FS = 10
const F_MAX = 3 * FS
const FREQS = linspace(0, F_MAX, 601)
const TIMES = linspace(0, 1, 801)
/** f folded into [0, f_s/2]: the frequency a real sinusoid appears to have after sampling at f_s. */
const fold = (f: number) => Math.abs(f - FS * Math.round(f / FS))

/**
 * Left: the folding map from true to apparent frequency, a triangle wave with period f_s. Right: a sinusoid above
 * f_s/2 and its alias pass through the same samples.
 */
export function FoldingDiagram() {
  const f = useParam(13, { min: 0, max: F_MAX, step: 0.25 })
  const apparent = fold(f.value)

  const left: XYSeries[] = useMemo(
    () => [{ name: 'apparent frequency', type: 'line', x: FREQS, y: FREQS.map(fold), slot: 0 }],
    [],
  )
  const handles: Handle[] = [{ kind: 'x', at: f.value, label: 'f', onDrag: (v) => f.set(v) }]

  const right = useMemo((): XYSeries[] => {
    const n = Array.from({ length: FS + 1 }, (_, i) => i / FS)
    // The alias has the same samples; for a cosine with zero phase, cos(2π f t) and cos(2π f_a t) agree at t = n/f_s.
    return [
      {
        name: `true: ${formatNumber(f.value)} Hz`,
        type: 'line',
        x: TIMES,
        y: TIMES.map((t) => Math.cos(2 * Math.PI * f.value * t)),
        slot: 0,
      },
      {
        name: `alias: ${formatNumber(apparent)} Hz`,
        type: 'line',
        x: TIMES,
        y: TIMES.map((t) => Math.cos(2 * Math.PI * apparent * t)),
        slot: 1,
        dashed: true,
      },
      { name: 'samples', type: 'scatter', x: n, y: n.map((t) => Math.cos(2 * Math.PI * f.value * t)), emphasis: true },
    ]
  }, [f.value, apparent])

  return (
    <Interactive
      title="Frequencies fold back"
      caption="Sampling at f_s = 10 Hz maps every frequency onto [0, f_s/2]: the map folds back and forth like a triangle wave. Drag the line on the left or use the slider. On the right, the true cosine and its alias at the folded frequency pass through exactly the same samples, so after sampling they cannot be told apart."
      controls={<ParamSlider label="true frequency f (Hz)" param={f} withArrows />}
      readout={
        <>
          <Readout label="true f" value={`${formatNumber(f.value)} Hz`} />
          <Readout label="apparent" value={`${formatNumber(apparent)} Hz`} />
          <Readout label="nearest multiple of f_s" value={`${FS * Math.round(f.value / FS)} Hz`} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart
          series={left}
          xLabel="true frequency (Hz)"
          yLabel="apparent frequency (Hz)"
          xRange={[0, F_MAX]}
          yRange={[0, FS / 2 + 0.5]}
          handles={handles}
          height={280}
        />
        <XYChart series={right} xLabel="t (s)" yLabel="x(t)" xRange={[0, 1]} yRange={[-1.4, 1.4]} height={280} />
      </div>
    </Interactive>
  )
}
