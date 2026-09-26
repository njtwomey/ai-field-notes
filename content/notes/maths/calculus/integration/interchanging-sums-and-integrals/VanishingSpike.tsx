import { Interactive, ParamSlider, Readout, XYChart, useParam, type XYSeries } from '@/components/viz'

/**
 * The partial sums of the counterexample are spikes g_N = N on (0, 1/N): every one has area 1, yet at each x > 0 the
 * spikes are eventually 0. The integral of the limit (0) is not the limit of the integrals (1).
 */
export function VanishingSpike() {
  const n = useParam(3, { min: 1, max: 40, step: 1 })
  const N = n.value
  const spike: XYSeries = {
    name: `partial sum g_${N}`,
    type: 'line',
    x: [0, 0, 1 / N, 1 / N, 1],
    y: [0, N, N, 0, 0],
    slot: 0,
    area: true,
  }
  return (
    <Interactive
      title="A spike that keeps its area"
      caption="The partial sum of the first N terms is a spike of height N on (0, 1/N). Its area is always 1, so the sum of the integrals is 1. But at any fixed x > 0 the spike moves past x once N > 1/x, so the partial sums tend to 0 everywhere and the integral of the sum is 0. Step N up with the arrows."
      controls={<ParamSlider label="terms summed, N" param={n} format={(v) => String(v)} withArrows />}
      readout={
        <>
          <Readout label="height N" value={N} />
          <Readout label="width 1/N" value={(1 / N).toFixed(3)} />
          <Readout label="area" value="1" />
          <Readout label="value at x = 0.1" value={N < 10 ? N : 0} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-xl">
        <XYChart series={[spike]} xLabel="x" yLabel="partial sum" xRange={[0, 1]} yRange={[0, 42]} height={300} />
      </div>
    </Interactive>
  )
}
