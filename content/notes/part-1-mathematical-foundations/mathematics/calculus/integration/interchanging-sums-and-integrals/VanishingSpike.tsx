import { Figure, int, Plot, Readout, seriesLayers, type SeriesSpec, useAxis, useFigureState } from 'aifn-render'

/**
 * The partial sums of the counterexample are spikes g_N = N on (0, 1/N): every one has area 1, yet at each x > 0 the
 * spikes are eventually 0. The integral of the limit (0) is not the limit of the integrals (1).
 */
export function VanishingSpike() {
  const state = useFigureState({
    n: int(3, { min: 1, max: 40, step: 1, label: 'terms summed, N', format: (v) => String(v) }),
  })
  const N = state.n
  const spike: SeriesSpec = {
    name: `partial sum g_${N}`,
    type: 'line',
    x: [0, 0, 1 / N, 1 / N, 1],
    y: [0, N, N, 0, 0],
    slot: 0,
    area: true,
  }
  const xAxis = useAxis({ label: 'x', range: [0, 1] })
  const yAxis = useAxis({ label: 'partial sum', range: [0, 42] })
  return (
    <Figure
      title="A spike that keeps its area"
      state={state}
      caption="The partial sum of the first N terms is a spike of height N on (0, 1/N). Its area is always 1, so the sum of the integrals is 1. But at any fixed x > 0 the spike moves past x once N > 1/x, so the partial sums tend to 0 everywhere and the integral of the sum is 0. Step N up with the arrows."

      readouts={
        <>
          <Readout label="height N" value={N} />
          <Readout label="width 1/N" value={(1 / N).toFixed(3)} />
          <Readout label="area" value="1" />
          <Readout label="value at x = 0.1" value={N < 10 ? N : 0} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-xl">
        <Plot x={xAxis} y={yAxis} height={300}>
          {seriesLayers([spike])}
        </Plot>
      </div>
    </Figure>
  )
}
