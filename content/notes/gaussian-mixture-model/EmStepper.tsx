import { useEffect, useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  StepControls,
  XYChart,
  formatNumber,
  type XYSeries,
} from '@/components/viz'
import type { PointCloud2d } from '@/generated/contracts'
import { useFigure } from '@/lib/generated'
import { CENTRE_INIT_OPTIONS, type CentreInit, type Point } from '@/lib/math/cluster'
import { ellipse, finished, initialise, seek, step, type EmState } from './em'

const MAX_RUN = 300

/** EM for a diagonal Gaussian mixture, one iteration at a time, on data from python/mlc/figures/gmm.py. */
export function EmStepper() {
  const { data } = useFigure<PointCloud2d>('gaussian-mixture-model/blobs')
  const points = useMemo(() => (data ? data.x.map((x, i) => [x, data.y[i]] as Point) : []), [data])
  const [k, setK] = useState(3)
  const [init, setInit] = useState<CentreInit>('random')
  const [seed, setSeed] = useState(5)
  const [state, setState] = useState<EmState>()

  const reset = () => {
    if (points.length) setState(initialise(points, k, init, seed))
  }
  // Re-initialise whenever the data or a setting changes.
  useEffect(reset, [points, k, init, seed]) // eslint-disable-line react-hooks/exhaustive-deps

  const series = useMemo((): XYSeries[] => {
    if (!data || !state) return []
    const mixture = state.mixtures[state.cursor]
    const names = mixture.means.map((_, j) => `component ${j + 1}`)
    // Each point is coloured by its most likely component.
    const labels = state.responsibilities.map((r) => r.indexOf(Math.max(...r)))
    const ellipses = mixture.means.flatMap((_, j) =>
      [1, 2].map((radius): XYSeries => ({
        name: names[j],
        type: 'line',
        ...ellipse(mixture, j, radius),
        slot: j,
        dashed: radius === 2,
      })),
    )
    return [
      { name: 'points', type: 'scatter', x: data.x, y: data.y, group: labels, groupNames: names },
      ...ellipses,
      {
        name: 'means',
        type: 'scatter',
        x: mixture.means.map((m) => m[0]),
        y: mixture.means.map((m) => m[1]),
        emphasis: true,
      },
    ]
  }, [data, state])

  const curve = useMemo(
    (): XYSeries[] =>
      state
        ? [
            { name: 'log-likelihood', type: 'line', x: state.history.map((_, i) => i), y: state.history },
            // Marks the iteration on display.
            { name: 'shown', type: 'scatter', x: [state.cursor], y: [state.history[state.cursor]], emphasis: true },
          ]
        : [],
    [state],
  )

  if (!data || !state) return null
  const runToEnd = () => {
    let s = step(points, state)
    for (let i = 1; i < MAX_RUN && !s.done; i++) s = step(points, s)
    setState(s)
  }
  const last = state.history.length - 1

  return (
    <Interactive
      title="Expectation-maximisation, step by step"
      caption="Each step re-estimates every component from all points, weighted by how likely each point is to belong to it, then recomputes those probabilities. Points take the colour of their most likely component. Solid ellipses are one standard deviation from the mean; dashed ones are two. The log-likelihood never decreases. With random initialisation, seed 5 converges to a worse optimum and seed 3 stalls on a plateau before escaping. k-means++ reaches the better fit for every seed from 0 to 20. Click the log-likelihood curve to go back to any iteration; Step or Run from there continues from that point."
      controls={
        <>
          <ParamSlider label="k" value={k} onChange={setK} min={1} max={6} step={1} />
          <ParamSlider label="seed" value={seed} onChange={setSeed} min={0} max={20} step={1} />
          <ParamChoice label="initialisation" value={init} onChange={setInit} options={CENTRE_INIT_OPTIONS} />
          <StepControls
            onStep={() => setState(step(points, state))}
            onRun={runToEnd}
            onReset={reset}
            done={finished(state)}
          />
        </>
      }
      readout={
        <>
          <Readout label="iteration" value={state.cursor === last ? last : `${state.cursor} of ${last}`} />
          <Readout label="mean log-likelihood" value={formatNumber(state.history[state.cursor])} />
          <Readout label="status" value={finished(state) ? 'converged' : state.cursor < last ? 'rewound' : 'running'} />
          <Readout label="weights" value={state.mixtures[state.cursor].weights.map((w) => w.toFixed(2)).join(' · ')} />
        </>
      }
    >
      <div className="grid items-center gap-4 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <XYChart equalAspect xRange={[-9, 7]} yRange={[-3, 5]} xLabel="x₁" yLabel="x₂" series={series} />
        <XYChart
          height={320}
          xLabel="iteration"
          yLabel="mean log-likelihood"
          series={curve}
          onPlotClick={([x]) => setState(seek(points, state, x))}
        />
      </div>
    </Interactive>
  )
}
