import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamSlider,
  ParamSwitch,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'
import { linspace } from '@/lib/math'
import { expit, logit } from '../_shared/binormal'

const P = linspace(0.0005, 0.9995, 400)
const Z = linspace(-8, 8, 161)

/** Posterior after keeping each negative with probability β: p' = p / (p + β(1 − p)). */
const warp = (p: number, beta: number) => p / (p + beta * (1 - p))
/** The inverse map, which recovers the original posterior. */
const unwarp = (q: number, beta: number) => (beta * q) / (beta * q - q + 1)

/**
 * The map from the true posterior p to the posterior p' learnt after random undersampling of the majority class, on
 * probability axes or on log-odds axes, where it is a straight line shifted up by −log β.
 */
export function UndersamplingWarp() {
  const beta = useParam(0.1, { min: 0.01, max: 1, step: 0.01 })
  const p = useParam(0.2, { min: 0.001, max: 0.999, step: 0.001 })
  const [logOdds, setLogOdds] = useState(false)

  const q = warp(p.value, beta.value)
  const pEquivalent = beta.value / (1 + beta.value)

  const series = useMemo<XYSeries[]>(() => {
    if (logOdds) {
      return [
        { name: 'no resampling (β = 1)', type: 'line', x: [-8, 8], y: [-8, 8], dashed: true, muted: true },
        { name: 'after undersampling', type: 'line', x: Z, y: Z.map((z) => z - Math.log(beta.value)), slot: 0 },
      ]
    }
    return [
      { name: 'no resampling (β = 1)', type: 'line', x: [0, 1], y: [0, 1], dashed: true, muted: true },
      { name: 'after undersampling', type: 'line', x: P, y: P.map((v) => warp(v, beta.value)), slot: 0 },
      { name: "p' = 0.5", type: 'line', x: [0, 1], y: [0.5, 0.5], dashed: true, slot: 2 },
    ]
  }, [logOdds, beta.value])

  const mx = logOdds ? logit(p.value) : p.value
  const my = logOdds ? logit(q) : q
  const all = useMemo<XYSeries[]>(
    () => [...series, { name: 'example', type: 'scatter', x: [mx], y: [my], emphasis: true }],
    [series, mx, my],
  )

  const handles: Handle[] = [
    {
      kind: 'x',
      at: mx,
      label: 'true posterior',
      onDrag: (x) => p.set(logOdds ? expit(x) : x),
    },
  ]

  return (
    <Interactive
      title="Undersampling warps the posterior"
      caption="Keep every positive and each negative with probability β. A model fitted to the undersampled data learns p' instead of the true posterior p. Drag the vertical line to pick an example's true posterior. On probability axes the warp is a steep curve for small β; on log-odds axes it is the identity shifted up by −log β, which is why a single subtraction undoes it."
      controls={
        <>
          <ParamSlider label="fraction of negatives kept β" param={beta} />
          <ParamSlider label="true posterior p" param={p} />
          <ParamSwitch label="log-odds axes" checked={logOdds} onChange={setLogOdds} />
        </>
      }
      readout={
        <>
          <Readout label="p" value={formatNumber(p.value)} />
          <Readout label="p' after undersampling" value={formatNumber(q)} />
          <Readout label="corrected β p' / (β p' − p' + 1)" value={formatNumber(unwarp(q, beta.value))} />
          <Readout label="log-odds shift −log β" value={formatNumber(-Math.log(beta.value))} />
          <Readout label="p' = 0.5 is p =" value={formatNumber(pEquivalent)} />
        </>
      }
    >
      <XYChart
        height={320}
        xLabel={logOdds ? 'logit p' : 'true posterior p'}
        yLabel={logOdds ? "logit p'" : "posterior after undersampling p'"}
        xRange={logOdds ? [-8, 8] : [0, 1]}
        yRange={logOdds ? [-8, 13] : [0, 1]}
        handles={handles}
        series={all}
      />
    </Interactive>
  )
}
