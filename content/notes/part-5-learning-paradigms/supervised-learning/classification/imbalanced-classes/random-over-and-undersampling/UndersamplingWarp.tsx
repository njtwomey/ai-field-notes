import { useMemo } from 'react'
import {
  Annotation,
  Curve,
  Figure,
  Handle,
  Plot,
  Points,
  Readout,
  formatNumber,
  setting,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { expit, logit } from '../_shared/binormal'

const P = toFlat(linspace(0.0005, 0.9995, 400))
const Z = toFlat(linspace(-8, 8, 161))

/** Posterior after keeping each negative with probability β: p' = p / (p + β(1 − p)). */
const warp = (p: number, beta: number) => p / (p + beta * (1 - p))
/** The inverse map, which recovers the original posterior. */
const unwarp = (q: number, beta: number) => (beta * q) / (beta * q - q + 1)

/**
 * The map from the true posterior p to the posterior p' learnt after random undersampling of the majority class, on
 * probability axes or on log-odds axes, where it is a straight line shifted up by −log β.
 */
export function UndersamplingWarp() {
  const state = useFigureState({
    beta: slider(0.01, 1, 0.1, { step: 0.01, label: 'fraction of negatives kept β' }),
    p: slider(0.001, 0.999, 0.2, { step: 0.001, label: 'true posterior p' }),
    logOdds: setting(false, 'log-odds axes'),
  })
  const { beta, p, logOdds } = state

  const q = warp(p, beta)
  const pEquivalent = beta / (1 + beta)

  const curve = useMemo(
    () => (logOdds ? { x: Z, y: Z.map((z) => z - Math.log(beta)) } : { x: P, y: P.map((v) => warp(v, beta)) }),
    [logOdds, beta],
  )
  const identity = logOdds ? [-8, 8] : [0, 1]

  const mx = logOdds ? logit(p) : p
  const my = logOdds ? logit(q) : q

  const xAxis = useAxis({ label: logOdds ? 'logit p' : 'true posterior p', range: logOdds ? [-8, 8] : [0, 1] })
  const yAxis = useAxis({
    label: logOdds ? "logit p'" : "posterior after undersampling p'",
    range: logOdds ? [-8, 13] : [0, 1],
  })

  return (
    <Figure
      title="Undersampling warps the posterior"
      state={state}
      caption="Keep every positive and each negative with probability β. A model fitted to the undersampled data learns p' instead of the true posterior p. Drag the vertical line to pick an example's true posterior. On probability axes the warp is a steep curve for small β; on log-odds axes it is the identity shifted up by −log β, which is why a single subtraction undoes it."
      readouts={
        <>
          <Readout label="p" value={formatNumber(p)} />
          <Readout label="p' after undersampling" value={formatNumber(q)} />
          <Readout label="corrected β p' / (β p' − p' + 1)" value={formatNumber(unwarp(q, beta))} />
          <Readout label="log-odds shift −log β" value={formatNumber(-Math.log(beta))} />
          <Readout label="p' = 0.5 is p =" value={formatNumber(pEquivalent)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={320}>
        <Curve name="no resampling (β = 1)" x={identity} y={identity} dashed muted />
        <Curve name="after undersampling" x={curve.x} y={curve.y} slot={0} />
        {!logOdds && <Annotation y={0.5} text="p' = 0.5" dashed />}
        <Points name="example" x={[mx]} y={[my]} emphasis />
        <Handle
          kind="x"
          at={mx}
          label="true posterior"
          onDrag={(x) => state.set('p', Math.min(Math.max(logOdds ? expit(x) : x, 0.001), 0.999))}
        />
      </Plot>
    </Figure>
  )
}
