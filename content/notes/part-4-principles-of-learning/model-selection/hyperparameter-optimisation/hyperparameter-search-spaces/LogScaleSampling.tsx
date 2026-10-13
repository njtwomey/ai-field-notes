import { useMemo } from 'react'
import { Curve, Figure, formatNumber, int, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { child, stream } from 'aifn-compute/foundation/random'
import { real, sample, space } from 'aifn-compute/foundation/space'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { svmError } from '../_shared/svmSurface'

const LOG_SPACE = space({ C: real(0.01, 1000, { scale: 'log' }) })
const LINEAR_SPACE = space({ C: real(0.01, 1000) })
const LOG_C = toFlat(linspace(-2, 3, 201))

/**
 * The same number of random trials of the SVM penalty C, drawn uniformly on a log scale and on a linear scale, over
 * the cross-validated error at one kernel width γ.
 */
export function LogScaleSampling() {
  const state = useFigureState({
    trials: int(20, { ge: 1, le: 200, suggestions: [5, 10, 20, 50, 100], label: 'trials N' }),
    logGamma: slider(-3, 2, -2, { step: 0.25, label: 'kernel width log₁₀ γ' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
  })
  const { trials, logGamma, seed } = state

  const curve = useMemo(
    () => ({ x: LOG_C.map((v) => 10 ** v), y: LOG_C.map((v) => svmError(v, logGamma)) }),
    [logGamma],
  )
  const draws = useMemo(() => {
    const s = stream(`log-scale-sampling/${seed}`)
    const draw = (name: string, sp: typeof LOG_SPACE | typeof LINEAR_SPACE) =>
      Array.from({ length: trials }, (_, k) => sample(child(s, name, k), sp).C)
    return { log: draw('log', LOG_SPACE), linear: draw('linear', LINEAR_SPACE) }
  }, [trials, seed])

  const errors = (cs: number[]) => cs.map((c) => svmError(Math.log10(c), logGamma))
  const logErrors = errors(draws.log)
  const linearErrors = errors(draws.linear)
  const below = (cs: number[]) => cs.filter((c) => c < 10).length

  const xAxis = useAxis({ label: 'penalty C', log: true, range: [0.01, 1000] })
  const yAxis = useAxis({ label: 'cross-validated error' })
  return (
    <Figure
      title="Random trials of C on a log scale and on a linear scale"
      state={state}
      caption="The cross-validated error of an RBF support vector machine against its penalty C, at the kernel width set by the slider, on a log axis. Orange points: N trials with log C drawn uniformly, so each of the five decades gets a fifth of them. Green points: N trials with C drawn uniformly on [0.01, 1000], which puts 90% of them in the top decade. At the default width the lowest error is at C = 10, and the linear-scale trials rarely reach below it. Move the width to shift the optimum, and change the seed for a new draw."
      readouts={
        <>
          <Readout label="trials with C < 10, log scale" value={String(below(draws.log))} />
          <Readout label="trials with C < 10, linear scale" value={String(below(draws.linear))} />
          <Readout label="best error, log scale" value={formatNumber(Math.min(...logErrors))} />
          <Readout label="best error, linear scale" value={formatNumber(Math.min(...linearErrors))} />
          <Readout label="lowest error at this γ" value={formatNumber(Math.min(...curve.y))} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        <Curve name="error at this γ" x={curve.x} y={curve.y} muted />
        <Points name="log-uniform trials" x={draws.log} y={logErrors} slot={1} />
        <Points name="uniform trials" x={draws.linear} y={linearErrors} slot={2} />
      </Plot>
    </Figure>
  )
}
