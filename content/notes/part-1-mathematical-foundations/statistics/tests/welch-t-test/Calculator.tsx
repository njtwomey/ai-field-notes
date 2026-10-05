import { useMemo } from 'react'
import { Figure, float, formatNumber, int, Readout, useFigureState } from 'aifn-render'
import { studentTCdf } from 'aifn-compute/numerics/special'

type Result = { t: number; df: number; p: number }

const twoSided = (t: number, df: number) => 2 * (1 - studentTCdf(Math.abs(t), df))

/** Both tests from summary statistics, updated live. */
export function Calculator() {
  const state = useFigureState({
    diff: float(1, { min: 0, max: 3, step: 0.05, label: 'difference in means' }),
    sd1: float(1, { min: 0.2, max: 5, step: 0.1, label: 'sd, group 1' }),
    sd2: float(3, { min: 0.2, max: 5, step: 0.1, label: 'sd, group 2' }),
    n1: int(40, { min: 2, max: 100, step: 1, label: 'n, group 1' }),
    n2: int(10, { min: 2, max: 100, step: 1, label: 'n, group 2' }),
  })

  const { pooled, welch } = useMemo(() => {
    const v1 = state.sd1 * state.sd1
    const v2 = state.sd2 * state.sd2
    const sp = ((state.n1 - 1) * v1 + (state.n2 - 1) * v2) / (state.n1 + state.n2 - 2)
    const tp = state.diff / Math.sqrt(sp * (1 / state.n1 + 1 / state.n2))
    const a = v1 / state.n1
    const b = v2 / state.n2
    const tw = state.diff / Math.sqrt(a + b)
    const dfw = (a + b) ** 2 / ((a * a) / (state.n1 - 1) + (b * b) / (state.n2 - 1))
    const pooled: Result = { t: tp, df: state.n1 + state.n2 - 2, p: twoSided(tp, state.n1 + state.n2 - 2) }
    const welch: Result = { t: tw, df: dfw, p: twoSided(tw, dfw) }
    return { pooled, welch }
  }, [state.diff, state.sd1, state.sd2, state.n1, state.n2])

  const row = (label: string, r: Result) => (
    <div className="flex flex-wrap items-baseline gap-x-6 gap-y-1 rounded-md border px-4 py-3">
      <span className="w-32 text-sm font-medium">{label}</span>
      <Readout label="t" value={formatNumber(r.t)} />
      <Readout label="df" value={formatNumber(r.df)} />
      <Readout label="p" value={formatNumber(r.p)} />
      <span className={r.p < 0.05 ? 'text-xs font-medium' : 'text-xs text-muted-foreground'}>
        {r.p < 0.05 ? 'reject at α = 0.05' : 'do not reject'}
      </span>
    </div>
  )

  return (
    <Figure
      title="Same summary statistics, two tests"
      state={state}
      caption="Enter each group's size and standard deviation and the observed difference in means. When the smaller group has the larger spread, the pooled test understates the standard error and reports a p-value that is too small."
    >
      <div className="space-y-2">
        {row("Pooled (Student's)", pooled)}
        {row("Welch's", welch)}
      </div>
    </Figure>
  )
}
