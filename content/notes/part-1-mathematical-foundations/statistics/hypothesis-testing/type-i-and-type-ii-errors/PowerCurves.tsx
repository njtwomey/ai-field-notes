import { useMemo } from 'react'
import {
  Area,
  choice,
  Curve,
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { normalCdf, normalPdf, normalQuantile } from 'aifn-compute/numerics/special'

type Sides = 'two' | 'one'

/**
 * The z statistic under H₀ (centred at 0) and under H₁ (centred at δ√n/σ). Shaded: α, the rejection region under H₀,
 * and β, the acceptance region under H₁.
 */
export function PowerCurves() {
  const state = useFigureState({
    effect: slider(0, 1.5, 0.4, { step: 0.01, label: 'effect δ (standard deviations)' }),
    n: int(25, { min: 2, max: 200, step: 1, suggestions: [10, 25, 50, 100, 200], label: 'sample size n' }),
    alpha: slider(0.001, 0.2, 0.05, { step: 0.001, label: 'significance level α' }),
    sides: choice<Sides>(
      [
        { value: 'two', label: 'two-sided' },
        { value: 'one', label: 'one-sided' },
      ],
      'two',
      { label: 'test' },
    ),
  })
  const { effect, n, alpha, sides } = state

  const result = useMemo(() => {
    const shift = effect * Math.sqrt(n)
    const crit = normalQuantile(1 - (sides === 'two' ? alpha / 2 : alpha))
    const lo = Math.min(-4, shift - 4)
    const hi = Math.max(4, shift + 4)
    const xs = toFlat(linspace(lo, hi, 400))
    const h0 = (x: number) => normalPdf(x)
    const h1 = (x: number) => normalPdf(x - shift)
    const region = (f: (x: number) => number, keep: (x: number) => boolean) => {
      const inside = xs.filter(keep)
      return { x: inside, y: inside.map(f) }
    }
    const reject = (x: number) => (sides === 'two' ? Math.abs(x) >= crit : x >= crit)
    const power = sides === 'two' ? 1 - normalCdf(crit - shift) + normalCdf(-crit - shift) : 1 - normalCdf(crit - shift)
    // n giving 80% power for this effect: n = ((z_crit + z_0.8) / δ)².
    const needed = effect > 0 ? Math.ceil(((crit + normalQuantile(0.8)) / effect) ** 2) : Infinity
    const tails =
      sides === 'two' ? [region(h0, (x) => x <= -crit), region(h0, (x) => x >= crit)] : [region(h0, (x) => x >= crit)]
    return {
      h0: { x: xs, y: xs.map(h0) },
      h1: { x: xs, y: xs.map(h1) },
      tails,
      missed: region(h1, (x) => !reject(x)),
      power,
      crit,
      needed,
      shift,
    }
  }, [effect, n, alpha, sides])

  const xAxis = useAxis({ label: 'z', hold: 'union' })
  const yAxis = useAxis({ label: 'density', range: [0, 0.45] })
  return (
    <Figure
      title="α, β and power"
      state={state}
      caption="The blue curve is the test statistic when there is no effect; the orange curve is the same statistic when the effect is real. Shaded blue: rejections that are false (α). Shaded orange: real effects the test misses (β). Power is 1 − β. Raise n or the effect: the curves separate and β shrinks. Lower α: the cut-off moves out and β grows. Drag the dashed lines to move the cut-off, which sets α, or the mean under H₁, which sets the effect."
      readouts={
        <>
          <Readout label="critical value" value={formatNumber(result.crit)} />
          <Readout label="power 1 − β" value={`${(100 * result.power).toFixed(1)}%`} />
          <Readout label="β" value={`${(100 * (1 - result.power)).toFixed(1)}%`} />
          <Readout label="n for 80% power" value={Number.isFinite(result.needed) ? result.needed : '—'} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        <Curve name="Z under H₀" {...result.h0} slot={0} />
        <Curve name="Z under H₁" {...result.h1} slot={1} />
        {result.tails.map((t, i) => (
          <Area key={i} name="α: false rejection" {...t} slot={0} />
        ))}
        <Area name="β: missed effect" {...result.missed} slot={1} />
        {/* The cut-off maps back to α through the tail area beyond it; the H₁ mean δ√n maps back to the effect δ. */}
        <Handle
          kind="x"
          at={result.crit}
          label="critical value"
          onDrag={(c) => state.set('alpha', sides === 'two' ? 2 * (1 - normalCdf(Math.abs(c))) : 1 - normalCdf(c))}
        />
        <Handle kind="x" at={result.shift} label="H₁ mean" onDrag={(m) => state.set('effect', m / Math.sqrt(n))} />
      </Plot>
    </Figure>
  )
}
