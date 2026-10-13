import { useMemo } from 'react'
import { Annotation, Bars, Figure, formatNumber, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { mcnemarTest } from 'aifn-compute/probability/tests'

const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—')

/**
 * McNemar's test from its two discordant counts: the binomial null law of b given b + c, the outcomes at least as
 * extreme as the observed one, and the exact and χ² p-values.
 */
export function McnemarNull() {
  const state = useFigureState({
    b: int(14, { ge: 0, le: 80, suggestions: [5, 14, 30], label: 'b: only A correct' }),
    c: int(5, { ge: 0, le: 80, suggestions: [5, 10, 20], label: 'c: only B correct' }),
  })
  const { b, c } = state
  const view = useMemo(() => {
    const m = b + c
    const table = [
      [0, b],
      [c, 0],
    ]
    const exact = mcnemarTest(table)
    if (m === 0) return { m, exact, corrected: NaN, plain: NaN, ks: [], probs: [], extreme: [] as boolean[] }
    const corrected = mcnemarTest(table, { exact: false }).pValue
    const plain = mcnemarTest(table, { exact: false, correction: false }).pValue
    const ks = Array.from({ length: m + 1 }, (_, k) => k)
    const probs = ks.map((k) => exact.null.prob(k) as number)
    const observed = probs[b]
    const extreme = probs.map((p) => p <= observed * (1 + 1e-9))
    return { m, exact, corrected, plain, ks, probs, extreme }
  }, [b, c])
  const tail = view.ks.filter((k) => view.extreme[k])
  const body = view.ks.filter((k) => !view.extreme[k])
  const x = useAxis({
    label: 'b, the discordant cases A gets right',
    integer: true,
    range: [-0.5, Math.max(view.m, 1) + 0.5],
    nice: false,
  })
  const y = useAxis({ label: 'probability under H₀', range: [0, undefined] })
  return (
    <Figure
      title="McNemar's test reads b against a fair coin"
      state={state}
      caption="Only the b + c cases on which the classifiers disagree carry information. Under the null hypothesis of equal accuracy each disagreement is a fair coin, so b follows Binomial(b + c, 1/2). The highlighted bars are the outcomes no more probable than the observed b; their total is the exact two-sided p-value. The χ² form approximates the same tail with (|b − c| − 1)²/(b + c) on one degree of freedom. The number of cases both classifiers get right or both get wrong does not enter."
      readouts={{
        'p-value': (
          <>
            <Readout label="exact binomial" value={fmt(view.exact.pValue)} />
            <Readout label="χ², continuity-corrected" value={fmt(view.corrected)} />
            <Readout label="χ², uncorrected" value={fmt(view.plain)} />
          </>
        ),
        estimate: (
          <>
            <Readout label="b / (b + c)" value={fmt(view.m > 0 ? b / view.m : NaN)} />
            <Readout label="discordant cases b + c" value={String(view.m)} />
          </>
        ),
      }}
    >
      <Plot x={x} y={y}>
        {body.length > 0 && (
          <Bars name="less extreme than b" x={body} y={body.map((k) => view.probs[k])} muted width={0.8} />
        )}
        {tail.length > 0 && (
          <Bars name="as or more extreme (p-value)" x={tail} y={tail.map((k) => view.probs[k])} slot={1} width={0.8} />
        )}
        {view.m > 0 && <Annotation x={b} text="observed b" emphasis />}
      </Plot>
    </Figure>
  )
}
