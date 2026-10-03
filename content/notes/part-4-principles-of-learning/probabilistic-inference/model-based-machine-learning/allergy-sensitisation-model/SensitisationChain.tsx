import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from 'aifn-render'
import { posterior, type ChainParams } from '../_shared/sensitisation'

const AGES = [1, 3, 5, 8]
/** Illustrative classes, not fitted values: parameters of the sensitisation chain for each kind of child. */
const CLASSES: { name: string; params: ChainParams; prior: number }[] = [
  { name: 'rarely sensitised', params: { initial: 0.02, gain: 0.03, retain: 0.5 }, prior: 0.7 },
  { name: 'early and persistent', params: { initial: 0.4, gain: 0.4, retain: 0.95 }, prior: 0.1 },
  { name: 'later onset', params: { initial: 0.02, gain: 0.25, retain: 0.9 }, prior: 0.2 },
]
type Result = 'pos' | 'neg' | 'none'
const OPTIONS = [
  { value: 'pos' as const, label: '+' },
  { value: 'neg' as const, label: '−' },
  { value: 'none' as const, label: 'not done' },
]
const toTest = (r: Result) => (r === 'none' ? null : r === 'pos')

/** Toggle skin-test results at four ages; see sensitisation over time and the posterior over classes. */
export function SensitisationChain() {
  const [results, setResults] = useState<Result[]>(['neg', 'neg', 'pos', 'pos'])
  const [cls, setCls] = useState('unknown')
  const sensitivity = useParam(0.8, { min: 0.5, max: 1, step: 0.01 })
  const falsePositive = useParam(0.05, { min: 0, max: 0.3, step: 0.01 })

  const key = results.join()
  const { post, byClass } = useMemo(() => {
    const tests = key.split(',').map((r) => toTest(r as Result))
    const tp = { sensitivity: sensitivity.value, falsePositive: falsePositive.value }
    const known = cls === 'unknown' ? null : Number(cls)
    const classes = known === null ? CLASSES : [CLASSES[known]]
    const post = posterior(
      classes.map((c) => c.params),
      known === null ? CLASSES.map((c) => c.prior) : [1],
      tests,
      tp,
    )
    const byClass = posterior(
      CLASSES.map((c) => c.params),
      CLASSES.map((c) => c.prior),
      tests,
      tp,
    ).classPosterior
    return { post, byClass }
  }, [key, cls, sensitivity.value, falsePositive.value])

  const series = useMemo((): XYSeries[] => {
    const observed = key.split(',').map((r) => (r === 'none' ? null : r === 'pos' ? 1 : 0))
    const idx = observed.flatMap((o, i) => (o === null ? [] : [i]))
    return [
      { name: 'P(sensitised)', type: 'line', x: AGES, y: post.sensitised, slot: 0 },
      { name: 'P(sensitised) at test ages', type: 'scatter', x: AGES, y: post.sensitised, slot: 0 },
      {
        name: 'test result (1 = positive)',
        type: 'scatter',
        x: idx.map((i) => AGES[i]),
        y: idx.map((i) => observed[i] as number),
        emphasis: true,
      },
    ]
  }, [post, key])

  return (
    <Interactive
      title="Sensitisation over time, seen through a noisy test"
      caption="One child and one allergen. Sensitisation follows a two-state Markov chain from age 1 to age 8; a skin-prick test at each age is positive with probability equal to its sensitivity if the child is sensitised, and with the false-positive rate if not. The class, a gate, chooses the chain's initial, gain and retain probabilities. Choose 'unknown' to average over classes and read off the posterior class probabilities. The class parameters here are illustrative, not the study's."
      controls={
        <>
          {AGES.map((a, i) => (
            <ParamChoice
              key={a}
              label={`skin test at age ${a}`}
              value={results[i]}
              onChange={(v) => setResults((old) => old.map((r, j) => (j === i ? v : r)))}
              options={OPTIONS}
            />
          ))}
          <ParamChoice
            label="class"
            value={cls}
            onChange={setCls}
            options={[
              { value: 'unknown', label: 'unknown' },
              ...CLASSES.map((c, i) => ({ value: String(i), label: c.name })),
            ]}
          />
          <ParamSlider label="test sensitivity" param={sensitivity} format={(v) => v.toFixed(2)} />
          <ParamSlider label="false-positive rate" param={falsePositive} format={(v) => v.toFixed(2)} />
        </>
      }
      readout={
        <>
          {CLASSES.map((c, i) => (
            <Readout key={c.name} label={`P(${c.name})`} value={formatNumber(byClass[i])} />
          ))}
        </>
      }
    >
      <XYChart series={series} xLabel="age (years)" yLabel="probability" xRange={[0.5, 8.5]} yRange={[0, 1]} />
    </Interactive>
  )
}
