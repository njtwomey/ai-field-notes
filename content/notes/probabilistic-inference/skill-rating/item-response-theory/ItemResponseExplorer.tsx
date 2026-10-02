import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  type Handle,
  type XYSeries,
} from 'aifn-render'
import { linspace } from '@/lib/math'

type Form = '1PL' | '2PL' | '3PL' | '4PL'
type Item = { a: number; b: number; c: number; d: number }
type Answer = 'right' | 'wrong' | 'none'

const FORMS: Form[] = ['1PL', '2PL', '3PL', '4PL']
const START: Item[] = [
  { a: 1.0, b: -1.0, c: 0.2, d: 0.95 },
  { a: 1.5, b: 0.0, c: 0.2, d: 0.95 },
  { a: 2.0, b: 1.0, c: 0.2, d: 0.95 },
]
const THETA = linspace(-4, 4, 321)
const X_RANGE: [number, number] = [-4, 4]
const PRIOR_SD = 1

/** The parameters a form uses; the others take their defaults a = 1, c = 0, d = 1. */
function effective(item: Item, form: Form): Item {
  const k = FORMS.indexOf(form)
  return { a: k >= 1 ? item.a : 1, b: item.b, c: k >= 2 ? item.c : 0, d: k >= 3 ? item.d : 1 }
}

const logistic = (z: number) => 1 / (1 + Math.exp(-z))
const prob = ({ a, b, c, d }: Item, t: number) => c + (d - c) * logistic(a * (t - b))
/** Fisher information of one response: P'(θ)² / (P (1 − P)). */
function info({ a, b, c, d }: Item, t: number): number {
  const s = logistic(a * (t - b))
  const p = c + (d - c) * s
  const dp = a * (d - c) * s * (1 - s)
  return (dp * dp) / (p * (1 - p))
}

/** Posterior of θ on the grid under a N(0, 1) prior, with EAP, posterior sd, and the MLE with its standard error. */
function posterior(items: Item[], answers: Answer[]) {
  const logLik = THETA.map((t) =>
    items.reduce((acc, it, i) => {
      if (answers[i] === 'none') return acc
      const p = prob(it, t)
      return acc + Math.log(answers[i] === 'right' ? p : 1 - p)
    }, 0),
  )
  const logPost = logLik.map((l, k) => l - (THETA[k] * THETA[k]) / (2 * PRIOR_SD * PRIOR_SD))
  const top = Math.max(...logPost)
  const w = logPost.map((l) => Math.exp(l - top))
  const step = THETA[1] - THETA[0]
  const z = w.reduce((a, b) => a + b, 0) * step
  const dens = w.map((v) => v / z)
  const eap = dens.reduce((acc, v, k) => acc + v * THETA[k] * step, 0)
  const psd = Math.sqrt(dens.reduce((acc, v, k) => acc + v * (THETA[k] - eap) ** 2 * step, 0))
  // The MLE is interior only when the answers are mixed; at the grid edge it is reported as unbounded.
  let best = 0
  logLik.forEach((l, k) => {
    if (l > logLik[best]) best = k
  })
  const answered = answers.filter((x) => x !== 'none')
  const mixed = answered.includes('right') && answered.includes('wrong')
  let mle = mixed && best > 0 && best < THETA.length - 1 ? THETA[best] : null
  const testInfoAt = (t: number) => items.reduce((acc, it, i) => (answers[i] === 'none' ? acc : acc + info(it, t)), 0)
  if (mle !== null) {
    // Polish the grid maximum with Fisher scoring: θ ← θ + score / information.
    for (let n = 0; n < 20; n++) {
      const t: number = mle
      const score = items.reduce((acc, it, i) => {
        if (answers[i] === 'none') return acc
        const s = logistic(it.a * (t - it.b))
        const pr = it.c + (it.d - it.c) * s
        const dp = it.a * (it.d - it.c) * s * (1 - s)
        return acc + (((answers[i] === 'right' ? 1 : 0) - pr) * dp) / (pr * (1 - pr))
      }, 0)
      mle = t + score / testInfoAt(t)
    }
  }
  const testInfo = mle === null ? 0 : testInfoAt(mle)
  return { dens, eap, psd, mle, se: mle === null ? null : 1 / Math.sqrt(testInfo) }
}

export function ItemResponseExplorer() {
  const [form, setForm] = useState<Form>('3PL')
  const [items, setItems] = useState<Item[]>(START)
  const [sel, setSel] = useState<'0' | '1' | '2'>('1')
  const [answers, setAnswers] = useState<Answer[]>(['right', 'right', 'wrong'])
  const i = Number(sel)
  const k = FORMS.indexOf(form)

  const setParam = (key: keyof Item, v: number) =>
    setItems((prev) => prev.map((it, j) => (j === i ? { ...it, [key]: v } : it)))
  const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))

  const eff = useMemo(() => items.map((it) => effective(it, form)), [items, form])

  const icc = useMemo<XYSeries[]>(
    () =>
      eff.map((it, j) => ({
        name: `item ${j + 1}`,
        type: 'line',
        x: THETA,
        y: THETA.map((t) => prob(it, t)),
        slot: j,
      })),
    [eff],
  )
  const infoSeries = useMemo<XYSeries[]>(() => {
    const per = eff.map((it) => THETA.map((t) => info(it, t)))
    return [
      ...per.map<XYSeries>((y, j) => ({ name: `item ${j + 1}`, type: 'line', x: THETA, y, slot: j })),
      {
        name: 'test',
        type: 'line',
        x: THETA,
        y: THETA.map((_, q) => per.reduce((a, p) => a + p[q], 0)),
        emphasis: true,
      },
    ]
  }, [eff])

  const posts = useMemo(
    () =>
      FORMS.map((f) =>
        posterior(
          items.map((it) => effective(it, f)),
          answers,
        ),
      ),
    [items, answers],
  )
  const postSeries = useMemo<XYSeries[]>(
    () =>
      FORMS.map<XYSeries>((f, q) =>
        f === form
          ? { name: `${f} (shown)`, type: 'line', x: THETA, y: posts[q].dens, emphasis: true }
          : { name: f, type: 'line', x: THETA, y: posts[q].dens, muted: true, dashed: true },
      ),
    [posts, form],
  )

  const cur = items[i]
  const handles: Handle[] = [
    { kind: 'x', at: cur.b, label: 'b', onDrag: (x) => setParam('b', Math.round(clamp(x, -3, 3) * 20) / 20) },
    ...(k >= 2
      ? [
          {
            kind: 'y',
            at: cur.c,
            label: 'c',
            onDrag: (y: number) => setParam('c', Math.round(clamp(y, 0, 0.4) * 100) / 100),
          } as Handle,
        ]
      : []),
    ...(k >= 3
      ? [
          {
            kind: 'y',
            at: cur.d,
            label: 'd',
            onDrag: (y: number) => setParam('d', Math.round(clamp(y, 0.6, 1) * 100) / 100),
          } as Handle,
        ]
      : []),
  ]
  const p = posts[k]
  const est = (v: number | null, s: number | null) =>
    v === null ? 'unbounded' : `${formatNumber(v)} ± ${formatNumber(s ?? 0)}`

  return (
    <Interactive
      title="Item characteristic curves, information and a three-item test"
      caption="Pick a form and an item. On the top chart, drag the vertical line to move the item's difficulty b, and in 3PL and 4PL drag the horizontal lines to set the guessing floor c and the ceiling d; the discrimination a is a slider. Forms that drop a parameter set it back to its default (a = 1, c = 0, d = 1). The middle chart shows each item's information and the test information: guessing and slipping lower the peak, and the 3PL peak moves above b. The bottom chart is the posterior of ability for the answers chosen, with a standard normal prior, under the chosen form (solid) and the other forms (dashed). The default items and answers are those of the worked 2PL example in the note on ability estimation: in 2PL the EAP is 0.312."
      controls={
        <>
          <ParamChoice
            label="form"
            value={form}
            onChange={setForm}
            options={FORMS.map((f) => ({ value: f, label: f }))}
          />
          <ParamChoice
            label="item to edit"
            value={sel}
            onChange={setSel}
            options={[
              { value: '0', label: 'item 1' },
              { value: '1', label: 'item 2' },
              { value: '2', label: 'item 3' },
            ]}
          />
          <ParamSlider
            label="difficulty b"
            value={cur.b}
            onChange={(v) => setParam('b', v)}
            min={-3}
            max={3}
            step={0.05}
          />
          {k >= 1 && (
            <ParamSlider
              label="discrimination a"
              value={cur.a}
              onChange={(v) => setParam('a', v)}
              min={0.2}
              max={3}
              step={0.05}
            />
          )}
          {k >= 2 && (
            <ParamSlider
              label="guessing floor c"
              value={cur.c}
              onChange={(v) => setParam('c', v)}
              min={0}
              max={0.4}
              step={0.01}
            />
          )}
          {k >= 3 && (
            <ParamSlider
              label="ceiling d"
              value={cur.d}
              onChange={(v) => setParam('d', v)}
              min={0.6}
              max={1}
              step={0.01}
            />
          )}
          {answers.map((ans, j) => (
            <ParamChoice
              key={j}
              label={`answer to item ${j + 1}`}
              value={ans}
              onChange={(v) => setAnswers((prev) => prev.map((x, q) => (q === j ? v : x)))}
              options={[
                { value: 'right', label: 'right' },
                { value: 'wrong', label: 'wrong' },
                { value: 'none', label: 'skip' },
              ]}
            />
          ))}
        </>
      }
      readout={
        <>
          <Readout label={`${form} MLE ± SE`} value={est(p.mle, p.se)} />
          <Readout label={`${form} EAP ± posterior sd`} value={est(p.eap, p.psd)} />
          {FORMS.filter((f) => f !== form).map((f) => (
            <Readout key={f} label={`${f} EAP`} value={formatNumber(posts[FORMS.indexOf(f)].eap)} />
          ))}
        </>
      }
    >
      <div className="space-y-2">
        <XYChart
          series={icc}
          xLabel="ability θ"
          yLabel="P(correct)"
          xRange={X_RANGE}
          yRange={[0, 1]}
          height={260}
          handles={handles}
        />
        <XYChart
          series={infoSeries}
          xLabel="ability θ"
          yLabel="information"
          xRange={X_RANGE}
          yRange={[0, undefined]}
          height={200}
        />
        <XYChart series={postSeries} xLabel="ability θ" yLabel="posterior density" xRange={X_RANGE} height={200} />
      </div>
    </Interactive>
  )
}
