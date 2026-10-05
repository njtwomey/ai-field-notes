import { useMemo } from 'react'
import {
  choice,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Readout,
  row,
  seriesLayers,
  slider,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'

type Form = '1PL' | '2PL' | '3PL' | '4PL'
type Item = { a: number; b: number; c: number; d: number }
type Answer = 'right' | 'wrong' | 'none'

const FORMS: Form[] = ['1PL', '2PL', '3PL', '4PL']
const START: Item[] = [
  { a: 1.0, b: -1.0, c: 0.2, d: 0.95 },
  { a: 1.5, b: 0.0, c: 0.2, d: 0.95 },
  { a: 2.0, b: 1.0, c: 0.2, d: 0.95 },
]
const ANSWERS = [
  { value: 'right' as const, label: 'right' },
  { value: 'wrong' as const, label: 'wrong' },
  { value: 'none' as const, label: 'skip' },
]
type Shown = { sel: string; form: string }
/** One item's parameters as fields, shown while the item is selected and its form uses them. */
const itemFields = <const J extends '1' | '2' | '3'>(j: J, it: Item) => {
  const sel = String(Number(j) - 1)
  const uses = (from: number) => (v: Readonly<Record<string, unknown>>) =>
    (v as Shown).sel === sel && FORMS.indexOf((v as Shown).form as Form) >= from
  return {
    [`b${j}`]: slider(-3, 3, it.b, { step: 0.05, label: 'difficulty b', when: uses(0) }),
    [`a${j}`]: slider(0.2, 3, it.a, { step: 0.05, label: 'discrimination a', when: uses(1) }),
    [`c${j}`]: slider(0, 0.4, it.c, { step: 0.01, label: 'guessing floor c', when: uses(2) }),
    [`d${j}`]: slider(0.6, 1, it.d, { step: 0.01, label: 'ceiling d', when: uses(3) }),
  } as Record<`${'a' | 'b' | 'c' | 'd'}${J}`, ReturnType<typeof slider>>
}
const THETA = toFlat(linspace(-4, 4, 321))
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
  const state = useFigureState({
    form: choice<Form>(
      FORMS.map((f) => ({ value: f, label: f })),
      '3PL',
      { label: 'form' },
    ),
    sel: choice<'0' | '1' | '2'>(
      [
        { value: '0', label: 'item 1' },
        { value: '1', label: 'item 2' },
        { value: '2', label: 'item 3' },
      ],
      '1',
      { label: 'item to edit' },
    ),
    ...itemFields('1', START[0]),
    ...itemFields('2', START[1]),
    ...itemFields('3', START[2]),
    answers: row('answers to items 1, 2, 3', {
      ans1: choice(ANSWERS, 'right', { label: 'item 1' }),
      ans2: choice(ANSWERS, 'right', { label: 'item 2' }),
      ans3: choice(ANSWERS, 'wrong', { label: 'item 3' }),
    }),
  })
  const v = state.values
  const items = useMemo(
    (): Item[] => [
      { a: v.a1, b: v.b1, c: v.c1, d: v.d1 },
      { a: v.a2, b: v.b2, c: v.c2, d: v.d2 },
      { a: v.a3, b: v.b3, c: v.c3, d: v.d3 },
    ],
    [v.a1, v.b1, v.c1, v.d1, v.a2, v.b2, v.c2, v.d2, v.a3, v.b3, v.c3, v.d3],
  )
  const { ans1, ans2, ans3 } = state.answers
  const answers = useMemo((): Answer[] => [ans1, ans2, ans3], [ans1, ans2, ans3])
  const i = Number(state.sel)
  const k = FORMS.indexOf(state.form)

  const setParam = (key: keyof Item, value: number) => state.set(`${key}${i + 1}`, value)
  const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))

  const eff = useMemo(() => items.map((it) => effective(it, state.form)), [items, state.form])

  const icc = useMemo<SeriesSpec[]>(
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
  const infoSeries = useMemo<SeriesSpec[]>(() => {
    const per = eff.map((it) => THETA.map((t) => info(it, t)))
    return [
      ...per.map<SeriesSpec>((y, j) => ({ name: `item ${j + 1}`, type: 'line', x: THETA, y, slot: j })),
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
  const postSeries = useMemo<SeriesSpec[]>(
    () =>
      FORMS.map<SeriesSpec>((f, q) =>
        f === state.form
          ? { name: `${f} (shown)`, type: 'line', x: THETA, y: posts[q].dens, emphasis: true }
          : { name: f, type: 'line', x: THETA, y: posts[q].dens, muted: true, dashed: true },
      ),
    [posts, state.form],
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

  const xAxis = useAxis({ label: 'ability θ', range: X_RANGE })
  const yAxis = useAxis({ label: 'P(correct)', range: [0, 1] })
  const xAxis2 = useAxis({ label: 'ability θ', range: X_RANGE })
  const yAxis2 = useAxis({ label: 'information', range: [0, undefined], hold: 'union' })
  const xAxis3 = useAxis({ label: 'ability θ', range: X_RANGE })
  const yAxis3 = useAxis({ label: 'posterior density', hold: 'union' })
  return (
    <Figure
      title="Item characteristic curves, information and a three-item test"
      state={state}
      caption="Pick a form and an item. On the top chart, drag the vertical line to move the item's difficulty b, and in 3PL and 4PL drag the horizontal lines to set the guessing floor c and the ceiling d; the discrimination a is a slider. Forms that drop a parameter set it back to its default (a = 1, c = 0, d = 1). The middle chart shows each item's information and the test information: guessing and slipping lower the peak, and the 3PL peak moves above b. The bottom chart is the posterior of ability for the answers chosen, with a standard normal prior, under the chosen form (solid) and the other forms (dashed). The default items and answers are those of the worked 2PL example in the note on ability estimation: in 2PL the EAP is 0.312."
      readouts={
        <>
          <Readout label={`${state.form} MLE ± SE`} value={est(p.mle, p.se)} />
          <Readout label={`${state.form} EAP ± posterior sd`} value={est(p.eap, p.psd)} />
          {FORMS.filter((f) => f !== state.form).map((f) => (
            <Readout key={f} label={`${f} EAP`} value={formatNumber(posts[FORMS.indexOf(f)].eap)} />
          ))}
        </>
      }
    >
      <div className="space-y-2">
        <Plot x={xAxis} y={yAxis} height={260}>
          {seriesLayers(icc)}
          {handles.map((h, i) => (
            <Handle key={i} {...h} />
          ))}
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={200}>
          {seriesLayers(infoSeries)}
        </Plot>
        <Plot x={xAxis3} y={yAxis3} height={200}>
          {seriesLayers(postSeries)}
        </Plot>
      </div>
    </Figure>
  )
}
