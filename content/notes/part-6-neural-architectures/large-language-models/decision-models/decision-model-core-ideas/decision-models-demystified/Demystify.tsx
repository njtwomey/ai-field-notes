import { useMemo } from 'react'
import { Bars, choice, Figure, formatNumber, Plot, Readout, setting, useAxis, useFigureState } from 'aifn-render'

type Option = { name: string; score: number }
type Scenario = { text: string; question: string; options: Option[]; none: number; rating?: boolean }

/** Made-up scores a fine-tuned model might give each answer's letter; they follow the option, not its letter. */
const SCENARIOS: Record<string, Scenario> = {
  yesno: {
    text: 'Hi, I was charged twice this month. Can you refund one of the payments?',
    question: 'Is the customer asking for a refund?',
    options: [
      { name: 'no', score: -1 },
      { name: 'yes', score: 4 },
    ],
    none: -2,
  },
  categories: {
    text: 'Hi, I was charged twice this month. Can you refund one of the payments?',
    question: 'Which team should handle this message?',
    options: [
      { name: 'billing', score: 3.8 },
      { name: 'technical', score: 0.5 },
      { name: 'account', score: 1.2 },
      { name: 'sales', score: -0.5 },
    ],
    none: 0.2,
  },
  rating: {
    text: 'Hi, I was charged twice this month. Can you refund one of the payments?',
    question: 'How upset is the customer, from 0 (calm) to 4 (furious)?',
    options: [
      { name: '0', score: 0.8 },
      { name: '1', score: 2.0 },
      { name: '2', score: 1.6 },
      { name: '3', score: 0.2 },
      { name: '4', score: -1 },
    ],
    none: -3,
    rating: true,
  },
  nofit: {
    text: 'My parcel says delivered but nothing arrived. Where is it?',
    question: 'Which team should handle this message?',
    options: [
      { name: 'billing', score: 0.4 },
      { name: 'technical', score: -0.5 },
      { name: 'account', score: 0.1 },
      { name: 'sales', score: -1 },
    ],
    none: 2.5,
  },
}

/** Things the model might also want to write next, which are not answers; their scores are ignored. */
const OTHER = [
  { token: 'Sure', score: 1.5 },
  { token: 'The', score: 1.0 },
  { token: 'I', score: 0.5 },
]

const softmax = (z: number[]) => {
  const m = Math.max(...z)
  const e = z.map((v) => Math.exp(v - m))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / s)
}

export function Demystify() {
  const state = useFigureState({
    scenario: choice(
      [
        { value: 'yesno', label: 'a yes-or-no question' },
        { value: 'categories', label: 'pick one of several categories' },
        { value: 'rating', label: 'a rating from 0 to 4' },
        { value: 'nofit', label: 'a message no category fits' },
      ],
      'categories',
      { label: 'scenario' },
    ),
    none: setting(false, 'offer "none of these"'),
    reverse: setting(false, 'list the options in reverse order'),
  })
  const sc = SCENARIOS[state.scenario]

  const options = useMemo(() => {
    let list = [...sc.options]
    if (state.reverse) list = list.reverse()
    if (state.none && !sc.rating) list.push({ name: 'none of these', score: sc.none })
    return list.map((o, i) => ({ ...o, letter: String.fromCharCode(65 + i) }))
  }, [sc, state.none, state.reverse])

  const all = useMemo(() => softmax([...options.map((o) => o.score), ...OTHER.map((o) => o.score)]), [options])
  const q = useMemo(() => softmax(options.map((o) => o.score)), [options])
  const k = q.indexOf(Math.max(...q))
  const onLetters = all.slice(0, options.length).reduce((a, b) => a + b, 0)
  const expected = sc.rating ? options.reduce((s, o, i) => s + Number(o.name) * q[i], 0) : NaN

  const n = options.length
  const letterX = useMemo(() => options.map((_, i) => i), [options])
  const otherX = useMemo(() => OTHER.map((_, i) => n + i), [n])
  const ax = useAxis({
    label: 'what the model could write next',
    categories: [...options.map((o) => o.letter), ...OTHER.map((o) => o.token)],
    key: n,
  })
  const ay = useAxis({ label: 'chance (whole vocabulary)', range: [0, 1] })
  const bx = useAxis({
    label: 'answer',
    categories: options.map((o) => `${o.letter} = ${o.name}`),
    key: options.map((o) => o.name).join(),
  })
  const by = useAxis({ label: 'probability', range: [0, 1] })

  return (
    <Figure
      title="One question, from prompt to answer"
      state={state}
      defaultSize="L"
      caption="Top: what the model reads, simplified. The allowed answers are written into the prompt as plain text, each with a letter, and the prompt stops where the reply would start. Left: how much the model wants to write each possible next word (a few of about 250,000); the letters are coloured. Right: keep only the letters, rescale them to add up to 1, and look each letter up in the list. Nothing is generated. Turn on 'none of these' in the scenario no category fits; reverse the order and see the letters move while each answer keeps its probability."
      readouts={
        <>
          <Readout label="answer" value={sc.rating ? `${formatNumber(expected)} (average level)` : options[k].name} />
          <Readout label="probability" value={formatNumber(q[k])} />
          <Readout label="share of the model's next-word chance on the letters" value={formatNumber(onLetters)} />
        </>
      }
    >
      <pre className="mb-4 rounded-md border bg-muted/40 p-3 font-mono text-xs leading-relaxed whitespace-pre-wrap">
        {`Choose the best answer. Reply with its letter only.\n\nText: ${sc.text}\nQuestion: ${sc.question}\n`}
        {options.map((o) => `${o.letter}: ${o.name}\n`).join('')}
        {'\nAnswer: '}
        <span className="rounded-sm border border-dashed border-primary px-1 text-primary">? (read, not written)</span>
      </pre>
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={ax} y={ay} height={240}>
          <Bars name="answer letters" x={letterX} y={all.slice(0, n)} slot={0} />
          <Bars name="other words" x={otherX} y={all.slice(n)} muted />
        </Plot>
        <Plot x={bx} y={by} height={240}>
          <Bars name="probability" x={letterX} y={q} slot={0} />
        </Plot>
      </div>
    </Figure>
  )
}
