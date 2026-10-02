/**
 * Concept activation vectors (`aifn/learning/explain`'s TCAV): an MLP trained on 8 × 8 images built from three
 * concepts (stripes, a corner dot, a vertical bar) with the label given by a rule over them. For each concept a
 * linear probe on a hidden layer's activations separates its examples from random images; the TCAV score of class 1 is
 * the share of its images whose logit rises along that direction, tested against random-against-random directions.
 */
import { useMemo, useState } from 'react'
import { CONCEPTS, conceptExamples, conceptImages, type Concept } from 'aifn-applied/data/synthetic'
import { stream } from 'aifn/foundation/random'
import { fromData, toFlat } from 'aifn/foundation/tensor'
import { denseForward, denseFunction, tcav, type TcavResult } from 'aifn/learning/explain'
import { Figure } from '@lab/layout'
import { call, choice, int, row, slider, useComputed, useFigureState } from '@lab/state'
import { Annotation, Bars, Plot, Plots, Points, Raster, Readout, useAxis } from '@lab/viz'
import { TrainRow } from './training'
import { fmt, imageRows, useTrainedMlp, type MlpSetup } from './mlp'

type Rule = 'stripes' | 'dot' | 'stripes-or-dot' | 'stripes-and-dot'
type Setup = { rule: Rule; n: number; noise: number; seed: number; width: number; steps: number }
const SHOWN: (Concept | 'random')[] = [...CONCEPTS, 'random']

export function ConceptsShowcase() {
  const state = useFigureState({
    data: row('1 · data and model', {
      rule: choice(
        [
          { value: 'stripes', label: 'class 1 ⇔ stripes' },
          { value: 'dot', label: 'class 1 ⇔ corner dot' },
          { value: 'stripes-or-dot', label: 'class 1 ⇔ stripes or dot' },
          { value: 'stripes-and-dot', label: 'class 1 ⇔ stripes and dot' },
        ],
        'stripes',
        { label: 'label rule' },
      ),
      n: int(1000, { ge: 100, le: 10000, suggestions: [500, 1000, 3000], label: 'training images' }),
      noise: slider(0.05, 1, 0.3, { step: 0.05, label: 'noise sd' }),
      seed: slider(0, 20, 0, { step: 1, label: 'seed' }),
      width: int(16, { ge: 2, le: 64, suggestions: [8, 16, 32], label: 'hidden units per layer' }),
      steps: int(600, { ge: 1, le: 5000, suggestions: [300, 600, 1500], label: 'Adam steps' }),
    }),
    probe: row('2 · TCAV', {
      layer: choice(
        [
          { value: 1, label: 'hidden layer 1' },
          { value: 2, label: 'hidden layer 2' },
        ],
        1,
        { label: 'layer' },
      ),
      randoms: int(8, { ge: 2, le: 40, suggestions: [4, 8, 20], label: 'random sets' }),
      examples: int(40, { ge: 5, le: 500, suggestions: [20, 40, 100], label: 'examples per set' }),
    }),
  })
  const current: Setup = { ...state.data, rule: state.data.rule as Rule }
  const [trained, setTrained] = useState<Setup | null>(null)
  const stale = trained !== null && JSON.stringify(trained) !== JSON.stringify(current)
  const shown = trained ?? current
  const setup = useMemo(
    (): MlpSetup | null =>
      trained && {
        data: call('data/synthetic/conceptImages', call('foundation/random/stream', `tcav-${trained.seed}`), {
          n: trained.n,
          rule: trained.rule,
          noise: trained.noise,
        }),
        inputs: 64,
        width: trained.width,
        depth: 2,
        activation: 'tanh',
        steps: trained.steps,
        rate: 0.01,
        l2: 1e-3,
        seed: trained.seed,
      },
    [trained],
  )
  const mlp = useTrainedMlp(setup)
  const net = mlp.net
  const { layer, randoms, examples } = { ...state.probe, layer: Number(state.probe.layer) }
  const sets = useMemo(() => {
    const noise = shown.noise
    const concept = Object.fromEntries(
      CONCEPTS.map((c) => [c, conceptExamples(stream(`tcav-concept-${c}`), c, { n: examples, noise }).x]),
    ) as Record<Concept, ReturnType<typeof conceptExamples>['x']>
    const random = Array.from(
      { length: randoms },
      (_, k) => conceptExamples(stream(`tcav-random-${k}`), 'random', { n: examples, noise }).x,
    )
    const test = conceptImages(stream(`tcav-test-${shown.seed}`), { n: 400, rule: shown.rule, noise })
    const y = toFlat(test.y!)
    const X = toFlat(test.x)
    const rows = Array.from({ length: 400 }, (_, i) => i)
      .filter((i) => y[i] === 1)
      .slice(0, 150)
    const positives = fromData(Float64Array.from(rows.flatMap((i) => Array.from(X.slice(i * 64, (i + 1) * 64)))), [
      rows.length,
      64,
    ])
    return { concept, random, positives }
  }, [shown.noise, shown.seed, shown.rule, randoms, examples])
  const result = useComputed(
    () => {
      if (!net) return null
      const act = (X: Parameters<typeof denseForward>[1]) => {
        const f = denseForward(net, X)
        return fromData(f.post[layer], [f.rows, f.sizes[layer]])
      }
      const head = denseFunction(net, { from: layer })
      const classActs = act(sets.positives)
      const randomActs = sets.random.map(act)
      return Object.fromEntries(
        CONCEPTS.map((c) => [c, tcav(head, classActs, act(sets.concept[c]), randomActs, { l2: 0.01 })]),
      ) as Record<Concept, TcavResult>
    },
    [net, sets, layer],
    { mode: 'release' },
  )
  const r = result.value
  const px = useAxis({ label: '', range: [-0.5, 7.5], nice: false })
  const py = useAxis({ label: '', range: [-0.5, 7.5], nice: false, equal: px })
  const cAxis = useAxis({ label: 'concept', range: [0.4, 3.6], integer: true, format: (v) => CONCEPTS[v - 1] ?? '' })
  const sAxis = useAxis({ label: 'TCAV score of class 1', range: [0, 1] })
  const thumb = (c: Concept | 'random') => {
    const X = c === 'random' ? sets.random[0] : sets.concept[c]
    return toFlat(X).slice(0, 64)
  }
  const jitter = (k: number, j: number, count: number) => k + 1 + (count > 1 ? (j / (count - 1) - 0.5) * 0.3 : 0)
  return (
    <Figure
      title="Does class 1 depend on a concept? TCAV"
      purpose="A concept activation vector points from random images towards a concept's images in a hidden layer; class 1's TCAV score is the share of its images whose logit increases along it. A concept the label depends on scores near 1 (or 0 if it lowers the logit) consistently across random sets, while one it ignores scores like random directions do."
      state={state}
      defaultSize="XL"
      controls={
        <TrainRow
          label="3 · train"
          trained={trained !== null}
          stale={stale}
          mlp={mlp}
          onTrain={() => setTrained({ ...current })}
        />
      }
      readouts={{
        'TCAV of class 1: mean ± sd over random sets · p-value against random directions · probe accuracy · mean directional derivative':
          (
            <>
              {CONCEPTS.map((c) => (
                <Readout
                  key={c}
                  label={c}
                  value={
                    r
                      ? `${fmt(r[c].mean, 2)} ± ${fmt(r[c].sd, 2)} · p = ${fmt(r[c].pValue, 2)}${r[c].significant ? ' (significant)' : ''} · ${fmt(r[c].accuracies.reduce((a, b) => a + b, 0) / r[c].accuracies.length, 2)} · ${fmt(r[c].sensitivity, 2)}`
                      : '—'
                  }
                />
              ))}
            </>
          ),
      }}
      caption={`Data: aifn conceptImages (seeded): ${shown.n} noisy 8 × 8 images, each holding horizontal stripes, a corner dot and a vertical bar independently with probability ½; the label rule says which make class 1 (the bar never matters). Press Train to fit a 64 → ${shown.width} → ${shown.width} → 1 tanh MLP by Adam in the worker. Top: one example of each concept's set (every image holds the concept, the other concepts at random) and a random image (drawn like the training images), so a concept set differs from a random set only by its concept. For each concept, a CAV is fitted (L2 logistic regression on the chosen layer's activations) against each of ${randoms} random sets of ${examples} images; the score is the share of 150 class-1 test images whose logit's gradient with respect to the activations points along the CAV. Bottom: each concept's mean score (bar), its score per random set (coloured points) and the scores of random-against-random CAVs (grey points), the null; a two-sided t-test compares them at level 0.05. Change the rule and watch the scores follow it. The score counts only signs: a concept the network barely responds to can still score 0 or 1 when the sign of its tiny effect is the same for every image, and then the t-test calls it significant; the last readout, the mean directional derivative, shows how much the logit actually moves along each CAV. At hidden layer 2 the rest of the network is linear, so every image has the same gradient and every score is 0 or 1: TCAV needs a layer with nonlinearity above it.`}
    >
      <Plots cols={4} scale={0.35}>
        {SHOWN.map((c) => (
          <Plot key={c} x={px} y={py} bare title={c === 'random' ? 'a random image' : `${c} set`}>
            <Raster
              x={Array.from({ length: 8 }, (_, k) => k)}
              y={Array.from({ length: 8 }, (_, k) => k)}
              z={imageRows(thumb(c), 8, 8)}
              colorBar={false}
            />
          </Plot>
        ))}
      </Plots>
      <Plots cols={1}>
        <Plot x={cAxis} y={sAxis}>
          {r && (
            <Bars
              name="mean TCAV score"
              x={[1, 2, 3]}
              y={CONCEPTS.map((c) => r[c].mean)}
              slot={2}
              opacity={0.5}
              stale={result.stale}
            />
          )}
          {r && (
            <Points
              name="random-against-random"
              x={CONCEPTS.flatMap((c, k) =>
                Array.from(r[c].randomScores, (_, j) => jitter(k, j, r[c].randomScores.length) + 0.06),
              )}
              y={CONCEPTS.flatMap((c) => Array.from(r[c].randomScores))}
              muted
            />
          )}
          {r && (
            <Points
              name="concept against each random set"
              x={CONCEPTS.flatMap((c, k) => Array.from(r[c].scores, (_, j) => jitter(k, j, r[c].scores.length)))}
              y={CONCEPTS.flatMap((c) => Array.from(r[c].scores))}
              slot={2}
            />
          )}
          <Annotation y={0.5} dashed />
        </Plot>
      </Plots>
    </Figure>
  )
}
