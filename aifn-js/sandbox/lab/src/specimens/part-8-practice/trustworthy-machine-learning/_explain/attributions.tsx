/**
 * Attributions compared and evaluated (`aifn/learning/explain`): an MLP trained on noise images (or sequences) in which
 * class 1 carries a planted motif. A test example is explained by seven methods beside the motif's true cells; deletion
 * and insertion curves rank them by faithfulness; cascading model randomisation and a model trained on random labels
 * check whether each explanation depends on the model at all (Adebayo et al.'s sanity checks).
 */
import { useMemo, useState } from 'react'
import { plantedMask, plantedPatterns, plantedShape } from 'aifn-methods/data/synthetic'
import { normal, stream } from 'aifn/foundation/random'
import { fromData, mean, toFlat, type Tensor } from 'aifn/foundation/tensor'
import {
  deepLift,
  deletionCurve,
  denseFunction,
  denseOutput,
  expectedGradients,
  explanationSimilarity,
  inputGradient,
  integratedGradients,
  occlusion,
  randomiseNetwork,
  relevanceMass,
  smoothGrad,
  type DenseNetwork,
} from 'aifn/learning/explain'
import { sigmoid } from 'aifn/numerics/special'
import { Figure } from '@lab/layout'
import { call, choice, int, row, slider, toggle, useComputed, useFigureState } from '@lab/state'
import { Bars, Curve, Plot, Plots, Raster, Readout, useAxis } from '@lab/viz'
import { TrainRow } from './training'
import { fmt, imageRows, useTrainedMlp, type MlpSetup } from './mlp'

type Kind = 'image' | 'sequence'
type Setup = { kind: Kind; n: number; noise: number; seed: number; width: number; steps: number }

const METHODS = [
  { key: 'gradient', label: 'gradient', slot: 0 },
  { key: 'gxi', label: 'gradient × (x − x′)', slot: 1 },
  { key: 'ig', label: 'integrated gradients', slot: 2 },
  { key: 'smooth', label: 'SmoothGrad', slot: 3 },
  { key: 'deeplift', label: 'DeepLIFT', slot: 4 },
  { key: 'expected', label: 'expected gradients', slot: 5 },
  { key: 'occlusion', label: 'occlusion', slot: 6 },
  { key: 'random', label: 'random', slot: 7 },
] as const
type MethodKey = (typeof METHODS)[number]['key']

/** Every method's attribution of the network's logit at x. */
function attribute(
  net: DenseNetwork,
  x: number[],
  baseline: number[],
  background: Tensor,
  kind: Kind,
  seed: string,
): Record<MethodKey, Float64Array> {
  const f = denseFunction(net)
  const model = (Z: Tensor) => denseOutput(net, Z)
  const g = inputGradient(f, x)
  const shape = plantedShape(kind)
  return {
    gradient: g.gradient,
    gxi: Float64Array.from(g.gradient, (v, i) => v * (x[i] - baseline[i])),
    ig: integratedGradients(f, x, { baseline, steps: 32 }).values,
    smooth: smoothGrad(f, x, stream(`sg-${seed}`), { samples: 32, noise: 0.3 }).values,
    deeplift: deepLift(net, x, { baseline }).values,
    expected: expectedGradients(f, x, background, stream(`eg-${seed}`), { samples: 128 }).values,
    occlusion: occlusion(model, x, {
      shape: kind === 'image' ? shape : [shape[1]],
      window: kind === 'image' ? [2, 2] : [3],
      baseline,
    }).values,
    random: Float64Array.from(toFlat(normal(stream(`random-${seed}`), 0, 1, { shape: [x.length] }))),
  }
}

export function AttributionsShowcase() {
  const state = useFigureState({
    data: row('1 · data and model', {
      kind: choice(
        [
          { value: 'image', label: '8 × 8 images, planted plus' },
          { value: 'sequence', label: 'length-32 sequences, planted bump' },
        ],
        'image',
        { label: 'inputs' },
      ),
      n: int(1200, { ge: 100, le: 10000, suggestions: [600, 1200, 3000], label: 'training rows' }),
      noise: slider(0.1, 1, 0.35, { step: 0.05, label: 'noise sd' }),
      seed: slider(0, 20, 0, { step: 1, label: 'seed' }),
      width: int(32, { ge: 4, le: 128, suggestions: [16, 32, 64], label: 'hidden units' }),
      steps: int(800, { ge: 1, le: 10000, suggestions: [400, 800, 2000], label: 'Adam steps' }),
    }),
    explain: row('2 · explain', {
      example: slider(0, 19, 0, { step: 1, label: 'test example (class 1)' }),
      baseline: choice(
        [
          { value: 'zero', label: 'zero (the noise mean)' },
          { value: 'mean', label: 'mean training input' },
          { value: 'noise', label: 'a class-0 test example' },
        ],
        'zero',
        { label: 'baseline' },
      ),
    }),
  })
  const current: Setup = { ...state.data, kind: state.data.kind as Kind }
  const [trained, setTrained] = useState<Setup | null>(null)
  const stale = trained !== null && JSON.stringify(trained) !== JSON.stringify(current)
  const shown = trained ?? current
  const kind = shown.kind
  const [h, w] = plantedShape(kind)
  const d = h * w
  const dataTask = (t: Setup) =>
    call('data/synthetic/plantedPatterns', call('foundation/random/stream', `attr-${t.seed}`), {
      n: t.n,
      kind: t.kind,
      noise: t.noise,
    })
  const setup = useMemo(
    (): MlpSetup | null =>
      trained && {
        data: dataTask(trained),
        inputs: plantedShape(trained.kind)[0] * plantedShape(trained.kind)[1],
        width: trained.width,
        depth: 2,
        activation: 'relu',
        steps: trained.steps,
        rate: 0.01,
        l2: 1e-3,
        seed: trained.seed,
      },
    [trained],
  )
  // The same network trained on labels flipped at random with probability ½ (labels independent of the inputs).
  const randomSetup = useMemo(
    (): MlpSetup | null =>
      setup && {
        ...setup,
        data: call(
          'data/synthetic/withLabelNoise',
          call('foundation/random/stream', `attr-shuffle-${trained!.seed}`),
          setup.data,
          { rate: 0.5 },
        ),
      },
    [setup, trained],
  )
  const mlp = useTrainedMlp(setup)
  const randomMlp = useTrainedMlp(randomSetup, { latest: true })
  const train = useMemo(
    () => plantedPatterns(stream(`attr-${shown.seed}`), { n: shown.n, kind, noise: shown.noise }),
    [shown.seed, shown.n, kind, shown.noise],
  )
  const test = useMemo(
    () => plantedPatterns(stream(`attr-test-${shown.seed}`), { n: 200, kind, noise: shown.noise }),
    [shown.seed, kind, shown.noise],
  )
  const testX = useMemo(() => Float64Array.from(toFlat(test.x)), [test])
  const positives = useMemo(() => {
    const y = toFlat(test.y!)
    return Array.from({ length: 200 }, (_, i) => i).filter((i) => y[i] === 1)
  }, [test])
  const negative = useMemo(() => Array.from(toFlat(test.y!)).indexOf(0), [test])
  const index = positives[Math.min(state.explain.example, positives.length - 1)]
  const x = useMemo(() => Array.from(testX.slice(index * d, (index + 1) * d)), [testX, index, d])
  const mask = useMemo(() => plantedMask(kind, toFlat(test.t!)[index]), [kind, test, index])
  const background = useMemo(() => {
    const rows = Math.min(100, train.x.shape[0])
    return fromData(Float64Array.from(toFlat(train.x).slice(0, rows * d)), [rows, d])
  }, [train, d])
  const baseline = useMemo(() => {
    if (state.explain.baseline === 'mean') return Array.from(toFlat(mean(train.x, 0) as Tensor))
    if (state.explain.baseline === 'noise') return Array.from(testX.slice(negative * d, (negative + 1) * d))
    return new Array<number>(d).fill(0)
  }, [state.explain.baseline, train, testX, negative, d])

  const net = mlp.net
  const maps = useComputed(
    () => (net ? attribute(net, x, baseline, background, kind, `${index}`) : null),
    [net, x, baseline, background, kind, index],
    { mode: 'release' },
  )
  const m = maps.value
  const px = useAxis({ label: '', range: [-0.5, w - 0.5], nice: false })
  const py = useAxis({ label: '', range: [-0.5, h - 0.5], nice: false, ...(kind === 'image' ? { equal: px } : {}) })
  const panel = (title: string, v: ArrayLike<number> | null, signed: boolean) => {
    const top = v ? Math.max(1e-9, ...Array.from(v, Math.abs)) : 1
    return (
      <Plot key={title} x={px} y={py} bare title={title}>
        {v && (
          <Raster
            x={Array.from({ length: w }, (_, c) => c)}
            y={Array.from({ length: h }, (_, r) => r)}
            z={imageRows(v, h, w)}
            scale={signed ? 'diverging' : 'sequential'}
            range={signed ? [-top, top] : [Math.min(0, ...Array.from(v)), top]}
            colorBar={false}
            stale={maps.stale}
          />
        )}
      </Plot>
    )
  }
  return (
    <>
      <Figure
        title="Seven attributions of a planted motif"
        purpose="Each method assigns the network's logit to the input cells; on this task the right answer is known (the motif's cells), so the maps can be compared with it and with each other."
        state={state}
        defaultSize="XL"
        controls={
          <TrainRow
            label="3 · train"
            trained={trained !== null}
            stale={stale}
            mlp={mlp}
            onTrain={() => setTrained({ ...current })}
            note={randomMlp.run.running ? 'random-label model training…' : undefined}
          />
        }
        readouts={{
          [`test example ${index}: share of |attribution| on the motif`]: (
            <>
              {METHODS.map((k) => (
                <Readout key={k.key} label={k.label} value={m ? fmt(relevanceMass(m[k.key], mask), 2) : '—'} />
              ))}
              <Readout label="motif's share of cells" value={fmt(mask.filter((v) => v > 0).length / d, 2)} />
            </>
          ),
        }}
        caption={`Data: aifn plantedPatterns (seeded): ${shown.n} ${kind === 'image' ? '8 × 8 images' : 'length-32 sequences'} of Gaussian noise (sd ${shown.noise}); half carry a motif of height 1.2 at a random place (${kind === 'image' ? 'a 3 × 3 plus' : 'a five-step bump'}). Press Train to fit a ${d} → ${shown.width} → ${shown.width} → 1 ReLU MLP by Adam in the worker (a second copy is trained at the same time on labels flipped with probability ½, for the sanity check below). The panels show the chosen class-1 test example, the motif's true cells and each attribution of the logit (red positive, blue negative, each scaled to its own largest value). Gradients and SmoothGrad (32 noisy copies, sd 0.3) are local slopes; gradient × (x − baseline x′), integrated gradients (32 steps), DeepLIFT (rescale rule) and occlusion (2 × 2 windows set to the baseline) measure change from the baseline; expected gradients averages integrated gradients over 100 training inputs as baselines. Change the baseline to see the baseline-dependent methods move: from the mean input, cells that are typical carry nothing. The readout is each map's share of absolute mass on the motif.`}
      >
        <Plots rows={2} cols={5}>
          {panel('input', x, true)}
          {panel('motif (truth)', mask, false)}
          {METHODS.map((k) => panel(k.label, m ? m[k.key] : null, true))}
        </Plots>
      </Figure>
      <DeletionFigure
        net={net}
        x={x}
        baseline={baseline}
        maps={m}
        positives={positives}
        testX={testX}
        d={d}
        background={background}
        kind={kind}
      />
      <SanityFigure
        net={net}
        randomNet={randomMlp.net}
        x={x}
        baseline={baseline}
        background={background}
        kind={kind}
        index={index}
      />
    </>
  )
}

// ── Deletion and insertion ───────────────────────────────────────────────────────────────────────────────────────────

function DeletionFigure({
  net,
  x,
  baseline,
  maps,
  positives,
  testX,
  d,
  background,
  kind,
}: {
  net: DenseNetwork | null
  x: number[]
  baseline: number[]
  maps: Record<MethodKey, Float64Array> | null
  positives: number[]
  testX: Float64Array
  d: number
  background: Tensor
  kind: Kind
}) {
  const state = useFigureState({
    curve: row('1 · curve', {
      mode: choice(
        [
          { value: 'deletion', label: 'deletion (most important first)' },
          { value: 'insertion', label: 'insertion (from the baseline)' },
        ],
        'deletion',
        { label: 'curve' },
      ),
      average: toggle(false, 'average area over 20 test examples'),
    }),
  })
  const mode = state.curve.mode as 'deletion' | 'insertion'
  // P(class 1), so every curve and area lies in [0, 1].
  const model = (Z: Tensor) =>
    net ? Float64Array.from(denseOutput(net, Z), (z) => sigmoid(z) as number) : new Float64Array(Z.shape[0])
  const curves = useMemo(() => {
    if (!maps || !net) return null
    return METHODS.map((k) =>
      deletionCurve(model, x, maps[k.key], { mode, baseline, step: Math.max(1, Math.round(d / 32)) }),
    )
  }, [maps, net, x, mode, baseline, d]) // eslint-disable-line react-hooks/exhaustive-deps
  const averages = useComputed(
    () => {
      if (!net || !state.curve.average) return null
      const areas = METHODS.map(() => 0)
      const rows = positives.slice(0, 20)
      for (const i of rows) {
        const xi = Array.from(testX.slice(i * d, (i + 1) * d))
        const a = attribute(net, xi, baseline, background, kind, `avg-${i}`)
        METHODS.forEach((k, j) => {
          areas[j] +=
            deletionCurve(model, xi, a[k.key], { mode, baseline, step: Math.max(1, Math.round(d / 16)) }).area /
            rows.length
        })
      }
      return areas
    },
    [net, state.curve.average, positives, testX, d, baseline, background, kind, mode],
    { mode: 'release' },
  )
  const fAxis = useAxis({
    label: mode === 'deletion' ? 'fraction of cells removed' : 'fraction of cells restored',
    range: [0, 1],
  })
  const oAxis = useAxis({
    label: 'P(class 1)',
    range: [0, 1],
    hold: 'union',
    key: `${mode}-${x.join(',').slice(0, 40)}`,
  })
  const mAxis = useAxis({
    label: 'method',
    range: [0.4, METHODS.length + 0.6],
    integer: true,
    format: (v) => METHODS[v - 1]?.label.split(' ')[0] ?? '',
  })
  const aAxis = useAxis({ label: `area under the ${mode} curve`, range: [0, 1] })
  const shownAreas = averages.value ?? curves?.map((c) => c.area) ?? null
  return (
    <Figure
      title="Deletion and insertion curves"
      purpose="A faithful attribution names the cells the output depends on: removing them in its order should drop the class probability fastest (a small area under the deletion curve), and restoring them from the baseline should raise it fastest (a large area under the insertion curve); a random order is the reference."
      state={state}
      defaultSize="L"
      readouts={{
        [`area under the curve (${averages.value ? 'mean over 20 examples' : 'this example'})`]: (
          <>
            {METHODS.map((k, j) => (
              <Readout key={k.key} label={k.label} value={shownAreas ? fmt(shownAreas[j]) : '—'} />
            ))}
          </>
        ),
      }}
      caption={`Uses the network, test example and baseline chosen above. Left: the network's P(class 1) as cells are set to the baseline in each method's order (or, for insertion, as they are restored from it), ${Math.max(1, Math.round(d / 32))} at a time. Right: the area under each curve, for this example or averaged over 20 class-1 test examples (toggle; recomputed on release). For deletion lower is better, for insertion higher. The random order (dashed) is the reference every method should beat.`}
    >
      <Plots cols={2} widths={[60, 40]}>
        <Plot x={fAxis} y={oAxis}>
          {curves &&
            METHODS.map((k, j) => (
              <Curve
                key={k.key}
                name={k.label}
                x={Array.from(curves[j].fraction)}
                y={Array.from(curves[j].output)}
                slot={k.slot}
                dashed={k.key === 'random'}
              />
            ))}
        </Plot>
        <Plot x={mAxis} y={aAxis}>
          {shownAreas && (
            <Bars name="area" x={METHODS.map((_, j) => j + 1)} y={shownAreas} slot={2} stale={averages.stale} />
          )}
        </Plot>
      </Plots>
    </Figure>
  )
}

// ── Sanity checks ────────────────────────────────────────────────────────────────────────────────────────────────────

function SanityFigure({
  net,
  randomNet,
  x,
  baseline,
  background,
  kind,
  index,
}: {
  net: DenseNetwork | null
  randomNet: DenseNetwork | null
  x: number[]
  baseline: number[]
  background: Tensor
  kind: Kind
  index: number
}) {
  const state = useFigureState({
    check: row('1 · check', {
      seed: slider(0, 20, 0, { step: 1, label: 'randomisation seed' }),
    }),
  })
  const result = useComputed(
    () => {
      if (!net) return null
      const nets = randomiseNetwork(net, stream(`sanity-${state.check.seed}`))
      const maps = nets.map((nn) => attribute(nn, x, baseline, background, kind, `${index}`))
      const cascade = METHODS.map((k) =>
        maps.map((mp) => explanationSimilarity(maps[0][k.key], mp[k.key]).spearmanAbsolute),
      )
      const data = randomNet
        ? METHODS.map(
            (k) =>
              explanationSimilarity(
                maps[0][k.key],
                attribute(randomNet, x, baseline, background, kind, `${index}`)[k.key],
              ).spearmanAbsolute,
          )
        : null
      return { cascade, data, layers: nets.length - 1 }
    },
    [net, randomNet, x, baseline, background, kind, index, state.check.seed],
    { mode: 'release' },
  )
  const r = result.value
  const kAxis = useAxis({
    label: 'layers re-initialised from the top',
    range: [-0.2, (r?.layers ?? 3) + 0.2],
    integer: true,
  })
  const sAxis = useAxis({ label: 'rank correlation of |attribution| with the original', range: [-0.4, 1.05] })
  return (
    <Figure
      title="Sanity checks: randomise the model, then the labels"
      purpose="An explanation of a model should change when the model does: re-initialising the network's layers from the top down, or training it on random labels, should destroy the attribution's agreement with the original; a method whose map survives is showing the input, not the model."
      state={state}
      defaultSize="L"
      readouts={{
        'random-label model: rank correlation with the original': (
          <>
            {METHODS.map((k, j) => (
              <Readout key={k.key} label={k.label} value={r?.data ? fmt(r.data[j], 2) : '—'} />
            ))}
          </>
        ),
      }}
      caption="Uses the network, example and baseline above. Cascading randomisation (Adebayo et al., 2018): step k redraws the weights and biases of the top k layers from normals with each layer's own weight spread, and every method explains the same example again; the curves are Spearman's rank correlation of the absolute attributions with the original's. A method that passes falls towards 0; one that stays high is reading the input rather than the model. The random map (dashed) never depends on the model, so it stays at 1: the failing reference. The readouts compare each method with the same network trained on random labels (trained beside the first when you press Train)."
    >
      <Plot x={kAxis} y={sAxis}>
        {r &&
          METHODS.map((k, j) => (
            <Curve
              key={k.key}
              name={k.label}
              x={r.cascade[j].map((_, s) => s)}
              y={r.cascade[j]}
              slot={k.slot}
              showPoints
              dashed={k.key === 'random'}
              stale={result.stale}
            />
          ))}
      </Plot>
    </Figure>
  )
}
