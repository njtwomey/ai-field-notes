import { useMemo } from 'react'
import {
  anisotropicBlobs,
  arSeries,
  blobs,
  casino,
  checkerboard,
  checkerboardImage,
  circles,
  clickLog,
  digits,
  gradientImage,
  moons,
  regression1d,
  rings,
  sCurve,
  seasonalSeries,
  shapesImage,
  spirals,
  swissRoll,
  xor,
  zipfCatalogue,
  type RegressionFunction,
} from 'aifn-methods/data/synthetic'
import { anscombe, iris, oldFaithful } from 'aifn-methods/data/real'
import { type Dataset } from 'aifn-methods/data'
import { child, stream } from 'aifn/foundation/random'
import { toFlat, toRows, type Tensor } from 'aifn/foundation/tensor'
import { Figure } from '@lab/layout'
import { int, choice, row, slider, useFigureState, variants } from '@lab/state'
import { Area, Curve, Plot, Plots, Points, Raster, Readout, useAxis } from '@lab/viz'
import { useDatasetSeries } from '@lab/views'

// ---------------------------------------------------------------------------------------------------------------------
// 1. A gallery of 2-D generators.

const GENERATORS = variants(
  {
    moons: { label: 'moons', params: { noise: slider(0, 0.4, 0.1, { label: 'noise sd' }) } },
    circles: { label: 'circles', params: { noise: slider(0, 0.3, 0.05, { label: 'noise sd' }) } },
    rings: { label: 'rings', params: { noise: slider(0, 0.5, 0.1, { label: 'noise sd' }) } },
    spirals: { label: 'spirals', params: { noise: slider(0, 0.2, 0.03, { label: 'noise sd' }) } },
    xor: { label: 'XOR (Gaussian)', params: { noise: slider(0.05, 1, 0.4, { label: 'cluster sd' }) } },
    checkerboard: { label: 'checkerboard', params: {} },
    blobs: { label: 'blobs', params: { noise: slider(0.2, 4, 1, { label: 'cluster sd' }) } },
    anisotropic: { label: 'anisotropic blobs', params: { noise: slider(0.2, 4, 1, { label: 'cluster sd' }) } },
  },
  { label: '1 · generator', choiceLabel: 'dataset' },
)

type GeneratorKey = keyof typeof GENERATORS.specs

const MAKE: Record<GeneratorKey, (seed: number, n: number, noise: number) => Dataset> = {
  moons: (k, n, e) => moons(child(stream('gallery'), k), { n, noise: e }),
  circles: (k, n, e) => circles(child(stream('gallery'), k), { n, noise: e }),
  rings: (k, n, e) => rings(child(stream('gallery'), k), { n, noise: e }),
  spirals: (k, n, e) => spirals(child(stream('gallery'), k), { n, arms: 3, noise: e }),
  xor: (k, n, e) => xor(child(stream('gallery'), k), { n, kind: 'gaussian', sd: e }),
  checkerboard: (k, n) => checkerboard(child(stream('gallery'), k), { n }),
  blobs: (k, n, e) => blobs(child(stream('gallery'), k), { n, sd: e }),
  anisotropic: (k, n, e) => anisotropicBlobs(child(stream('gallery'), k), { n, sd: e }),
}

export function GallerySpecimen() {
  const state = useFigureState({
    gen: GENERATORS,
    sample: row('2 · sample', {
      n: slider(20, 1000, 300, { label: 'points n', step: 10 }),
      seed: int(0, { ge: 0, label: 'stream' }),
    }),
  })
  const { gen } = state
  const generator = gen.key
  const noise = 'noise' in gen.values ? gen.values.noise : 0
  const { seed, n } = state.sample
  const data = useMemo(() => MAKE[generator](seed, n, noise), [generator, seed, n, noise])
  const { points, xLabel, yLabel } = useDatasetSeries(data, undefined, 'label')
  const key = `${generator}:${seed}`
  const x = useAxis({ label: xLabel, hold: 'union', key })
  const y = useAxis({ label: yLabel, hold: 'union', key, equal: x })
  return (
    <Figure
      title="Seeded 2-D datasets"
      purpose="The same stream gives the same points: the noise and the point count change only their own draws, a new stream redraws everything."
      state={state}
      defaultSize="L"
      readouts={
        <>
          <Readout label="stream" value={data.meta.key?.path ?? '—'} />
          <Readout label="x" value={data.x.shape.join(' × ')} />
          <Readout label="classes" value={data.meta.labelNames?.length ?? 0} />
        </>
      }
      caption={
        data.meta.description +
        (data.meta.source ? ` Recipe: ${data.meta.source}.` : '') +
        ' Axes hold while the noise and n move; a new generator or stream refits them.'
      }
    >
      <Plot x={x} y={y}>
        <Points {...points} />
      </Plot>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 2. Manifolds in three dimensions.

export function ManifoldSpecimen() {
  const state = useFigureState({
    which: choice(
      [
        { value: 'swiss', label: 'Swiss roll' },
        { value: 's', label: 'S-curve' },
      ],
      'swiss',
      { label: 'manifold' },
    ),
  })
  const which = state.which
  const data = useMemo(
    () => (which === 'swiss' ? swissRoll(stream('manifold'), { n: 800 }) : sCurve(stream('manifold'), { n: 800 })),
    [which],
  )
  const top = useDatasetSeries(data, [0, 2], 't')
  const side = useDatasetSeries(data, [0, 1], 't')
  const x = useAxis({ label: 'x', hold: 'initial', key: which })
  const z = useAxis({ label: 'z', hold: 'initial', key: which, equal: x })
  const x2 = useAxis({ label: 'x', hold: 'initial', key: which })
  const y = useAxis({ label: 'y', hold: 'initial', key: which })
  return (
    <Figure
      title="The Swiss roll and the S-curve"
      purpose="A flat 2-D sheet rolled or bent in three dimensions: neighbours along the sheet (similar colour) can be far apart in space, which is what a manifold embedding must undo."
      state={state}
      defaultSize="L"
      caption={`${data.meta.description} Colour is the coordinate t along the sheet. Left: seen from above (x, z), where the roll's turns are visible; right: from the side (x, y), the sheet's width.`}
    >
      <Plots cols={2}>
        <Plot x={x} y={z}>
          <Points {...top.points} />
        </Plot>
        <Plot x={x2} y={y}>
          <Points {...side.points} />
        </Plot>
      </Plots>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 3. Real data.

const IRIS = iris()
const FAITHFUL = oldFaithful()
const ANSCOMBE = anscombe()
const IRIS_FEATURES = IRIS.meta.featureNames.map((name, i) => ({ value: i, label: name }))

export function RealDataSpecimen() {
  const state = useFigureState({
    iris: row('Iris features', {
      fx: choice(IRIS_FEATURES, 2, { label: 'x' }),
      fy: choice(IRIS_FEATURES, 3, { label: 'y' }),
    }),
  })
  const { fx, fy } = state.iris
  const dims = useMemo<[number, number]>(() => [fx, fy], [fx, fy])
  const irisSeries = useDatasetSeries(IRIS, dims, 'label')
  const faithful = useDatasetSeries(FAITHFUL, undefined, 'none')
  const ix = useAxis({ label: irisSeries.xLabel })
  const iy = useAxis({ label: irisSeries.yLabel })
  const fxAxis = useAxis({ label: 'eruption (min)' })
  const fyAxis = useAxis({ label: 'waiting (min)' })
  const ax = useAxis({ label: 'x' })
  const ay = useAxis({ label: 'y' })
  const line = useMemo(() => ({ x: [3, 20], y: [4.5, 13] }), [])
  return (
    <>
      <Figure
        title="Iris and Old Faithful"
        purpose="Two embedded datasets with visible structure: petal length and width split the three Iris species, and Old Faithful's eruptions fall into a short-wait and a long-wait cluster."
        state={state}
        defaultSize="L"
        caption={`Left: Iris, two of its four measurements, coloured by species. Right: Old Faithful, eruption length against the wait before it. ${IRIS.meta.source}. ${FAITHFUL.meta.source}.`}
      >
        <Plots cols={2}>
          <Plot x={ix} y={iy}>
            <Points {...irisSeries.points} />
          </Plot>
          <Plot x={fxAxis} y={fyAxis}>
            <Points {...faithful.points} />
          </Plot>
        </Plots>
      </Figure>
      <Figure
        title="Anscombe's quartet"
        purpose="Four sets with the same means, variances, correlation and least-squares line y = 3 + x/2, and four different shapes: summary statistics do not describe a dataset."
        defaultSize="L"
        caption={`The dashed line is the least-squares fit, the same for all four. ${ANSCOMBE[0].meta.source}`}
      >
        <Plots rows={2} cols={2}>
          {ANSCOMBE.map((d) => (
            <Plot key={d.meta.name} x={ax} y={ay} title={d.meta.name} legend={false}>
              <Points name={d.meta.name} x={toFlat(d.x)} y={toFlat(d.y!)} slot={0} />
              <Curve name="y = 3 + x/2" x={line.x} y={line.y} dashed slot={1} />
            </Plot>
          ))}
        </Plots>
      </Figure>
    </>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 4. Regression functions.

const FUNCTIONS: RegressionFunction[] = ['sine', 'linear', 'cubic', 'step', 'sinc', 'bump', 'doppler']

export function RegressionSpecimen() {
  const state = useFigureState({
    fn: row('1 · function', { f: choice(FUNCTIONS, 'sine', { label: 'f' }) }),
    noise: row('2 · noise', {
      sd: slider(0, 1, 0.2, { label: 'noise sd' }),
      hetero: slider(0, 4, 2, { label: 'growth to the right' }),
    }),
  })
  const fn = state.fn.f
  const { sd, hetero } = state.noise
  const data = useMemo(
    () => regression1d(stream('regression'), { n: 120, fn, noise: sd, heteroscedastic: hetero }),
    [fn, sd, hetero],
  )
  const xs = toFlat(data.x)
  const x = useAxis({ label: 'x', hold: 'initial', key: fn })
  const y = useAxis({ label: 'y', hold: 'union', key: fn })
  return (
    <Figure
      title="Noisy regression functions"
      purpose="y = f(x) + ε, where the noise sd can grow along x (heteroscedastic): the scatter widens to the right, which a constant-variance model misreads."
      state={state}
      caption={`${data.meta.description} Axes hold while the noise moves; a new f refits them.`}
    >
      <Plot x={x} y={y}>
        <Points name="y" x={xs} y={toFlat(data.y!)} slot={0} />
        <Curve name="f(x)" x={xs} y={toFlat(data.f!)} emphasis />
      </Plot>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 5. Sequences: the casino and synthetic time series.

export function SequencesSpecimen() {
  const state = useFigureState({
    casino: row('casino switching', {
      toLoaded: slider(0.01, 0.3, 0.05, { label: 'P(fair → loaded)' }),
      toFair: slider(0.01, 0.3, 0.1, { label: 'P(loaded → fair)' }),
    }),
  })
  const { toLoaded, toFair } = state.casino
  const c = useMemo(() => casino(stream('casino'), { n: 300, toLoaded, toFair }), [toLoaded, toFair])
  const rolls = toFlat(c.x)
  const states = toFlat(c.z)
  const t = useMemo(() => rolls.map((_, i) => i), [rolls])
  const loaded = useMemo(() => states.map((s) => s * 6.5), [states])
  const ar = useMemo(() => arSeries(stream('ar'), { coefficients: [0.6, -0.3], n: 300 }), [])
  const seasonal = useMemo(
    () => seasonalSeries(stream('seasonal'), { n: 300, period: 24, amplitudes: [1, 0.4], persistence: 0.6 }),
    [],
  )
  const time = useAxis({ label: 't' })
  const face = useAxis({ label: 'face', range: [0.5, 6.5] })
  const value = useAxis({ label: 'value' })
  return (
    <Figure
      title="The occasionally dishonest casino and synthetic time series"
      purpose="A hidden state changes how the data look: sixes crowd into the loaded stretches (shaded) of the casino's rolls. Below, an AR(2) series and a seasonal series with trend."
      state={state}
      defaultSize="L"
      readouts={
        <>
          <Readout label="loaded share" value={(states.reduce((a, b) => a + b, 0) / states.length).toFixed(2)} />
          <Readout label="stationary loaded" value={(toLoaded / (toLoaded + toFair)).toFixed(2)} />
          <Readout label="sixes" value={rolls.filter((r) => r === 6).length} />
        </>
      }
      caption="Top: 300 rolls of a die that switches between fair and loaded, the hidden Markov model of Durbin et al.; the shading marks the loaded state. Bottom: aifn arSeries and seasonalSeries; every series is drawn from its own stream."
    >
      <Plots rows={2} hoverGroup>
        <Plot x={time} y={face}>
          <Area name="loaded" x={t} y={loaded} muted line={false} />
          <Points name="roll" x={t} y={rolls} slot={0} />
        </Plot>
        <Plot x={time} y={value}>
          <Curve name="AR(2)" x={toFlat(ar.t)} y={toFlat(ar.y)} slot={1} />
          <Curve name="trend + season + AR(1) noise" x={toFlat(seasonal.t)} y={toFlat(seasonal.y)} slot={2} />
        </Plot>
      </Plots>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 6. Recommendation data.

export function RecommendationSpecimen() {
  const state = useFigureState({
    exponent: slider(0.3, 2, 1, { label: 'Zipf exponent s' }),
    eta: slider(0, 2, 1, { label: 'position bias η' }),
  })
  const { exponent, eta } = state
  const cat = useMemo(() => zipfCatalogue(stream('zipf'), { items: 500, exponent, n: 20000 }), [exponent])
  const zipf = useMemo(() => {
    const counts = toFlat(cat.counts)
    const rank = counts.map((_, i) => i + 1)
    return {
      rank,
      w: toFlat(cat.weights),
      seenRank: rank.filter((_, i) => counts[i] > 0),
      share: counts.filter((c) => c > 0).map((c) => c / 20000),
    }
  }, [cat])
  const log = useMemo(() => clickLog(stream('clicks'), { sessions: 2000, eta }), [eta])
  const byRank = useMemo(() => {
    const r = toFlat(log.rank)
    const c = toFlat(log.clicked)
    const e = toFlat(log.examined)
    const clicks = new Array<number>(10).fill(0)
    const exams = new Array<number>(10).fill(0)
    r.forEach((k, i) => {
      clicks[k - 1] += c[i] / 2000
      exams[k - 1] += e[i] / 2000
    })
    const ranks = Array.from({ length: 10 }, (_, i) => i + 1)
    return { ranks, clicks, exams, model: ranks.map((k) => k ** -eta) }
  }, [log, eta])
  const rank = useAxis({ label: 'rank', log: true })
  const share = useAxis({ label: 'share', log: true, hold: 'union' })
  const k = useAxis({ label: 'rank k', range: [0.5, 10.5] })
  const rate = useAxis({ label: 'rate per session', range: [0, 1] })
  return (
    <Figure
      title="Zipf catalogues and click logs"
      purpose="Popularity falls as a power of rank (a straight line on log–log axes, slope −s), and position bias makes clicks fall with rank even for equally relevant items."
      state={state}
      defaultSize="L"
      caption="Left: 20,000 interactions with a 500-item catalogue whose popularity is p_i ∝ i^−s (aifn zipfCatalogue). Right: a click log of 2,000 sessions under the position-based model (aifn clickLog), where the rank-k result is examined with probability k^−η and clicked if examined and relevant; the logging ranker sorts ten items by noisy relevance."
    >
      <Plots cols={2}>
        <Plot x={rank} y={share}>
          <Points name="observed share" x={zipf.seenRank} y={zipf.share} slot={0} />
          <Curve name="p_i ∝ i^−s" x={zipf.rank} y={zipf.w} emphasis />
        </Plot>
        <Plot x={k} y={rate}>
          <Curve name="examined" x={byRank.ranks} y={byRank.exams} slot={1} showPoints />
          <Curve name="clicked" x={byRank.ranks} y={byRank.clicks} slot={2} showPoints />
          <Curve name="π(k) = k^−η" x={byRank.ranks} y={byRank.model} dashed slot={1} />
        </Plot>
      </Plots>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 7. Test images.

function imageRows(t: Tensor): number[][] {
  // Row 0 is the top of an image; the raster's first row is drawn at the bottom, so flip.
  return (toRows(t) as number[][]).slice().reverse()
}

export function ImagesSpecimen() {
  const state = useFigureState({
    which: choice(['shapes', 'checkerboard', 'gradient', 'digits'], 'shapes', { label: 'image' }),
    noise: slider(0, 0.5, 0, {
      label: 'noise sd',
      when: (v) => v.which === 'shapes' || v.which === 'digits',
    }),
  })
  const { which, noise } = state
  const img = useMemo(() => {
    if (which === 'shapes') return shapesImage({ noise, stream: stream('image') })
    if (which === 'checkerboard') return checkerboardImage({ size: 64, tile: 8 })
    if (which === 'gradient') return gradientImage({ size: 64, angle: Math.PI / 6 })
    // Ten noisy digits side by side, 7 rows × 60 columns (a blank column between glyphs).
    const d = digits(stream('digits'), { perClass: 1, noise, flip: 0.03 })
    const x = toFlat(d.x)
    const rows = Array.from({ length: 7 }, (_, r) =>
      Array.from({ length: 60 }, (_, c) => {
        const digit = Math.floor(c / 6)
        const col = c % 6
        return col === 5 ? 0 : x[digit * 35 + r * 5 + col]
      }),
    )
    return { rows }
  }, [which, noise])
  const grid = useMemo(() => {
    const z = 'rows' in img ? img.rows.slice().reverse() : imageRows(img)
    return {
      z,
      x: Array.from({ length: z[0].length }, (_, j) => j),
      y: Array.from({ length: z.length }, (_, i) => i),
    }
  }, [img])
  const x = useAxis({ label: 'column' })
  const y = useAxis({ label: 'row (from the bottom)', equal: x })
  return (
    <Figure
      title="Test images"
      purpose="Greyscale test images for the image operators: straight and curved edges, corners and texture in one picture, a checkerboard, a gradient, and noisy 5 × 7 digit glyphs."
      state={state}
      caption="aifn shapesImage, checkerboardImage, gradientImage and digits; row 0 of an image is its top."
    >
      <Plot x={x} y={y}>
        <Raster x={grid.x} y={grid.y} z={grid.z} colorBar={false} valueLabel="intensity" />
      </Plot>
    </Figure>
  )
}
