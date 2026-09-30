import { useMemo, useState } from 'react'
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
} from 'aifn-applied/data/synthetic'
import { anscombe, iris, oldFaithful } from 'aifn-applied/data/real'
import { type Dataset } from 'aifn-applied/data'
import { child, stream } from 'aifn/foundation/random'
import { toFlat, toRows, type Tensor } from 'aifn/foundation/tensor'
import { Button, Select, Slider } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { Heatmap, Panel, Readout, Subplots, XYChart, type XYSeries } from '@lab/viz'
import { useDatasetSeries } from '@lab/views'

// ---------------------------------------------------------------------------------------------------------------------
// 1. A gallery of 2-D generators.

type PointsId = 'moons' | 'circles' | 'rings' | 'spirals' | 'xor' | 'checkerboard' | 'blobs' | 'anisotropic'

const GENERATORS: Record<
  PointsId,
  { label: string; make: (seed: number, n: number, noise: number) => Dataset; noise: [number, number, number] }
> = {
  moons: {
    label: 'moons',
    make: (k, n, e) => moons(child(stream('gallery'), k), { n, noise: e }),
    noise: [0, 0.4, 0.1],
  },
  circles: {
    label: 'circles',
    make: (k, n, e) => circles(child(stream('gallery'), k), { n, noise: e }),
    noise: [0, 0.3, 0.05],
  },
  rings: {
    label: 'rings',
    make: (k, n, e) => rings(child(stream('gallery'), k), { n, noise: e }),
    noise: [0, 0.5, 0.1],
  },
  spirals: {
    label: 'spirals',
    make: (k, n, e) => spirals(child(stream('gallery'), k), { n, arms: 3, noise: e }),
    noise: [0, 0.2, 0.03],
  },
  xor: {
    label: 'XOR (Gaussian)',
    make: (k, n, e) => xor(child(stream('gallery'), k), { n, kind: 'gaussian', sd: e }),
    noise: [0.05, 1, 0.4],
  },
  checkerboard: {
    label: 'checkerboard',
    make: (k, n) => checkerboard(child(stream('gallery'), k), { n }),
    noise: [0, 0, 0],
  },
  blobs: { label: 'blobs', make: (k, n, e) => blobs(child(stream('gallery'), k), { n, sd: e }), noise: [0.2, 4, 1] },
  anisotropic: {
    label: 'anisotropic blobs',
    make: (k, n, e) => anisotropicBlobs(child(stream('gallery'), k), { n, sd: e }),
    noise: [0.2, 4, 1],
  },
}

export function GallerySpecimen() {
  const [id, setId] = useState<PointsId>('moons')
  const [n, setN] = useState(300)
  const [noises, setNoises] = useState<Partial<Record<PointsId, number>>>({})
  const [seed, setSeed] = useState(0)
  const g = GENERATORS[id]
  const noise = noises[id] ?? g.noise[2]
  const data = useMemo(() => g.make(seed, n, noise), [g, seed, n, noise])
  const { series, xLabel, yLabel } = useDatasetSeries(data, undefined, 'label')
  return (
    <Figure
      title="Seeded 2-D datasets"
      description="Every generator draws from a named stream, so the same seed gives the same points in a figure, a test and a replicate; the noise and the point count change only their own draws."
      defaultSize="L"
      controls={
        <>
          <ControlRow label="1 · generator">
            <Select
              label="dataset"
              value={id}
              onChange={setId}
              options={(Object.keys(GENERATORS) as PointsId[]).map((k) => ({ value: k, label: GENERATORS[k].label }))}
            />
            <Button variant="outline" size="sm" onClick={() => setSeed((s) => s + 1)}>
              new stream
            </Button>
          </ControlRow>
          <ControlRow label="2 · size and noise">
            <Slider label="points n" value={n} onChange={(v) => setN(Math.round(v))} min={20} max={1000} step={10} />
            {g.noise[1] > 0 && (
              <Slider
                label={id.includes('blob') || id === 'xor' ? 'cluster sd' : 'noise sd'}
                value={noise}
                onChange={(v) => setNoises((m) => ({ ...m, [id]: v }))}
                min={g.noise[0]}
                max={g.noise[1]}
              />
            )}
          </ControlRow>
        </>
      }
      readouts={
        <>
          <Readout label="stream" value={data.meta.key?.path ?? '—'} />
          <Readout label="x" value={data.x.shape.join(' × ')} />
          <Readout label="classes" value={data.meta.labelNames?.length ?? 0} />
        </>
      }
      caption={data.meta.description + (data.meta.source ? ` Recipe: ${data.meta.source}.` : '')}
    >
      <XYChart series={series} xLabel={xLabel} yLabel={yLabel} aspect="equal" axisKey={`${id}:${seed}`} />
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 2. Manifolds in three dimensions.

export function ManifoldSpecimen() {
  const [which, setWhich] = useState<'swiss' | 's'>('swiss')
  const data = useMemo(
    () => (which === 'swiss' ? swissRoll(stream('manifold'), { n: 800 }) : sCurve(stream('manifold'), { n: 800 })),
    [which],
  )
  const top = useDatasetSeries(data, [0, 2], 't')
  const side = useDatasetSeries(data, [0, 1], 't')
  return (
    <Figure
      title="The Swiss roll and the S-curve"
      description="Points on a 2-D sheet rolled or bent in three dimensions; colour is the coordinate t along the sheet, which a good embedding recovers as a straight axis."
      defaultSize="L"
      controls={
        <Select
          label="manifold"
          value={which}
          onChange={setWhich}
          options={[
            { value: 'swiss', label: 'Swiss roll' },
            { value: 's', label: 'S-curve' },
          ]}
        />
      }
      caption={`${data.meta.description} Left: seen from above (x, z); right: from the side (x, y).`}
    >
      <Subplots cols={2}>
        <Panel>
          <XYChart series={top.series} xLabel="x" yLabel="z" aspect="equal" axisKey={which} />
        </Panel>
        <Panel>
          <XYChart series={side.series} xLabel="x" yLabel="y" axisKey={which} />
        </Panel>
      </Subplots>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 3. Real data.

const IRIS = iris()
const FAITHFUL = oldFaithful()
const ANSCOMBE = anscombe()

export function RealDataSpecimen() {
  const [fx, setFx] = useState(2)
  const [fy, setFy] = useState(3)
  const dims = useMemo<[number, number]>(() => [fx, fy], [fx, fy])
  const irisSeries = useDatasetSeries(IRIS, dims, 'label')
  const faithful = useDatasetSeries(FAITHFUL, undefined, 'none')
  const features = IRIS.meta.featureNames.map((name, i) => ({ value: String(i), label: name }))
  return (
    <>
      <Figure
        title="Iris and Old Faithful"
        description="Two embedded datasets: Fisher's Iris (three species, four measurements) and the Old Faithful geyser, whose eruptions fall into two clusters of short and long waits."
        defaultSize="L"
        controls={
          <ControlRow label="Iris features">
            <Select label="x" value={String(fx)} onChange={(v) => setFx(Number(v))} options={features} />
            <Select label="y" value={String(fy)} onChange={(v) => setFy(Number(v))} options={features} />
          </ControlRow>
        }
        caption={`${IRIS.meta.source}. ${FAITHFUL.meta.source}.`}
      >
        <Subplots cols={2}>
          <Panel>
            <XYChart
              series={irisSeries.series}
              xLabel={irisSeries.xLabel}
              yLabel={irisSeries.yLabel}
              axisKey={`${fx}${fy}`}
            />
          </Panel>
          <Panel>
            <XYChart series={faithful.series} xLabel="eruption (min)" yLabel="waiting (min)" />
          </Panel>
        </Subplots>
      </Figure>
      <Figure
        title="Anscombe's quartet"
        description="Four sets with the same means, variances, correlation and least-squares line y = 3 + x/2, and four different shapes: summary statistics do not describe a dataset."
        defaultSize="L"
        caption={ANSCOMBE[0].meta.source}
      >
        <Subplots rows={2} cols={2} sharex sharey>
          {ANSCOMBE.map((d) => {
            const x = toFlat(d.x)
            const series: XYSeries[] = [
              { name: d.meta.name, type: 'scatter', x, y: toFlat(d.y!), slot: 0 },
              { name: 'y = 3 + x/2', type: 'line', x: [3, 20], y: [4.5, 13], dashed: true, slot: 1 },
            ]
            return (
              <Panel key={d.meta.name}>
                <XYChart series={series} xLabel="x" yLabel="y" />
              </Panel>
            )
          })}
        </Subplots>
      </Figure>
    </>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 4. Regression functions.

const FUNCTIONS: RegressionFunction[] = ['sine', 'linear', 'cubic', 'step', 'sinc', 'bump', 'doppler']

export function RegressionSpecimen() {
  const [fn, setFn] = useState<RegressionFunction>('sine')
  const [noise, setNoise] = useState(0.2)
  const [hetero, setHetero] = useState(0)
  const data = useMemo(
    () => regression1d(stream('regression'), { n: 120, fn, noise, heteroscedastic: hetero }),
    [fn, noise, hetero],
  )
  const x = toFlat(data.x)
  const series: XYSeries[] = [
    { name: 'y', type: 'scatter', x, y: toFlat(data.y!), slot: 0 },
    { name: 'f(x)', type: 'line', x, y: toFlat(data.f!), emphasis: true },
  ]
  return (
    <Figure
      title="Noisy regression functions"
      description="y = f(x) + ε with named test functions; the noise can grow along x (heteroscedastic), which a constant-variance model misreads."
      controls={
        <>
          <ControlRow label="1 · function">
            <Select label="f" value={fn} onChange={setFn} options={FUNCTIONS} />
          </ControlRow>
          <ControlRow label="2 · noise">
            <Slider label="noise sd" value={noise} onChange={setNoise} min={0} max={1} />
            <Slider label="growth to the right" value={hetero} onChange={setHetero} min={0} max={4} />
          </ControlRow>
        </>
      }
      caption={data.meta.description}
    >
      <XYChart series={series} xLabel="x" yLabel="y" axisKey={fn} rescaleOnChange={false} holdFit="union" />
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 5. Sequences: the casino and synthetic time series.

export function SequencesSpecimen() {
  const [toLoaded, setToLoaded] = useState(0.05)
  const [toFair, setToFair] = useState(0.1)
  const c = useMemo(() => casino(stream('casino'), { n: 300, toLoaded, toFair }), [toLoaded, toFair])
  const rolls = toFlat(c.x)
  const states = toFlat(c.z)
  const t = rolls.map((_, i) => i)
  const casinoSeries: XYSeries[] = [
    { name: 'loaded', type: 'area', x: t, y: states.map((s) => s * 6.5), muted: true },
    { name: 'roll', type: 'scatter', x: t, y: rolls, slot: 0 },
  ]
  const ar = useMemo(() => arSeries(stream('ar'), { coefficients: [0.6, -0.3], n: 300 }), [])
  const seasonal = useMemo(
    () => seasonalSeries(stream('seasonal'), { n: 300, period: 24, amplitudes: [1, 0.4], persistence: 0.6 }),
    [],
  )
  const ts: XYSeries[] = [
    { name: 'AR(2)', type: 'line', x: toFlat(ar.t), y: toFlat(ar.y), slot: 1 },
    { name: 'trend + season + AR(1) noise', type: 'line', x: toFlat(seasonal.t), y: toFlat(seasonal.y), slot: 2 },
  ]
  return (
    <Figure
      title="The occasionally dishonest casino and synthetic time series"
      description="Top: rolls of a die that switches between fair and loaded (shaded), the hidden Markov model of Durbin et al.; bottom: an AR(2) series and a seasonal series with trend."
      defaultSize="L"
      controls={
        <ControlRow label="casino switching">
          <Slider label="P(fair → loaded)" value={toLoaded} onChange={setToLoaded} min={0.01} max={0.3} />
          <Slider label="P(loaded → fair)" value={toFair} onChange={setToFair} min={0.01} max={0.3} />
        </ControlRow>
      }
      readouts={
        <>
          <Readout label="loaded share" value={(states.reduce((a, b) => a + b, 0) / states.length).toFixed(2)} />
          <Readout label="stationary loaded" value={(toLoaded / (toLoaded + toFair)).toFixed(2)} />
          <Readout label="sixes" value={rolls.filter((r) => r === 6).length} />
        </>
      }
      caption="aifn/datasets casino, arSeries and seasonalSeries; every series is drawn from its own stream."
    >
      <Subplots rows={2} sharex heightRatios={[1, 1]}>
        <Panel>
          <XYChart series={casinoSeries} xLabel="roll" yLabel="face" yRange={[0.5, 6.5]} />
        </Panel>
        <Panel>
          <XYChart series={ts} xLabel="t" yLabel="value" />
        </Panel>
      </Subplots>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 6. Recommendation data.

export function RecommendationSpecimen() {
  const [exponent, setExponent] = useState(1)
  const [eta, setEta] = useState(1)
  const cat = useMemo(() => zipfCatalogue(stream('zipf'), { items: 500, exponent, n: 20000 }), [exponent])
  const counts = toFlat(cat.counts)
  const w = toFlat(cat.weights)
  const rank = counts.map((_, i) => i + 1)
  const zipf: XYSeries[] = [
    {
      name: 'observed share',
      type: 'scatter',
      x: rank.filter((_, i) => counts[i] > 0),
      y: counts.filter((c) => c > 0).map((c) => c / 20000),
      slot: 0,
    },
    { name: 'p_i ∝ i^−s', type: 'line', x: rank, y: w, emphasis: true },
  ]
  const log = useMemo(() => clickLog(stream('clicks'), { sessions: 2000, eta }), [eta])
  const byRank = useMemo(() => {
    const r = toFlat(log.rank)
    const c = toFlat(log.clicked)
    const e = toFlat(log.examined)
    const clicks = new Array(10).fill(0)
    const exams = new Array(10).fill(0)
    r.forEach((k, i) => {
      clicks[k - 1] += c[i] / 2000
      exams[k - 1] += e[i] / 2000
    })
    return { clicks, exams }
  }, [log])
  const ranks = Array.from({ length: 10 }, (_, i) => i + 1)
  const clicks: XYSeries[] = [
    { name: 'examined', type: 'line', x: ranks, y: byRank.exams, slot: 1, showPoints: true },
    { name: 'clicked', type: 'line', x: ranks, y: byRank.clicks, slot: 2, showPoints: true },
    { name: 'π(k) = k^−η', type: 'line', x: ranks, y: ranks.map((k) => k ** -eta), dashed: true, slot: 1 },
  ]
  return (
    <Figure
      title="Zipf catalogues and click logs"
      description="Left: 20,000 interactions with a catalogue whose popularity falls as a power of rank, on log–log axes; right: a click log under the position-based model, where rank-k results are examined with probability k^−η."
      defaultSize="L"
      controls={
        <ControlRow label="parameters">
          <Slider label="Zipf exponent s" value={exponent} onChange={setExponent} min={0.3} max={2} />
          <Slider label="position bias η" value={eta} onChange={setEta} min={0} max={2} />
        </ControlRow>
      }
      caption="aifn/datasets zipfCatalogue and clickLog (the logging ranker sorts ten items by noisy relevance)."
    >
      <Subplots cols={2}>
        <Panel>
          <XYChart series={zipf} xLabel="rank" yLabel="share" xLog yLog />
        </Panel>
        <Panel>
          <XYChart series={clicks} xLabel="rank k" yLabel="rate per session" yRange={[0, 1]} />
        </Panel>
      </Subplots>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 7. Test images.

function imageRows(t: Tensor): number[][] {
  // Row 0 is the top of an image; the heatmap's first row is drawn at the bottom, so flip.
  return (toRows(t) as number[][]).slice().reverse()
}

export function ImagesSpecimen() {
  const [which, setWhich] = useState<'shapes' | 'checkerboard' | 'gradient' | 'digits'>('shapes')
  const [noise, setNoise] = useState(0)
  const img = useMemo(() => {
    if (which === 'shapes') return shapesImage({ noise, stream: stream('image') })
    if (which === 'checkerboard') return checkerboardImage(64, 8)
    if (which === 'gradient') return gradientImage(64, { angle: Math.PI / 6 })
    // Ten noisy digits side by side, 7 rows × 50 columns.
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
  const z = 'rows' in img ? img.rows.slice().reverse() : imageRows(img)
  const h = z.length
  const w = z[0].length
  return (
    <Figure
      title="Test images"
      description="Greyscale test images for the image operators: straight and curved edges, corners and texture in one picture, a checkerboard, a gradient, and noisy 5 × 7 digit glyphs."
      controls={
        <ControlRow label="image">
          <Select
            label="image"
            value={which}
            onChange={setWhich}
            options={['shapes', 'checkerboard', 'gradient', 'digits']}
          />
          {(which === 'shapes' || which === 'digits') && (
            <Slider label="noise sd" value={noise} onChange={setNoise} min={0} max={0.5} />
          )}
        </ControlRow>
      }
      caption="aifn/datasets shapesImage, checkerboardImage, gradientImage and digits; row 0 of an image is its top."
    >
      <Heatmap
        x={Array.from({ length: w }, (_, j) => j)}
        y={Array.from({ length: h }, (_, i) => i)}
        z={z}
        xLabel="column"
        yLabel="row (from the bottom)"
        equalAspect
        colorBar={false}
      />
    </Figure>
  )
}
