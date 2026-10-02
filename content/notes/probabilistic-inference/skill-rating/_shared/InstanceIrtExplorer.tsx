import { useMemo, useState } from 'react'
import { Interactive, ParamChoice, ParamSlider, Readout, XYChart, formatNumber, type XYSeries } from 'aifn-render'
import { linspace } from '@/lib/math'
import { iccBetaThree, iccTwoPL, runExperiment } from './instanceIrt'

type Model = '2PL' | 'beta3'
type Colour = 'difficulty' | 'discrimination' | 'label'

const FEATURE_X: [number, number] = [-4.5, 4.5]
const FEATURE_Y: [number, number] = [-2.5, 2.5]
const THETA_2PL = linspace(-3.5, 3.5, 141)
const THETA_B3 = linspace(0.005, 0.995, 199)

/** Tertile bands of a list of values: 0 = lowest third, 1 = middle, 2 = highest. */
function tertiles(v: number[]): number[] {
  const s = [...v].sort((a, b) => a - b)
  const lo = s[Math.floor(s.length / 3)]
  const hi = s[Math.floor((2 * s.length) / 3)]
  return v.map((x) => (x < lo ? 0 : x < hi ? 1 : 2))
}

const BAND_NAMES: Record<Colour, string[]> = {
  difficulty: ['easiest third', 'middle third', 'hardest third'],
  discrimination: ['negative (a < 0)', 'low (0 ≤ a < 1)', 'high (a ≥ 1)'],
  label: ['given class 0', 'given class 1', 'flipped label'],
}

export function InstanceIrtExplorer({ initialModel = '2PL' }: { initialModel?: Model }) {
  const [model, setModel] = useState<Model>(initialModel)
  const [colour, setColour] = useState<Colour>('discrimination')
  const [noise, setNoise] = useState(0.15)
  const [seed, setSeed] = useState(1)
  const [picked, setPicked] = useState<number | null>(null)

  const run = useMemo(() => runExperiment(seed, noise), [seed, noise])
  const { data, resp, hardness, twoPL, betaThree } = run
  const n = data.test.x.length

  const a = model === '2PL' ? twoPL.a : betaThree.a
  const difficulty = model === '2PL' ? twoPL.b : betaThree.delta
  const ability = model === '2PL' ? twoPL.theta : betaThree.theta

  // Default to the first flipped instance, so the figure opens on a mislabelled point.
  const firstFlipped = data.test.flipped.indexOf(true)
  const sel = picked ?? (firstFlipped >= 0 ? firstFlipped : 0)

  const feature = useMemo<XYSeries[]>(() => {
    const group =
      colour === 'label'
        ? data.test.y.map((y, j) => (data.test.flipped[j] ? 2 : y))
        : colour === 'difficulty'
          ? tertiles(difficulty)
          : a.map((v) => (v < 0 ? 0 : v < 1 ? 1 : 2))
    return [
      {
        name: 'instances',
        type: 'scatter',
        x: data.test.x.map((p) => p[0]),
        y: data.test.x.map((p) => p[1]),
        group,
        groupNames: BAND_NAMES[colour],
      },
      { name: 'selected', type: 'scatter', x: [data.test.x[sel][0]], y: [data.test.x[sel][1]], emphasis: true },
    ]
  }, [data, colour, difficulty, a, sel])

  const icc = useMemo<XYSeries[]>(() => {
    const grid = model === '2PL' ? THETA_2PL : THETA_B3
    const curve =
      model === '2PL'
        ? grid.map((t) => iccTwoPL(twoPL.a[sel], twoPL.c[sel], t))
        : grid.map((t) => iccBetaThree(t, betaThree.delta[sel], betaThree.a[sel]))
    const responses = model === '2PL' ? resp.correct.map((row) => row[sel]) : resp.probs.map((row) => row[sel])
    return [
      { name: 'classifiers', type: 'scatter', x: ability, y: responses, slot: 0 },
      { name: 'item characteristic curve', type: 'line', x: grid, y: curve, emphasis: true },
    ]
  }, [model, twoPL, betaThree, resp, ability, sel])

  const flagged = a.map((v) => v < 0)
  const nFlagged = flagged.filter(Boolean).length
  const nFlipped = data.test.flipped.filter(Boolean).length
  const caught = flagged.filter((f, j) => f && data.test.flipped[j]).length
  const againstBayes = flagged.filter((f, j) => f && data.test.bayes[j] < 0.5).length

  const pick = ([x, y]: [number, number]) => {
    let best = 0
    let bestD = Infinity
    data.test.x.forEach((p, j) => {
      const d = (p[0] - x) ** 2 + (p[1] - y) ** 2
      if (d < bestD) {
        bestD = d
        best = j
      }
    })
    setPicked(best)
  }

  return (
    <Interactive
      title="Instances as test items, classifiers as respondents"
      caption={
        <>
          Two Gaussian classes; a fraction of the 70 test labels is flipped. Seventeen classifiers (QDA, LDA, naive
          Bayes, nearest centroid, k-NN and stumps on training sets of different sizes, QDA through input noise, and two
          random guessers) are trained on clean data and score every test instance. The 2PL is fitted to
          correct/incorrect; β³ to each classifier&apos;s probability of the given label (clipped to [0.01, 0.99]).
          Click an instance to see its item characteristic curve and every classifier&apos;s response at its fitted
          ability. Flipped labels sit in the negative-discrimination band; so do the few clean instances that lie on the
          other class&apos;s side of the Bayes boundary, which no classifier can tell from a flip.
        </>
      }
      controls={
        <>
          <ParamChoice
            label="IRT model"
            value={model}
            onChange={setModel}
            options={[
              { value: '2PL', label: '2PL' },
              { value: 'beta3', label: 'β³' },
            ]}
          />
          <ParamChoice
            label="colour by"
            value={colour}
            onChange={setColour}
            options={[
              { value: 'discrimination', label: 'discrimination' },
              { value: 'difficulty', label: 'difficulty' },
              { value: 'label', label: 'given label' },
            ]}
          />
          <ParamSlider
            label="label noise rate"
            value={noise}
            onChange={setNoise}
            min={0}
            max={0.3}
            step={0.05}
            debounceMs={150}
          />
          <ParamSlider label="seed" value={seed} onChange={setSeed} min={1} max={20} step={1} debounceMs={150} />
        </>
      }
      readout={
        <>
          <Readout
            label={`instance ${sel + 1}:`}
            value={`label ${data.test.y[sel]}${data.test.flipped[sel] ? ' (flipped)' : ''}`}
          />
          <Readout label="Bayes P(given label)" value={formatNumber(data.test.bayes[sel])} />
          <Readout label="instance hardness" value={formatNumber(hardness[sel])} />
          <Readout label="discrimination a" value={formatNumber(a[sel])} />
          <Readout label={model === '2PL' ? 'difficulty b' : 'difficulty δ'} value={formatNumber(difficulty[sel])} />
          <Readout
            label="a < 0:"
            value={`${nFlagged} flagged, ${caught} of ${nFlipped} flips, ${againstBayes} against the Bayes rule`}
          />
        </>
      }
    >
      <div className="space-y-2">
        <XYChart
          series={feature}
          xLabel="x₁"
          yLabel="x₂"
          xRange={FEATURE_X}
          yRange={FEATURE_Y}
          equalAspect
          onPlotClick={pick}
          ariaLabel="Test instances in feature space, coloured by fitted IRT parameter"
        />
        <XYChart
          series={icc}
          xLabel={model === '2PL' ? 'ability θ' : 'ability θ ∈ (0, 1)'}
          yLabel={model === '2PL' ? 'correct' : 'P(given label)'}
          xRange={model === '2PL' ? [-3.5, 3.5] : [0, 1]}
          yRange={[0, 1]}
          height={300}
          ariaLabel="Item characteristic curve of the selected instance with classifier responses"
        />
      </div>
      <p className="mt-2 text-xs text-muted-foreground">
        {n} test instances, {resp.names.length} classifiers. Best accuracy on the given labels:{' '}
        {formatNumber(Math.max(...resp.accuracy))}.
      </p>
    </Interactive>
  )
}
