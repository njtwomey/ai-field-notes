import { useContext, useMemo, type ReactNode } from 'react'
import {
  Annotation,
  Area,
  Bars,
  choice,
  Curve,
  Figure,
  formatNumber,
  FrameContext,
  int,
  Plot,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { sampleAcf, standardDeviation } from 'aifn-compute/probability/stats'
import { slidingMeanStd } from 'aifn-compute/signal/similarity'
import { periodogram } from 'aifn-compute/signal/spectral'
import { arSeries, randomWalk, seasonalSeries } from 'aifn-methods/data/synthetic'

const N = 240
const MAX_LAG = 36
const PANEL = 210

const KINDS = [
  { value: 'seasonal', label: 'season of period 12 with a trend' },
  { value: 'persistent', label: 'AR(1), φ = 0.9' },
  { value: 'alternating', label: 'AR(1), φ = −0.6' },
  { value: 'walk', label: 'random walk' },
  { value: 'noise', label: 'white noise' },
] as const
type Kind = (typeof KINDS)[number]['value']

const GENERATE: Record<Kind, (seed: string) => number[]> = {
  seasonal: (seed) =>
    toFlat(seasonalSeries(stream(seed), { n: N, period: 12, amplitudes: [1.5], trend: 0.01, noise: 0.6 }).y),
  persistent: (seed) => toFlat(arSeries(stream(seed), { n: N, coefficients: [0.9] }).y),
  alternating: (seed) => toFlat(arSeries(stream(seed), { n: N, coefficients: [-0.6] }).y),
  walk: (seed) => toFlat(randomWalk(stream(seed), { n: N }).y),
  noise: (seed) => toFlat(arSeries(stream(seed), { n: N, coefficients: [0] }).y),
}

/** Plots inside a Figure take the frame's height; this gives the plots inside it a fixed height instead. */
function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}

/** One series and three families of features computed from it: rolling statistics, autocorrelations and a spectrum. */
export function SeriesFeatures() {
  const state = useFigureState({
    kind: choice(KINDS, 'seasonal', { label: 'series' }),
    window: int(12, { ge: 2, le: 60, suggestions: [6, 12, 24], label: 'rolling window m' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
  })
  const { kind, window, seed } = state
  const view = useMemo(() => {
    const y = GENERATE[kind](`series-features/${kind}/${seed}`)
    const t = y.map((_, i) => i)
    // Trailing statistics: the value at t summarises y[t - m + 1 .. t], so it uses no future values.
    const { mean, std } = slidingMeanStd(y, window)
    const m = [...Array<number>(window - 1).fill(NaN), ...toFlat(mean)]
    const s = [...Array<number>(window - 1).fill(NaN), ...toFlat(std)]
    const acf = sampleAcf(y, MAX_LAG)
    const rho = toFlat(acf.acf)
    const spectrum = periodogram(y)
    const f = toFlat(spectrum.f).slice(1)
    const power = toFlat(spectrum.values).slice(1)
    const peak = f[power.indexOf(Math.max(...power))]
    return {
      t,
      y,
      m,
      lower: m.map((v, i) => v - s[i]),
      upper: m.map((v, i) => v + s[i]),
      lags: rho.map((_, k) => k),
      rho,
      band: acf.band,
      f,
      power,
      peak,
      sd: standardDeviation(y),
    }
  }, [kind, window, seed])
  const tAxis = useAxis({ label: 'time t', range: [0, N] })
  const yAxis = useAxis({ label: 'y', key: kind })
  const kAxis = useAxis({ label: 'lag k', range: [-0.5, MAX_LAG + 0.5] })
  const rAxis = useAxis({ label: 'autocorrelation', range: [-1, 1] })
  const fAxis = useAxis({ label: 'frequency (cycles per step)', range: [0, 0.5] })
  const pAxis = useAxis({ label: 'power', key: kind })
  return (
    <Figure
      title="Features of a series"
      state={state}
      hoverReadout={false}
      caption="Top: the series with its trailing rolling mean and a band of one rolling standard deviation over the last m steps. Bottom left: the sample autocorrelation at lags 0 to 36, with the band outside which a lag's autocorrelation is significant for white noise. Bottom right: the periodogram. A season shows as a peak in the autocorrelation at its period and a spike in the spectrum at its frequency; a persistent series has slowly decaying autocorrelations and power at low frequencies; an alternating series has autocorrelations that change sign and power at high frequencies; white noise has neither. The readouts are the kind of numbers a feature vector holds."
      readouts={
        <>
          <Readout label="lag-1 autocorrelation" value={formatNumber(view.rho[1])} />
          <Readout label="dominant period" value={`${formatNumber(1 / view.peak)} steps`} />
          <Readout label="standard deviation" value={formatNumber(view.sd)} />
        </>
      }
    >
      <FixedHeight height={PANEL}>
        <Plot x={tAxis} y={yAxis} title="series and rolling statistics">
          <Area
            name="rolling mean ± sd"
            x={view.t}
            y={view.upper}
            base={view.lower}
            slot={1}
            opacity={0.2}
            line={false}
          />
          <Curve name="series" x={view.t} y={view.y} slot={0} width={1} />
          <Curve name="rolling mean" x={view.t} y={view.m} slot={1} />
        </Plot>
      </FixedHeight>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <FixedHeight height={PANEL}>
          <Plot x={kAxis} y={rAxis} title="autocorrelation">
            <Bars name="autocorrelation" x={view.lags} y={view.rho} slot={2} width={0.6} />
            <Annotation y={view.band} dashed />
            <Annotation y={-view.band} dashed />
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={fAxis} y={pAxis} title="periodogram">
            <Curve name="power" x={view.f} y={view.power} slot={3} />
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
