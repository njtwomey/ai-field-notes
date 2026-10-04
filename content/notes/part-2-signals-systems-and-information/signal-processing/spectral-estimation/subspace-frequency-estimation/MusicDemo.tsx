import { useMemo } from 'react'
import { Figure, int, Plot, Readout, seriesLayers, type SeriesSpec, useAxis, useFigureState } from 'aifn-render'
import { periodogram, powerDb, symmetricEigen } from '../_shared/spectra'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { normal, stream, uniform } from 'aifn/foundation/random'

const GRID = toFlat(linspace(0, Math.PI, 2001))
const CENTRE = 0.3 * Math.PI

/** Local maxima of y, largest first. */
function topPeaks(y: number[], count: number): number[] {
  const peaks: number[] = []
  for (let i = 1; i < y.length - 1; i++) if (y[i] > y[i - 1] && y[i] >= y[i + 1]) peaks.push(i)
  return peaks.sort((a, b) => y[b] - y[a]).slice(0, count)
}

/**
 * Two sinusoids closer than the Fourier resolution limit. MUSIC projects steering vectors onto the noise subspace of
 * the sample correlation matrix; the pseudospectrum peaks where a steering vector is orthogonal to it.
 */
export function MusicDemo() {
  const state = useFigureState({
    sep: int(0.02, { min: 0.005, max: 0.1, step: 0.005, label: 'separation (×π rad/sample)' }),
    snr: int(20, { min: 0, max: 40, step: 1, label: 'SNR per sinusoid (dB)', format: (v) => `${v} dB` }),
    n: int(64, { min: 32, max: 256, step: 16, label: 'samples N', format: (v) => String(v) }),
    seed: int(3, { min: 1, max: 30, step: 1, label: 'seed', format: (v) => String(v) }),
  })

  const r = useMemo(() => {
    const g = stream(state.seed)
    const freqs = [CENTRE - (state.sep * Math.PI) / 2, CENTRE + (state.sep * Math.PI) / 2]
    const phases = freqs.map(() => 2 * Math.PI * uniform(g))
    const sigma = Math.sqrt(0.5 / 10 ** (state.snr / 10))
    const x = Array.from({ length: state.n }, (_, t) =>
      freqs.reduce((s, w, k) => s + Math.cos(w * t + phases[k]), sigma * normal(g)),
    )
    // Forward–backward averaged M × M sample correlation matrix of overlapping snapshots.
    const m = Math.min(24, Math.floor(state.n / 2))
    const snapshots = state.n - m + 1
    const R = Array.from({ length: m }, () => new Array<number>(m).fill(0))
    for (let s = 0; s < snapshots; s++)
      for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) R[i][j] += (x[s + i] * x[s + j]) / snapshots
    const fb = R.map((row, i) => row.map((v, j) => (v + R[m - 1 - i][m - 1 - j]) / 2))
    const { values, vectors } = symmetricEigen(fb)
    // Two real sinusoids are four complex exponentials: a four-dimensional signal subspace.
    const noise = vectors.slice(4)
    const music = GRID.map((w) => {
      let denom = 0
      for (const e of noise) {
        let re = 0
        let im = 0
        for (let k = 0; k < m; k++) {
          re += e[k] * Math.cos(w * k)
          im += e[k] * Math.sin(w * k)
        }
        denom += re * re + im * im
      }
      return 1 / denom
    })
    const maxMusic = Math.max(...music)
    const p = periodogram(x, 'hann', 4096)
    const maxP = Math.max(...p.psd)
    const est = topPeaks(music, 2)
      .map((i) => GRID[i] / Math.PI)
      .sort((a, b) => a - b)
    return {
      music: music.map((v) => 10 * Math.log10(v / maxMusic)),
      pOmega: p.omega.map((w) => w / Math.PI),
      pDb: p.psd.map((v) => powerDb(v / maxP)),
      freqs: freqs.map((w) => w / Math.PI),
      est,
      eig: values.slice(0, 6),
    }
  }, [state.sep, state.snr, state.n, state.seed])

  const series: SeriesSpec[] = [
    { name: 'periodogram (Hann)', type: 'line', x: r.pOmega, y: r.pDb, muted: true },
    { name: 'MUSIC pseudospectrum', type: 'line', x: GRID.map((w) => w / Math.PI), y: r.music, slot: 0 },
    ...r.freqs.map((f): SeriesSpec => ({
      name: 'true frequencies',
      type: 'line',
      x: [f, f],
      y: [-60, 5],
      slot: 1,
      dashed: true,
    })),
  ]

  const xAxis = useAxis({ label: 'ω / π', range: [0.15, 0.45] })
  const yAxis = useAxis({ label: 'normalised level (dB)', range: [-60, 5] })
  return (
    <Figure
      title="Resolving sinusoids below the Fourier limit"
      state={state}
      caption="Two unit-amplitude sinusoids centred on 0.3π rad/sample, in white noise. The periodogram (grey) cannot separate them when they are closer than about 2π/N. MUSIC estimates the correlation matrix of 24-sample snapshots, splits its eigenvectors into a four-dimensional signal subspace and a noise subspace, and peaks where the steering vector (1, e^{iω}, …) is orthogonal to the noise subspace. Its resolution improves with SNR, not only with N."

      readouts={
        <>
          <Readout label="true (×π)" value={r.freqs.map((f) => f.toFixed(4)).join(', ')} />
          <Readout label="MUSIC estimates (×π)" value={r.est.map((f) => f.toFixed(4)).join(', ')} />
          <Readout label="Fourier limit 2/N (×π)" value={(2 / state.n).toFixed(4)} />
          <Readout label="largest eigenvalues" value={r.eig.map((v) => v.toFixed(3)).join(', ')} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={320}>
        {seriesLayers(series)}
      </Plot>
    </Figure>
  )
}
