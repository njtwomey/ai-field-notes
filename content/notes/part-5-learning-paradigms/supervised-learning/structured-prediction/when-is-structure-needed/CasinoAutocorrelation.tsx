import { useMemo } from 'react'
import { Curve, Figure, formatNumber, int, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { stream, uniform } from 'aifn/foundation/random'

const LAGS = Array.from({ length: 30 }, (_, i) => i + 1)
const ROLLS = 20000
const FACES = [1, 2, 3, 4, 5, 6]

/**
 * Auto-correlation of the faces rolled in the occasionally dishonest casino: exact, from the two-state switching model,
 * against the sample auto-correlation of one simulated run. It decays geometrically at rate 1 − 2pₛ.
 */
export function CasinoAutocorrelation() {
  const state = useFigureState({
    switchP: slider(0.01, 0.5, 0.05, { step: 0.01, label: 'switching probability pₛ' }),
    loadedP: slider(0.17, 0.95, 0.5, { step: 0.01, label: 'probability of the loaded face' }),
    face: int(1, { min: 1, max: 6, step: 1, label: 'loaded face', format: (v) => String(v) }),
  })

  const r = useMemo(() => {
    const v = state.face
    const pv = state.loadedP
    const loaded = FACES.map((f) => (f === v ? pv : (1 - pv) / 5))
    const meanF = 3.5
    const meanL = FACES.reduce((s, f, i) => s + f * loaded[i], 0)
    const varF = 35 / 12
    const varL = FACES.reduce((s, f, i) => s + f * f * loaded[i], 0) - meanL * meanL
    // Faces are conditionally independent given the dice, so only the switching of the mean correlates them.
    const between = (meanF - meanL) ** 2 / 4
    const ratio = between / ((varF + varL) / 2 + between)
    const decay = 1 - 2 * state.switchP
    const exact = LAGS.map((n) => ratio * decay ** n)

    // One simulated run of the casino, for the sample auto-correlation.
    const g = stream(3)
    const rolls: number[] = []
    let isLoaded = false
    const cumulative = loaded.map((_, i) => loaded.slice(0, i + 1).reduce((a, b) => a + b, 0))
    for (let t = 0; t < ROLLS; t++) {
      if (t > 0 && uniform(g) < state.switchP) isLoaded = !isLoaded
      const u = uniform(g)
      rolls.push(isLoaded ? (FACES[cumulative.findIndex((c) => u < c)] ?? 6) : 1 + Math.floor(6 * u))
    }
    const mean = rolls.reduce((a, b) => a + b, 0) / ROLLS
    const centred = rolls.map((x) => x - mean)
    const variance = centred.reduce((a, b) => a + b * b, 0) / ROLLS
    const sample = LAGS.map((n) => {
      let s = 0
      for (let t = 0; t + n < ROLLS; t++) s += centred[t] * centred[t + n]
      return s / (ROLLS - n) / variance
    })
    const window = LAGS.find((n) => exact[n - 1] < 0.01)
    return { exact, sample, ratio, decay, window }
  }, [state.switchP, state.loadedP, state.face])

  const series = [
    { name: 'exact', x: LAGS, y: r.exact, slot: 0 },
    { name: 'one simulated run', x: LAGS, y: r.sample, slot: 1 },
  ] as const

  const xAxis = useAxis({ label: 'lag n', range: [1, 30] })
  const yAxis = useAxis({ label: 'auto-correlation', hold: 'union' })
  return (
    <Figure
      title="How fast the casino forgets"
      state={state}
      caption="The auto-correlation of the rolled faces at lag n, for a casino that switches between a fair die and one loaded towards a chosen face. The faces are independent given which die is in use, so all of their correlation comes from the die persisting: it is a fixed fraction of the die's own correlation (1 − 2pₛ)ⁿ. Rare switching gives slow decay and long-range dependence; frequent switching gives almost none. A loaded face near the middle, 3 or 4, barely changes the mean face, so the linear correlation is small even when the die persists."

      readouts={
        <>
          <Readout label="lag-1 correlation" value={formatNumber(r.exact[0])} />
          <Readout label="decay per lag 1 − 2pₛ" value={formatNumber(r.decay)} />
          <Readout label="first lag below 0.01" value={r.window ?? '> 30'} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        <Curve {...series[0]} />
        <Points {...series[1]} />
      </Plot>
    </Figure>
  )
}
