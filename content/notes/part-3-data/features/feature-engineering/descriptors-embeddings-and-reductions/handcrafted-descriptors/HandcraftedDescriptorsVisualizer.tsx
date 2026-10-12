import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { normal, stream } from 'aifn-compute/foundation/random'

export function HandcraftedDescriptorsVisualizer() {
  const state = useFigureState({
    signalType: choice<'harmonic' | 'transient_shock' | 'chaotic_drift'>(
      [
        { value: 'harmonic', label: 'Periodic harmonic vibration (motor/bearing)' },
        { value: 'transient_shock', label: 'Transient shock impulse (impact/fault)' },
        { value: 'chaotic_drift', label: 'Stochastic random walk (drift/wear)' },
      ],
      'harmonic',
      { label: 'signal source' },
    ),
    noiseLevel: slider(0.0, 0.5, 0.1, { step: 0.05, label: 'additive noise sigma' }),
  })

  const results = useMemo(() => {
    const rng = stream(99)
    const N = 80
    const tValues: number[] = []
    const signalValues: number[] = []

    for (let i = 0; i < N; i++) {
      const t = (i / (N - 1)) * 4 * Math.PI
      let val = 0

      if (state.signalType === 'harmonic') {
        val = Math.sin(2.0 * t) + 0.3 * Math.sin(4.0 * t)
      } else if (state.signalType === 'transient_shock') {
        const dt = t - 1.5 * Math.PI
        val = 2.5 * Math.exp(-0.8 * dt * dt) * Math.cos(5.0 * dt)
      } else {
        const prev = signalValues.length > 0 ? signalValues[signalValues.length - 1] : 0
        val = 0.85 * prev + normal(rng, 0, 0.4)
      }

      const noisyVal = val + normal(rng, 0, state.noiseLevel)
      tValues.push(i)
      signalValues.push(noisyVal)
    }

    const mean = signalValues.reduce((a, b) => a + b, 0) / N
    let variance = 0
    let m3 = 0
    let m4 = 0
    let zeroCrossings = 0
    let minVal = Infinity
    let maxVal = -Infinity

    for (let i = 0; i < N; i++) {
      const d = signalValues[i] - mean
      variance += d * d
      m3 += d * d * d
      m4 += d * d * d * d
      if (signalValues[i] < minVal) minVal = signalValues[i]
      if (signalValues[i] > maxVal) maxVal = signalValues[i]

      if (i > 0 && Math.sign(signalValues[i]) !== Math.sign(signalValues[i - 1])) {
        zeroCrossings++
      }
    }

    variance /= N
    const std = Math.sqrt(variance)
    const skewness = std > 1e-5 ? m3 / N / std ** 3 : 0
    const kurtosis = std > 1e-5 ? m4 / N / std ** 4 - 3 : 0
    const peakToPeak = maxVal - minVal

    let autoCorrNum = 0
    for (let i = 0; i < N - 1; i++) {
      autoCorrNum += (signalValues[i] - mean) * (signalValues[i + 1] - mean)
    }
    const lag1AutoCorr = variance > 1e-5 ? autoCorrNum / (N - 1) / variance : 0

    return {
      tValues,
      signalValues,
      mean,
      std,
      skewness,
      kurtosis,
      zeroCrossings,
      peakToPeak,
      lag1AutoCorr,
    }
  }, [state.signalType, state.noiseLevel])

  const xAxis = useAxis({
    label: 'discrete time sample index t',
    range: [0, 80],
  })
  const yAxis = useAxis({
    label: 'sensor waveform amplitude',
    range: [-3.0, 3.0],
  })

  return (
    <Figure
      title="Deterministic hand-built statistical and dynamical signal descriptors"
      state={state}
      hoverReadout={false}
      caption="Interactive calculation of hand-built time-series descriptors (statistical moments, zero-crossing rate, peak-to-peak amplitude, and lag-1 autocorrelation). These deterministic transforms reduce high-dimensional raw telemetry into lightweight, interpretable features that separate signal dynamics without requiring neural training."
      readouts={
        <>
          <Readout label="standard deviation (scale)" value={results.std.toFixed(2)} />
          <Readout label="skewness (asymmetry)" value={results.skewness.toFixed(2)} />
          <Readout label="excess kurtosis (peakedness)" value={results.kurtosis.toFixed(2)} />
          <Readout label="zero-crossing count" value={`${results.zeroCrossings} crossings`} />
          <Readout label="peak-to-peak amplitude" value={results.peakToPeak.toFixed(2)} />
          <Readout label="lag-1 autocorrelation" value={results.lag1AutoCorr.toFixed(2)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} title="Raw continuous signal and discrete sample points">
        <Curve
          name="continuous waveform interpolation"
          x={results.tValues}
          y={results.signalValues}
          slot={0}
          width={2}
        />
        <Points name="discrete telemetry observations" x={results.tValues} y={results.signalValues} slot={0} size={3} />
      </Plot>
    </Figure>
  )
}
