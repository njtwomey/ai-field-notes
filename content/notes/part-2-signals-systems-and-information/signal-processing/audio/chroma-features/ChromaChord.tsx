import { useMemo } from 'react'
import { Bars, choice, Figure, formatNumber, int, Plot, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { getWindow } from 'aifn-compute/signal/windows'
import { magnitudeSpectrum } from '../_shared/audio'

const FS = 16000
const N = 4096
const NAMES = ['C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B']

/** MIDI numbers of the notes in each chord (C4 = 60). */
const CHORDS = {
  cmaj: [60, 64, 67],
  amin: [57, 60, 64],
  g7: [55, 59, 62, 65],
  single: [60],
} as const
type Chord = keyof typeof CHORDS

const midiToHz = (m: number) => 440 * 2 ** ((m - 69) / 12)

/**
 * A chroma vector: each FFT bin's power is added to the pitch class of its nearest equal-tempered pitch, over a
 * frequency range, and the 12 values are normalised to a maximum of 1.
 */
export function ChromaChord() {
  const state = useFigureState({
    chord: choice<Chord>(
      [
        { value: 'single', label: 'single C' },
        { value: 'cmaj', label: 'C major' },
        { value: 'amin', label: 'A minor' },
        { value: 'g7', label: 'G7' },
      ],
      'cmaj',
      { label: 'chord' },
    ),
    harmonics: int(6, { min: 1, max: 12, step: 1, label: 'harmonics per note' }),
    decay: slider(0.2, 1, 0.6, { step: 0.05, label: 'harmonic amplitude ratio' }),
  })

  const r = useMemo(() => {
    const w = toFlat(getWindow('hann', N, { periodic: true }))
    const x = Array.from(
      { length: N },
      (_, n) =>
        CHORDS[state.chord].reduce((s, m) => {
          let v = 0
          for (let h = 1; h <= state.harmonics; h++)
            v += state.decay ** (h - 1) * Math.cos((2 * Math.PI * h * midiToHz(m) * n) / FS)
          return s + v
        }, 0) * w[n],
    )
    const mag = magnitudeSpectrum(x, N)
    const chroma = new Array(12).fill(0)
    for (let k = 1; k < mag.length; k++) {
      const f = (k * FS) / N
      if (f < 60 || f > 5000) continue
      const midi = 69 + 12 * Math.log2(f / 440)
      chroma[((Math.round(midi) % 12) + 12) % 12] += mag[k] * mag[k]
    }
    const top = Math.max(...chroma)
    return chroma.map((v) => v / top)
  }, [state.chord, state.harmonics, state.decay])

  const inChord = new Set(CHORDS[state.chord].map((m) => m % 12))
  const series = [
    { name: 'chord tones', x: [...inChord], y: [...inChord].map((c) => r[c]), slot: 0 },
    {
      name: 'other pitch classes',
      x: NAMES.map((_, c) => c).filter((c) => !inChord.has(c)),
      y: NAMES.map((_, c) => c)
        .filter((c) => !inChord.has(c))
        .map((c) => r[c]),
      muted: true,
    },
  ] as const
  const strongest = r.indexOf(Math.max(...r))

  const xAxis = useAxis({ label: 'pitch class (C = 0 … B = 11)', hold: 'union' })
  const yAxis = useAxis({ label: 'normalised energy', range: [0, 1.05] })
  return (
    <Figure
      title="The chroma vector of a chord"
      state={state}
      caption="Notes with decaying harmonics, analysed with one 4096-point FFT; each bin's power is folded onto the nearest of the 12 pitch classes, C = 0 to B = 11, regardless of octave. The chord tones dominate. Harmonics leak into other classes: the 3rd harmonic of each note is its fifth an octave up and the 5th harmonic its major third two octaves up, so a single C gains G and E. Brighter tones, with more and stronger harmonics, spread the chroma further."

      readouts={
        <>
          <Readout
            label="notes"
            value={CHORDS[state.chord].map((m) => `${NAMES[m % 12]}${Math.floor(m / 12) - 1}`).join(' ')}
          />
          <Readout label="strongest class" value={NAMES[strongest]} />
          <Readout
            label="largest non-chord class"
            value={formatNumber(Math.max(...NAMES.map((_, c) => (inChord.has(c) ? 0 : r[c]))))}
          />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        <Bars {...series[0]} />
        <Bars {...series[1]} />
      </Plot>
    </Figure>
  )
}
