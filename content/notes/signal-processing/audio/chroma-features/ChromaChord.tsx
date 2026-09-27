import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from '@/components/viz'
import { magnitudeSpectrum, makeWindow } from '@/lib/dsp'

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
  const [chord, setChord] = useState<Chord>('cmaj')
  const harmonics = useParam(6, { min: 1, max: 12, step: 1 })
  const decay = useParam(0.6, { min: 0.2, max: 1, step: 0.05 })

  const r = useMemo(() => {
    const w = makeWindow('hann', N, true)
    const x = Array.from(
      { length: N },
      (_, n) =>
        CHORDS[chord].reduce((s, m) => {
          let v = 0
          for (let h = 1; h <= harmonics.value; h++)
            v += decay.value ** (h - 1) * Math.cos((2 * Math.PI * h * midiToHz(m) * n) / FS)
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
  }, [chord, harmonics.value, decay.value])

  const inChord = new Set(CHORDS[chord].map((m) => m % 12))
  const series: XYSeries[] = [
    { name: 'chord tones', type: 'bar', x: [...inChord], y: [...inChord].map((c) => r[c]), slot: 0 },
    {
      name: 'other pitch classes',
      type: 'bar',
      x: NAMES.map((_, c) => c).filter((c) => !inChord.has(c)),
      y: NAMES.map((_, c) => c)
        .filter((c) => !inChord.has(c))
        .map((c) => r[c]),
      muted: true,
    },
  ]
  const strongest = r.indexOf(Math.max(...r))

  return (
    <Interactive
      title="The chroma vector of a chord"
      caption="Notes with decaying harmonics, analysed with one 4096-point FFT; each bin's power is folded onto the nearest of the 12 pitch classes, C = 0 to B = 11, regardless of octave. The chord tones dominate. Harmonics leak into other classes: the 3rd harmonic of each note is its fifth an octave up and the 5th harmonic its major third two octaves up, so a single C gains G and E. Brighter tones, with more and stronger harmonics, spread the chroma further."
      controls={
        <>
          <ParamChoice
            label="chord"
            value={chord}
            onChange={setChord}
            options={[
              { value: 'single', label: 'single C' },
              { value: 'cmaj', label: 'C major' },
              { value: 'amin', label: 'A minor' },
              { value: 'g7', label: 'G7' },
            ]}
          />
          <ParamSlider label="harmonics per note" param={harmonics} withArrows />
          <ParamSlider label="harmonic amplitude ratio" param={decay} />
        </>
      }
      readout={
        <>
          <Readout
            label="notes"
            value={CHORDS[chord].map((m) => `${NAMES[m % 12]}${Math.floor(m / 12) - 1}`).join(' ')}
          />
          <Readout label="strongest class" value={NAMES[strongest]} />
          <Readout
            label="largest non-chord class"
            value={formatNumber(Math.max(...NAMES.map((_, c) => (inChord.has(c) ? 0 : r[c]))))}
          />
        </>
      }
    >
      <XYChart
        series={series}
        xLabel="pitch class (C = 0 … B = 11)"
        yLabel="normalised energy"
        yRange={[0, 1.05]}
        height={300}
      />
    </Interactive>
  )
}
