import { Diagram, Figure } from 'aifn-render'
import type { DiagramNode, DiagramSpec } from 'aifn-render'

const step = (id: string, x: number, y: number, label: string, tone: number | 'neutral' = 'neutral'): DiagramNode => ({
  id,
  x,
  y,
  w: 2.4,
  h: 1,
  label,
  tone,
})

const spec: DiagramSpec = {
  nodes: [
    { id: 'x', x: 0, y: 0, shape: 'text', w: 1, label: 'audio' },
    step('pre', 2.2, 0, 'pre-emphasis\n$1 - 0.97 z^{-1}$'),
    step('frame', 5.2, 0, 'frame and window\n25 ms every 10 ms'),
    step('pow', 2.2, 2.4, 'power spectrum\n$\\abs{X_t[k]}^2$', 0),
    step('mel', 5.2, 2.4, 'mel filter bank\n$M$ bands, $E_m$', 0),
    step('log', 8.2, 2.4, 'logarithm\n$\\log E_m$', 0),
    step('dct', 11.2, 2.4, 'DCT-II\nkeep $K$ of $M$', 1),
    { id: 'c', x: 11.2, y: 4.3, shape: 'text', w: 2.4, label: 'MFCCs $c_0, \\dots, c_{K-1}$' },
  ],
  edges: [
    { from: 'x', to: 'pre' },
    { from: 'pre', to: 'frame' },
    {
      from: 'frame:s',
      to: 'pow:n',
      via: [
        [5.2, 1.2],
        [2.2, 1.2],
      ],
    },
    { from: 'pow', to: 'mel' },
    { from: 'mel', to: 'log' },
    { from: 'log', to: 'dct' },
    { from: 'dct', to: 'c' },
  ],
  groups: [
    {
      id: 'lm',
      label: 'log-mel spectrogram',
      labelAt: 'bottom-left',
      tone: 0,
      dashed: true,
      around: ['pow', 'mel', 'log'],
      pad: 0.25,
    },
  ],
}

/** The MFCC pipeline from audio samples to cepstral coefficients. */
export function MfccPipeline() {
  return (
    <Figure
      title="From audio to MFCCs"
      caption="Each frame's power spectrum is pooled into mel-spaced bands and compressed by the logarithm; those three steps give the log-mel spectrogram. The DCT decorrelates the log energies, and truncating it to the first K coefficients keeps the smooth spectral envelope."
    >
      <Diagram
        spec={spec}
        ariaLabel="Audio passes through pre-emphasis, framing and windowing, and a power spectrum; then a mel filter bank, a logarithm and a DCT give the MFCCs"
      />
    </Figure>
  )
}
