import { Diagram, Figure } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'

const spec: DiagramSpec = {
  unit: 40,
  nodes: [
    { id: 'a', x: 0, y: 2, w: 3, h: 1.2, label: 'input $a(x)$\nand $x$ on $n$ points', tone: 'neutral' },
    { id: 'lift', x: 4, y: 2, w: 2.6, h: 1.2, label: 'lift $P$\npointwise', tone: 'neutral' },
    { id: 'fft', x: 8, y: 0, w: 2.4, h: 1.1, label: 'FFT', tone: 0 },
    { id: 'mult', x: 11.2, y: 0, w: 3.6, h: 1.1, label: 'keep $k < k_{\\max}$\nmultiply by $R_k$', tone: 0 },
    { id: 'ifft', x: 15.6, y: 0, w: 2.4, h: 1.1, label: 'inverse FFT', tone: 0 },
    { id: 'w', x: 11.2, y: 4, w: 3.6, h: 1.1, label: 'pointwise $W v + b$', tone: 1 },
    {
      id: 'sum',
      x: 18.8,
      y: 2,
      w: 2.2,
      h: 1.1,
      shape: 'pill',
      label: '$\\sigma(\\,\\cdot + \\cdot\\,)$',
      tone: 'neutral',
    },
    { id: 'proj', x: 22.2, y: 2, w: 2.8, h: 1.2, label: 'project $Q$\npointwise', tone: 'neutral' },
    { id: 'u', x: 26.2, y: 2, w: 2.6, h: 1.2, label: 'output\n$u(x)$', tone: 'neutral' },
  ],
  edges: [
    { from: 'a', to: 'lift' },
    {
      from: 'lift:e',
      to: 'fft',
      via: [
        [7.2, 2],
        [7.2, 0],
      ],
    },
    { from: 'fft', to: 'mult' },
    { from: 'mult', to: 'ifft' },
    {
      from: 'lift:e',
      to: 'w',
      via: [
        [7.2, 2],
        [7.2, 4],
      ],
    },
    { from: 'ifft:e', to: 'sum', via: [[18.8, 0]] },
    { from: 'w:e', to: 'sum', via: [[18.8, 4]] },
    { from: 'sum', to: 'proj', label: 'repeat $L$ times', labelSide: 'left' },
    { from: 'proj', to: 'u' },
  ],
}

/** The layers of a Fourier neural operator. */
export function FnoDiagram() {
  return (
    <Figure
      title="Inside a Fourier neural operator"
      hoverReadout={false}
      caption="A Fourier neural operator lifts the input function and its coordinate to a vector of channels at every grid point, applies L Fourier layers, and projects the channels back to the output at every point. A Fourier layer has two paths. The spectral path (top) transforms each channel by the FFT, keeps the lowest frequencies, multiplies the channel vector at each kept frequency k by a learned complex matrix R_k and transforms back. The pointwise path (bottom) multiplies the channel vector at each grid point by one matrix W. The sum passes through an activation. Every learned weight is indexed by a frequency or a channel, never by a grid point."
    >
      <Diagram
        spec={spec}
        ariaLabel="The input and its coordinate are lifted pointwise; each Fourier layer adds a spectral path (FFT, truncation and multiplication by learned weights, inverse FFT) to a pointwise linear path and applies an activation; after L layers a pointwise projection gives the output"
      />
    </Figure>
  )
}
