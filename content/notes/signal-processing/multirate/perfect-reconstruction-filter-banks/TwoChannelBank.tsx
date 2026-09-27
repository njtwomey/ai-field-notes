import { Interactive } from '@/components/viz'
import { Diagram } from '@/components/diagram/Diagram'
import type { DiagramNode, DiagramSpec } from '@/components/diagram/types'

const filt = (id: string, x: number, y: number, label: string, tone: number): DiagramNode => ({
  id,
  x,
  y,
  w: 1.3,
  h: 0.8,
  label,
  tone,
})
const rate = (id: string, x: number, y: number, label: string): DiagramNode => ({
  id,
  x,
  y,
  shape: 'circle',
  w: 0.8,
  h: 0.8,
  label,
  tone: 'neutral',
})

const spec: DiagramSpec = {
  nodes: [
    { id: 'x', x: 0, y: 1.2, shape: 'text', w: 0.9, label: '$x[n]$' },
    { id: 'split', x: 1.2, y: 1.2, shape: 'dot' },
    filt('h0', 2.6, 0, '$H_0(z)$', 0),
    filt('h1', 2.6, 2.4, '$H_1(z)$', 1),
    rate('d0', 4.3, 0, '$\\downarrow 2$'),
    rate('d1', 4.3, 2.4, '$\\downarrow 2$'),
    rate('u0', 8.1, 0, '$\\uparrow 2$'),
    rate('u1', 8.1, 2.4, '$\\uparrow 2$'),
    filt('g0', 9.8, 0, '$G_0(z)$', 0),
    filt('g1', 9.8, 2.4, '$G_1(z)$', 1),
    { id: 'sum', x: 11.4, y: 1.2, shape: 'op', label: '$+$' },
    { id: 'xh', x: 12.6, y: 1.2, shape: 'text', w: 0.9, label: '$\\hat x[n]$' },
  ],
  edges: [
    { from: 'x', to: 'split', arrow: 'none' },
    { from: 'split', to: 'h0:w', via: [[1.2, 0]] },
    { from: 'split', to: 'h1:w', via: [[1.2, 2.4]] },
    { from: 'h0', to: 'd0' },
    { from: 'h1', to: 'd1' },
    { from: 'd0', to: 'u0', label: 'lowpass subband' },
    { from: 'd1', to: 'u1', label: 'highpass subband', labelSide: 'right' },
    { from: 'u0', to: 'g0' },
    { from: 'u1', to: 'g1' },
    { from: 'g0:e', to: 'sum:n', via: [[11.4, 0]] },
    { from: 'g1:e', to: 'sum:s', via: [[11.4, 2.4]] },
    { from: 'sum', to: 'xh' },
  ],
  groups: [
    { id: 'an', label: 'analysis', tone: 'neutral', around: ['h0', 'h1', 'd0', 'd1'], pad: 0.3 },
    { id: 'sy', label: 'synthesis', tone: 'neutral', around: ['u0', 'u1', 'g0', 'g1'], pad: 0.3 },
  ],
}

/** The two-channel analysis–synthesis filter bank. */
export function TwoChannelBank() {
  return (
    <Interactive
      title="A two-channel filter bank"
      caption="The analysis filters H₀ (lowpass) and H₁ (highpass) split the input, and each branch keeps every second sample. The synthesis side inserts zeros between samples, filters with G₀ and G₁ and adds the branches. The downsamplers create aliasing in each branch; perfect reconstruction chooses the four filters so that the aliasing cancels in the sum."
    >
      <Diagram
        spec={spec}
        ariaLabel="The input splits into a lowpass branch H0 and a highpass branch H1, each downsampled by 2, then upsampled by 2, filtered by G0 and G1, and summed to give the reconstruction"
      />
    </Interactive>
  )
}
