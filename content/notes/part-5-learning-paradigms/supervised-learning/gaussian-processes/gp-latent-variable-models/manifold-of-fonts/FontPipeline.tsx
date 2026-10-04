import { Diagram, Figure, projector } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'

const box = (id: string, x: number, y: number, label: string, tone: number | 'neutral', w = 2.8, h = 1.1) => ({
  id,
  x,
  y,
  label,
  tone,
  w,
  h,
  small: true,
})

const spec: DiagramSpec = {
  unit: 38,
  nodes: [
    box('ttf', 0, 0, '46 TrueType fonts\n62 characters each', 'neutral'),
    box('poly', 3.5, 0, 'sample each outline\nas $N = 512$ points', 'neutral'),
    box('norm', 7, 0, 'normalise, align\nstarting points', 'neutral'),
    box('match', 10.5, 0, 'joint matching:\ncurvature, normals,\nelasticity', 0),
    {
      id: 'c2f',
      x: 8.3,
      y: 1.45,
      w: 3.2,
      shape: 'text',
      small: true,
      label: 'matching runs coarse to fine\nover elliptic Fourier terms',
    },
    box('vec', 10.5, 3.4, 'one vector $\\uvec_m$ per font\n$D \\ge 2NH = 63{,}488$', 2, 3.2, 1.2),
    { id: 'x', x: 7, y: 3.4, shape: 'latent', label: '$\\xvec$', tone: 2 },
    projector('decoder', 'gp', 4.2, 3.4, 'GP-LVM\nmean map', { dir: 'left', w: 1.9, h: 1.9, tone: 1 }),
    box('font', 0.6, 3.4, 'novel font:\npolylines, then Bézier fit', 'neutral'),
  ],
  edges: [
    { from: 'ttf', to: 'poly' },
    { from: 'poly', to: 'norm' },
    { from: 'norm', to: 'match' },
    { from: 'match', to: 'vec' },
    { from: 'vec', to: 'x', label: 'fit $\\Xmat$, $\\thetavec$', dashed: true },
    { from: 'x', to: 'gp' },
    { from: 'gp', to: 'font' },
  ],
  groups: [
    {
      id: 'stage1',
      label: 'stage 1: character matching',
      tone: 0,
      around: ['poly', 'norm', 'match', 'c2f'],
      pad: 0.35,
    },
    { id: 'stage2', label: 'stage 2: manifold', tone: 1, around: ['x', 'gp'], pad: 0.45, labelAt: 'bottom-left' },
  ],
}

/** The two-stage pipeline of Campbell and Kautz (2014). */
export function FontPipeline() {
  return (
    <Figure
      title="From font files to a manifold of fonts"
      caption="Stage 1 matches every character across all fonts at once, so that sample i of an outline is the same place on the glyph in every font. Stage 2 stacks the matched outlines of all characters into one vector per font and fits a GP-LVM. Any latent point x then generates a complete font through the Gaussian process mean."
    >
      <Diagram
        spec={spec}
        ariaLabel="Pipeline from TrueType fonts through character matching to a GP-LVM font manifold"
      />
    </Figure>
  )
}
