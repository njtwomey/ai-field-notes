import { MathText } from 'aifn-render'
import { Diagram } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'
import { Interactive } from 'aifn-render'

const box = (id: string, x: number, y: number, label: string, tone: number | 'neutral') => ({
  id,
  x,
  y,
  w: 2.6,
  h: 1.2,
  label,
  tone,
})

/** From pixels to a bag of visual words, then to a topic model. */
const spec: DiagramSpec = {
  nodes: [
    box('image', 0, 0, 'image', 'neutral'),
    box('patches', 3.7, 0, 'patches\n(grid or detector)', 0),
    box('desc', 7.4, 0, 'descriptors\n$\\xvec \\in \\reals^{128}$', 0),
    box('codebook', 7.4, 2.6, 'codebook: $k$-means\n$V$ centres', 1),
    box('words', 11.1, 0, 'visual words\n$w \\in \\set{1, \\dots, V}$', 1),
    box('bag', 14.8, 0, 'bag of visual words\n(counts)', 2),
    box('topics', 14.8, 2.6, 'pLSA / LDA\ntopics', 2),
  ],
  edges: [
    { from: 'image', to: 'patches' },
    { from: 'patches', to: 'desc' },
    { from: 'desc', to: 'words', label: 'nearest centre', labelOffset: 0.2 },
    { from: 'codebook:n', to: 'desc:s', dashed: true, label: 'learned on training descriptors', labelRotate: false },
    { from: 'words', to: 'bag' },
    { from: 'bag:s', to: 'topics:n' },
  ],
}

export function VisualWordsPipeline() {
  return (
    <Interactive
      title="From an image to a document"
      caption={
        <MathText text="Local patches are described by vectors such as SIFT descriptors. A codebook of $V$ centres, learned by $k$-means on descriptors from many images, quantises each descriptor to its nearest centre, a visual word. The image becomes a vector of word counts, which any topic model of text accepts." />
      }
    >
      <Diagram spec={spec} ariaLabel="Pipeline from image patches to visual words to a topic model" />
    </Interactive>
  )
}
