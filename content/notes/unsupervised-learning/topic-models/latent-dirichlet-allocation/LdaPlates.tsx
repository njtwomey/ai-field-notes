import { MathText } from '@/components/content/MathText'
import { GraphDiagram, Interactive } from '@/components/viz'

/** LDA in plate notation: documents d contain word positions n; topics k sit on their own plate. */
export function LdaPlates() {
  return (
    <Interactive
      title="LDA in plate notation"
      caption={
        <MathText text="A plate repeats its contents: $D$ documents, $N_d$ word positions in document $d$, and $K$ topics. Only the words $w_{dn}$ (shaded) are observed. Each word depends on its topic assignment $z_{dn}$ and on all $K$ topics $\phivec_k$." />
      }
    >
      <GraphDiagram
        nodes={[
          { id: 'alpha', label: 'α', x: 0, y: 1.5 },
          { id: 'theta', label: '\\mathbf{θ}_d', x: 1.3, y: 1.5 },
          { id: 'z', label: 'z_{dn}', x: 2.6, y: 1.5 },
          { id: 'w', label: 'w_{dn}', x: 3.9, y: 1.5, kind: 'observed' },
          { id: 'phi', label: '\\mathbf{φ}_k', x: 3.9, y: 0 },
          { id: 'beta', label: 'β', x: 5.3, y: 0 },
        ]}
        edges={[
          { source: 'alpha', target: 'theta' },
          { source: 'theta', target: 'z' },
          { source: 'z', target: 'w' },
          { source: 'phi', target: 'w' },
          { source: 'beta', target: 'phi' },
        ]}
        plates={[
          { label: 'D', x0: 0.75, y0: 0.9, x1: 4.55, y1: 2.45 },
          { label: 'N_d', x0: 2.1, y0: 1.02, x1: 4.35, y1: 2.08 },
          { label: 'K', x0: 3.35, y0: -0.55, x1: 4.55, y1: 0.6 },
        ]}
        height={280}
        ariaLabel="Plate diagram of latent Dirichlet allocation"
      />
    </Interactive>
  )
}
