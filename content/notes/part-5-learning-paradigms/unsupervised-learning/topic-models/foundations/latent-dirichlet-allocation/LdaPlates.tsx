import { MathText } from 'aifn-render'
import { Diagram } from 'aifn-render'
import { lda } from 'aifn-render'
import { Interactive } from 'aifn-render'

/** LDA in plate notation: documents d contain word positions n; topics k sit on their own plate. */
export function LdaPlates() {
  return (
    <Interactive
      title="LDA in plate notation"
      caption={
        <MathText text="A plate repeats its contents: $D$ documents, $N_d$ word positions in document $d$, and $K$ topics. Only the words $w_{dn}$ (shaded) are observed. Each word depends on its topic assignment $z_{dn}$ and on all $K$ topics $\phivec_k$." />
      }
    >
      <Diagram spec={lda} ariaLabel="Plate diagram of latent Dirichlet allocation" />
    </Interactive>
  )
}
