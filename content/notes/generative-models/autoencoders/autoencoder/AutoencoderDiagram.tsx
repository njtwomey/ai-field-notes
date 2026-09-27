import { Interactive } from '@/components/viz'
import { Diagram } from '@/components/diagram/Diagram'
import { autoencoder } from '@/components/diagram/specs/generative'

/** An autoencoder */
export function AutoencoderDiagram() {
  return (
    <Interactive
      title="An autoencoder"
      caption="The encoder narrows the input to a code of dimension k < d. The decoder widens it back. The loss compares the reconstruction with the input, and it is the only training signal."
    >
      <Diagram
        spec={autoencoder}
        ariaLabel="Autoencoder: encoder to bottleneck code z, decoder to reconstruction, loss compares input and reconstruction"
      />
    </Interactive>
  )
}
