import { Diagram, Figure, vae } from 'aifn-render'

/** A variational autoencoder */
export function VaeDiagram() {
  return (
    <Figure
      title="A variational autoencoder"
      caption="The encoder outputs the mean and standard deviation of the approximate posterior. The reparameterisation block draws the noise from a fixed standard Gaussian and forms the latent by a product and a sum, so gradients reach the encoder by ordinary backpropagation. The KL term acts on the encoder's output; the reconstruction term acts on the decoder's."
    >
      <Diagram
        spec={vae}
        ariaLabel="Variational autoencoder: encoder outputs mu and sigma, z equals mu plus sigma times epsilon, decoder reconstructs x, KL term on the encoder output"
      />
    </Figure>
  )
}
