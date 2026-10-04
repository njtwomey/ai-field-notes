import { autoencoder, Diagram, Figure } from 'aifn-render'

/** An autoencoder */
export function AutoencoderDiagram() {
  return (
    <Figure
      title="An autoencoder"
      caption="The encoder narrows the input to a code of dimension k < d. The decoder widens it back. The loss compares the reconstruction with the input, and it is the only training signal."
    >
      <Diagram
        spec={autoencoder}
        ariaLabel="Autoencoder: encoder to bottleneck code z, decoder to reconstruction, loss compares input and reconstruction"
      />
    </Figure>
  )
}
