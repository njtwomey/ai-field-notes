import { Diagram, Figure, latentDiffusion } from 'aifn-render'

/** Latent diffusion */
export function LatentDiffusionDiagram() {
  return (
    <Figure
      title="Latent diffusion"
      caption="The frozen encoder maps the image into a smaller latent grid. The forward process noises the latent (dashed arrows, used only in training). The denoiser runs the reverse process in latent space and reads the condition through cross-attention. The decoder maps the final latent back to pixels once."
    >
      <Diagram
        spec={latentDiffusion}
        ariaLabel="Latent diffusion: encoder to latent, forward noising to z_T, conditioned denoiser, decoder back to pixels"
      />
    </Figure>
  )
}
