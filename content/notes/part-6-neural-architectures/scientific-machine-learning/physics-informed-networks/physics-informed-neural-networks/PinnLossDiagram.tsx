import { Diagram, Figure } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'

const spec: DiagramSpec = {
  unit: 40,
  nodes: [
    { id: 'theta', x: 0, y: 2.2, w: 2.6, h: 0.9, label: 'weights $\\thetavec$', tone: 'neutral' },
    { id: 'col', x: 4.4, y: 0, w: 3.6, h: 1.1, label: 'collocation points\n$(x_i, t_i)$ inside', tone: 'neutral' },
    {
      id: 'cond',
      x: 4.4,
      y: 4.4,
      w: 3.6,
      h: 1.1,
      label: 'initial, boundary\nand data points',
      tone: 'neutral',
    },
    { id: 'net', x: 4.4, y: 2.2, w: 3.6, h: 1.1, label: 'network $u_{\\thetavec}(x, t)$', tone: 0 },
    {
      id: 'ad',
      x: 9.4,
      y: 0,
      w: 4,
      h: 1.2,
      label: 'derivatives in the inputs\n$u_t,\\ u_x,\\ u_{xx}$ by autodiff',
      tone: 1,
    },
    { id: 'res', x: 14.2, y: 0, w: 3.4, h: 1.2, label: 'residual\n$\\mathcal{N}[u_{\\thetavec}](x_i, t_i)$', tone: 1 },
    { id: 'mis', x: 14.2, y: 4.4, w: 3.4, h: 1.2, label: 'misfit\n$u_{\\thetavec} - g$', tone: 2 },
    { id: 'loss', x: 14.2, y: 2.2, w: 3.4, h: 1, shape: 'pill', label: 'loss $L(\\thetavec)$', tone: 'neutral' },
  ],
  edges: [
    { from: 'theta', to: 'net' },
    { from: 'col', to: 'net' },
    { from: 'cond', to: 'net' },
    { from: 'net:e', to: 'ad', via: [[9.4, 2.2]] },
    { from: 'ad', to: 'res' },
    {
      from: 'net:e',
      to: 'mis',
      via: [
        [9.4, 2.2],
        [9.4, 4.4],
      ],
    },
    { from: 'res', to: 'loss', label: 'mean square', labelSide: 'right' },
    { from: 'mis', to: 'loss', label: 'mean square', labelSide: 'right' },
    {
      from: 'loss:s',
      to: 'theta:s',
      via: [
        [14.2, 6],
        [0, 6],
      ],
      label: 'gradient in $\\thetavec$ by reverse mode; Adam, then L-BFGS',
      labelSide: 'right',
    },
  ],
}

/** How a physics-informed network's loss is assembled from one network evaluated at two kinds of points. */
export function PinnLossDiagram() {
  return (
    <Figure
      title="Where a physics-informed loss comes from"
      hoverReadout={false}
      caption="One network is evaluated at two kinds of points. At the collocation points inside the domain, automatic differentiation of the network in its inputs gives the derivatives the equation needs, and the equation's residual is squared and averaged. At the initial, boundary and data points the network's values are compared with the given ones. The sum is differentiated in the weights by reverse mode, which differentiates through the input derivatives too, and an optimiser updates the weights."
    >
      <Diagram
        spec={spec}
        ariaLabel="The weights define a network evaluated at collocation points and at initial, boundary and data points; derivatives of the network in its inputs give the equation's residual; the residual and the misfit are squared and averaged into a loss whose gradient in the weights updates them"
      />
    </Figure>
  )
}
