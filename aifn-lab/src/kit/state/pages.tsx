/**
 * The figure-state pages of the UI kit (`/ui-kit/<page>`): state with its URL round trip, probes, the scheduler and
 * equations. `make lab-shots ARGS="--only ui-kit"` shoots them with the rest of the kit.
 */
import type { ReactNode } from 'react'
import type { LabPage } from '@lab/layout'
import {
  EquationFigure,
  EquationStepsFigure,
  FigureStateFigure,
  RasterProbeFigure,
  SchedulerFigure,
  SharedProbeFigure,
} from './StateFigures'

const page = (slug: string, title: string, description: string, body: () => ReactNode): LabPage => ({
  key: `ui-kit/${slug}`,
  title,
  description,
  render: () => <div className="flex flex-col gap-6">{body()}</div>,
})

export const stateKitPages: LabPage[] = [
  page(
    'state',
    'State: useFigureState',
    'One declaration gives the control rows (variants, a reveal row, a setting), typed values, a handle on the chart, reset and the URL round trip.',
    () => <FigureStateFigure />,
  ),
  page(
    'probes',
    'State: probes',
    'A probe shared by two plots, and a point probe on a raster set by pressing anywhere, with its slice and readout.',
    () => (
      <>
        <SharedProbeFigure />
        <RasterProbeFigure />
      </>
    ),
  ),
  page(
    'scheduler',
    'State: useComputed scheduler',
    'An expensive derived path under a draggable start: frame and release modes against a plain memo, with run times.',
    () => <SchedulerFigure />,
  ),
  page(
    'equations',
    'State: equations',
    'The equation band with live values, and EquationSteps walking through the chain rule with a Player.',
    () => (
      <>
        <EquationFigure />
        <EquationStepsFigure />
      </>
    ),
  ),
]
