/**
 * The Plot pages of the UI kit (`/ui-kit/<page>`): one page per layer and one for `Plots`, each a review page for that
 * building block against seeded data from aifn. `make lab-shots ARGS="--only ui-kit"` shoots them with the kit.
 */
import type { ReactNode } from 'react'
import type { LabPage } from '@lab/layout'
import {
  AnnotationFigure,
  AreaFigure,
  BarsFigure,
  CurveFigure,
  DensePointsFigure,
  HandleFigure,
  PointsFigure,
  RugFigure,
  SegmentsFigure,
  SignedAreaFigure,
  VectorsFigure,
} from './MarkLayers'
import {
  CategoricalFigure,
  ChangeOfVariablesFigure,
  RasterAlignedFigure,
  SharedAxisFigure,
  UnitSquareFigure,
} from './PlotsFigures'
import {
  ContoursFigure,
  DensityFigure,
  HistogramFigure,
  MassFigure,
  RasterFigure,
  SupportBandFigure,
} from './ProbabilityLayers'

const page = (slug: string, title: string, description: string, body: () => ReactNode): LabPage => ({
  key: `ui-kit/${slug}`,
  title,
  description,
  render: () => <div className="flex flex-col gap-6">{body()}</div>,
})

export const plotKitPages: LabPage[] = [
  page(
    'plots',
    'Plot: Plots and axes',
    'Plots and useAxis: axis models shared by identity, equal units, a categorical x, a raster aligned with curves, and a square unit square with a tight strip.',
    () => (
      <>
        <SharedAxisFigure />
        <ChangeOfVariablesFigure />
        <CategoricalFigure />
        <RasterAlignedFigure />
        <UnitSquareFigure />
      </>
    ),
  ),
  page('curve', 'Layer: Curve', 'Lines through points, hovered by x; a live curve moves by patch.', () => (
    <CurveFigure />
  )),
  page(
    'points',
    'Layer: Points',
    'Scatter marks: classes in their slots, marker shapes per point, a dense mode.',
    () => (
      <>
        <PointsFigure />
        <DensePointsFigure />
      </>
    ),
  ),
  page(
    'bars',
    'Layer: Bars',
    'Data-unit rectangles on any axis, categorical axes included, in either orientation.',
    () => <BarsFigure />,
  ),
  page('area', 'Layer: Area', 'The region between a curve and a base, a constant or a second curve.', () => (
    <AreaFigure />
  )),
  page(
    'signed-area',
    'Layer: SignedArea',
    'Area above and below zero in the diverging ends, with its net area.',
    () => <SignedAreaFigure />,
  ),
  page('histogram', 'Layer: Histogram', 'Samples binned by aifn/probability/stats, along x or y.', () => (
    <HistogramFigure />
  )),
  page(
    'density',
    'Layer: Density',
    'The density of an aifn distribution, evaluated by the distribution object.',
    () => <DensityFigure />,
  ),
  page('mass', 'Layer: Mass', 'The mass function of a discrete aifn distribution, a bar per integer.', () => (
    <MassFigure />
  )),
  page('rug', 'Layer: Rug', 'A tick per value along the plot edge.', () => <RugFigure />),
  page('support-band', 'Layer: SupportBand', 'A support interval on the axis, with open and closed ends.', () => (
    <SupportBandFigure />
  )),
  page('vectors', 'Layer: Vectors', 'Arrows, clipped to the plot with a chevron where they leave it.', () => (
    <VectorsFigure />
  )),
  page('segments', 'Layer: Segments', 'Thin line segments: residuals, meshes, steps.', () => <SegmentsFigure />),
  page(
    'raster',
    'Layer: Raster',
    'A value grid as one cached canvas image, with a colour bar and a hit layer for hover.',
    () => <RasterFigure />,
  ),
  page('contours', 'Layer: Contours', 'Level sets of a field by marching squares.', () => <ContoursFigure />),
  page('handle', 'Layer: Handle', 'A draggable point or guide line bound to a parameter with a natural place.', () => (
    <HandleFigure />
  )),
  page('annotation', 'Layer: Annotation', 'Labelled points and lines.', () => <AnnotationFigure />),
]
