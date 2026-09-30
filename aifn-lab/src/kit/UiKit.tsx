import {
  ClippedVectorFigure,
  ClusterFigure,
  DerivativeFigure,
  HistogramFigure,
  LogFigure,
  VectorFigure,
} from './ChartDemos'
import { ControlsDemo } from './ControlsDemo'
import { FunctionFamilyFigure } from './FunctionFamily'
import { HeatmapFigure, LinkedFigure } from './HeatmapDemos'
import { PaletteDemo } from './PaletteDemo'
import { Section } from './Section'
import { GridFigure, SharedXFigure, SharedYFigure } from './SubplotsDemos'
import { DataViewsFigures } from './DataViewsDemos'

/**
 * The UI kit: a review page for the v2 building blocks. Every control, chart type, zoom and pan, hover, handles, figure
 * sizing, data export and both themes appear here, against seeded data from aifn.
 */
export function UiKit() {
  return (
    <div className="flex flex-col gap-10">
      <Section
        title="Theme and palette"
        description="The lab owns its tokens: shadcn colours in lab.css, data colours in src/design/palette.ts (seeded from the site's palette). Switch the theme here or in the sidebar."
      >
        <PaletteDemo />
      </Section>
      <Section
        title="Controls"
        description="Sliders are steppable by default and take a typed value; categorical choices are dropdowns, searchable when long."
      >
        <ControlsDemo />
      </Section>
      <Section
        title="Conditional controls"
        description="defineVariants: each function brings its own parameters; the controls follow the chosen one, and each keeps its values. Params can also show conditionally within a variant."
      >
        <FunctionFamilyFigure />
      </Section>
      <Section
        title="Hover"
        description="Line charts hover by x: every series' value at the pointer, in the tooltip and in the figure's readout."
      >
        <DerivativeFigure />
      </Section>
      <Section
        title="Charts, handles and zoom"
        description="Every figure frame has a size picker (S, M, L, XL, full width), a corner to drag, and a button that copies its data as JSON. Sizes are remembered per page."
      >
        <ClusterFigure />
        <HistogramFigure />
        <LogFigure />
        <VectorFigure />
        <ClippedVectorFigure />
      </Section>
      <Section
        title="Heatmaps"
        description="Cells, contours, overlays, a marker, vectors and a handle, with zoom and pan."
      >
        <HeatmapFigure />
      </Section>
      <Section
        title="Subplots"
        description="Subplots and Panel: aligned plot areas, shared axes (one zoom for all), tick labels on the outer panels only, height and width ratios within the frame."
      >
        <SharedXFigure />
        <SharedYFigure />
        <GridFigure />
      </Section>
      <Section
        title="Data views"
        description="PairPlotView and ParallelCoordinatesView from @lab/views: a labelled dataset in every feature at once, classes in their categorical slots, class chips that hide a class everywhere, and brushing that filters rows across panels or axes."
      >
        <DataViewsFigures />
      </Section>
      <Section
        title="Linked panels"
        description="Charts with the same hoverGroup share the hovered x; ChartSize sets each panel's share of the frame."
      >
        <LinkedFigure />
      </Section>
    </div>
  )
}
