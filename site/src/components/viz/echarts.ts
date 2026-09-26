/** Tree-shaken ECharts build. Register new chart types or components here, once. */
import { BarChart, CustomChart, GraphChart, LineChart, ScatterChart } from 'echarts/charts'
import {
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  TooltipComponent,
  VisualMapComponent,
} from 'echarts/components'
import * as echarts from 'echarts/core'
import { CanvasRenderer, SVGRenderer } from 'echarts/renderers'

echarts.use([
  BarChart,
  LineChart,
  ScatterChart,
  CustomChart,
  GraphChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  VisualMapComponent,
  MarkLineComponent,
  DataZoomComponent,
  SVGRenderer,
  CanvasRenderer,
])

export { echarts }
export type { EChartsCoreOption as EChartsOption } from 'echarts/core'
