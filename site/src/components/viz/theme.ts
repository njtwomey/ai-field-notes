import { categorical, chrome, type Mode } from './palette'

const FONT = "'Geist Variable', system-ui, sans-serif"

/**
 * Base ECharts option applied under every chart. Chart components add series and axes on top. Keep all chart
 * styling decisions here so that every figure on the site looks the same.
 */
export function baseOption(mode: Mode) {
  const c = chrome(mode)
  const axis = {
    axisLine: { lineStyle: { color: c.axis } },
    axisTick: { show: false },
    axisLabel: { color: c.muted, fontSize: 11, fontFamily: FONT },
    splitLine: { lineStyle: { color: c.grid, width: 1 } },
    nameTextStyle: { color: c.inkSecondary, fontSize: 12, fontFamily: FONT },
    nameLocation: 'middle' as const,
    nameGap: 28,
  }
  return {
    backgroundColor: 'transparent',
    color: categorical(mode),
    textStyle: { fontFamily: FONT, color: c.ink },
    // No entrance animation. ECharts draws lines through an animated clip path; a resize during that animation (font
    // load, scrollbar, layout shift) can leave lines hidden until a hover repaints them.
    animationDuration: 0,
    animationDurationUpdate: 80,
    grid: { left: 52, right: 16, top: 36, bottom: 44, containLabel: false },
    legend: {
      top: 0,
      right: 0,
      icon: 'circle',
      itemWidth: 8,
      itemHeight: 8,
      textStyle: { color: c.inkSecondary, fontSize: 12, fontFamily: FONT },
    },
    tooltip: {
      backgroundColor: mode === 'dark' ? '#262624' : '#ffffff',
      borderColor: mode === 'dark' ? 'rgba(255,255,255,0.10)' : 'rgba(11,11,11,0.10)',
      textStyle: { color: c.ink, fontSize: 12, fontFamily: FONT },
      extraCssText: 'border-radius: 8px; box-shadow: 0 4px 16px rgba(0,0,0,0.08);',
    },
    xAxis: axis,
    yAxis: axis,
  }
}

export const LINE_WIDTH = 2
export const MARKER_SIZE = 8

/** Round tick labels so axes read 0.5 rather than 0.49999999. */
export function formatNumber(v: number): string {
  if (!Number.isFinite(v)) return String(v)
  const a = Math.abs(v)
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(1)
  return String(Number(v.toPrecision(4)))
}
