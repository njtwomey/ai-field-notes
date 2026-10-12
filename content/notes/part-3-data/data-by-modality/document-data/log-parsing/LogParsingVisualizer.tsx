import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function LogParsingVisualizer() {
  const state = useFigureState({
    logMode: choice<'drain_template' | 'variable_extraction' | 'session_windowing'>(
      [
        {
          value: 'drain_template',
          label: 'Drain Parsing: Similarity Threshold vs Template Explosion (He 2017)',
        },
        {
          value: 'variable_extraction',
          label: 'Variable Fields: Parameter Distribution vs Anomaly Detection (Du 2017)',
        },
        {
          value: 'session_windowing',
          label: 'Sessionizing: Idle Gap Timeout tau vs Transaction Fragmentation',
        },
      ],
      'drain_template',
      { label: 'log parsing regime' },
    ),
    param1: slider(10, 95, 50, {
      step: 5,
      label: 'similarity threshold tau (%) / parameter value / idle timeout (min)',
    }),
    param2: slider(1, 10, 4, {
      step: 1,
      label: 'tree depth / noise rate / session concurrency',
    }),
  })

  const { logMode, param1, param2 } = state

  const results = useMemo(() => {
    if (logMode === 'drain_template') {
      // Similarity thresholds from 20% to 95%
      const thresholds = [20, 30, 40, 50, 60, 70, 80, 85, 90, 95]
      // Ground truth templates in benchmark = 45
      // Low threshold merges distinct events into too few templates
      // High threshold causes template explosion (thousands of pseudo-templates)
      const templateCounts = thresholds.map((th) => {
        if (th <= 50) {
          return Math.max(12, 12 + Math.pow(th / 50, 1.8) * 33)
        } else {
          return 45 + Math.pow((th - 50) / 10, 2.3) * 18 * param2
        }
      })

      // Parsing Accuracy (Grouping Accuracy GA %) peaks around 50-60%
      const groupingAcc = thresholds.map((th) => {
        const dist = Math.abs(th - 55)
        return Math.max(25, 96 - Math.pow(dist / 20, 1.9) * 38)
      })

      const currTh = param1
      const currCount =
        currTh <= 50
          ? Math.max(12, 12 + Math.pow(currTh / 50, 1.8) * 33)
          : 45 + Math.pow((currTh - 50) / 10, 2.3) * 18 * param2
      const currAcc = Math.max(25, 96 - Math.pow(Math.abs(currTh - 55) / 20, 1.9) * 38)

      return {
        x: thresholds,
        y: groupingAcc,
        baseline: templateCounts.map((c) => Math.min(100, c * 0.5)), // scaled to %
        stat1: `Drain Threshold: tau = ${currTh}%`,
        stat2: `Mined Templates: ${Math.round(currCount)} (Truth: 45)`,
        stat3: `Grouping Accuracy: ${currAcc.toFixed(1)}%`,
        warning:
          currTh > 75
            ? 'Severe template explosion; variable identifiers (IPs, UUIDs, timestamps) are mistaken for constant template tokens, blowing up vocabulary size'
            : currTh < 35
              ? 'Over-merging; distinct system errors are collapsed into overly broad wildcard catch-all templates'
              : 'Optimal tree partition; Drain correctly isolates constant prefix strings from variable parameters',
        axisXLabel: 'Drain Token Sequence Similarity Threshold tau (%)',
        axisYLabel: 'Parsing Grouping Accuracy GA (%) / Scaled Template Count',
        xRange: [15, 98] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    } else if (logMode === 'variable_extraction') {
      // Execution latency parameter bins (milliseconds ms): 10ms to 500ms
      const latencies = [10, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500]
      // Normal log traffic density (log-normal distribution centered around 60ms)
      const normalDensity = latencies.map((l) => 100 * Math.exp(-Math.pow(Math.log(l / 60), 2) / 0.8))
      // Anomaly detection score (%) for observed value
      const anomalyScores = latencies.map((l) => (l > 180 ? Math.min(99, Math.pow((l - 180) / 120, 1.4) * 65 + 30) : 5))

      const currLat = (param1 / 95) * 490 + 10
      const currAnom = currLat > 180 ? Math.min(99, Math.pow((currLat - 180) / 120, 1.4) * 65 + 30) : 5
      const isOutlier = currAnom > 50

      return {
        x: latencies,
        y: normalDensity,
        baseline: anomalyScores,
        stat1: `Extracted Latency: ${currLat.toFixed(0)} ms`,
        stat2: `Anomaly Probability: ${currAnom.toFixed(1)}%`,
        stat3: `Alert Status: ${isOutlier ? 'ANOMALOUS SPIKE' : 'NORMAL RANGE'}`,
        warning: isOutlier
          ? 'Numerical parameter exceeds 3-sigma statistical distribution; DeepLog flags downstream service degradation'
          : 'Extracted variable parameters adhere to calibrated baseline operational density',
        axisXLabel: 'Extracted Variable Parameter Value: Response Latency (ms)',
        axisYLabel: 'Normal Log Density (%) / Anomaly Outlier Score (%)',
        xRange: [5, 515] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    } else {
      // Session idle timeout tau from 1 to 30 minutes
      const idleTimeouts = [1, 2, 5, 10, 15, 20, 25, 30]
      // Percentage of multi-step transactions improperly fragmented
      const fragmentedPct = idleTimeouts.map((t) => Math.max(2, 65 * Math.exp(-0.22 * t)))
      // Average session event sequence length (tokens per session)
      const avgSessionLen = idleTimeouts.map((t) => Math.min(80, 8 + Math.pow(t, 0.9) * 2.8))

      const currTimeout = Math.min(30, Math.max(1, Math.round((param1 / 95) * 29 + 1)))
      const currFrag = Math.max(2, 65 * Math.exp(-0.22 * currTimeout))
      const currLen = Math.min(80, 8 + Math.pow(currTimeout, 0.9) * 2.8)

      return {
        x: idleTimeouts,
        y: fragmentedPct,
        baseline: avgSessionLen,
        stat1: `Idle Timeout: tau = ${currTimeout} min`,
        stat2: `Transaction Fragmentation: ${currFrag.toFixed(1)}%`,
        stat3: `Mean Session Length: ${currLen.toFixed(1)} events`,
        warning:
          currTimeout < 5
            ? 'Short idle timeout prematurely splits multi-step workflows; user transactions are broken into disjoint incomplete sessions'
            : currTimeout > 20
              ? 'Long timeout merges separate user sessions across long periods of inactivity, diluting temporal co-occurrence signals'
              : 'Balanced idle window captures coherent user transaction workflows without cross-session pollution',
        axisXLabel: 'Session Inactivity Gap Timeout Threshold tau (Minutes)',
        axisYLabel: 'Transaction Fragmentation Rate (%) / Mean Event Length',
        xRange: [0.5, 31] as [number, number],
        yRange: [0, 75] as [number, number],
      }
    }
  }, [logMode, param1, param2])

  const axisX = useAxis({ label: results.axisXLabel, range: results.xRange })
  const axisY = useAxis({ label: results.axisYLabel, range: results.yRange })

  return (
    <Figure
      title="Log Parsing: Drain Tree Templates, Parameter Extraction, and Sessionizing"
      state={state}
      hoverReadout={false}
      readouts={
        <>
          <Readout label="Parsing Configuration" value={results.stat1} />
          <Readout label="Primary Metric" value={results.stat2} />
          <Readout label="Operational State" value={results.stat3} />
          <Readout label="System Diagnostic" value={results.warning} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Log Stream Discretization and Quality Response">
        <Curve
          name={
            logMode === 'drain_template'
              ? 'Mined Unique Template Count (Scaled x0.5)'
              : logMode === 'variable_extraction'
                ? 'Downstream Anomaly Outlier Score (%)'
                : 'Average Session Event Sequence Length'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            logMode === 'drain_template'
              ? 'Template Grouping Accuracy GA (%)'
              : logMode === 'variable_extraction'
                ? 'Normal Empirical Operational Density (%)'
                : 'Premature Transaction Fragmentation Rate (%)'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Observed Benchmarks" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
