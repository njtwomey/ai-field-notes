import { Diagram, Figure } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'

const spec: DiagramSpec = {
  unit: 40,
  nodes: [
    { id: 'parent', x: 0, y: 0, w: 3.6, h: 1.2, label: 'customers\none row per customer', tone: 0 },
    { id: 'child', x: 0, y: 3, w: 3.6, h: 1.2, label: 'orders\nmany rows per customer', tone: 'neutral' },
    { id: 'cutoff', x: 5, y: 0, w: 3.6, h: 1.2, label: 'cutoff time $t_0$\nfor each training row', tone: 'neutral' },
    {
      id: 'filter',
      x: 5,
      y: 3,
      w: 3.6,
      h: 1.2,
      label: 'keep orders with\n$t_0 - w \\le$ time $< t_0$',
      tone: 1,
    },
    {
      id: 'agg',
      x: 10,
      y: 3,
      w: 3.8,
      h: 1.2,
      label: 'group by customer:\ncount, sum, mean, latest',
      tone: 1,
    },
    { id: 'row', x: 10, y: 0, w: 3.8, h: 1.2, shape: 'pill', label: 'one feature row\nper (customer, $t_0$)', tone: 0 },
  ],
  edges: [
    { from: 'child', to: 'filter' },
    { from: 'cutoff', to: 'filter' },
    { from: 'filter', to: 'agg' },
    { from: 'agg', to: 'row', label: 'join on customer key', labelSide: 'right' },
    { from: 'parent', to: 'cutoff' },
  ],
}

/** How aggregates over a child table become features of the parent entity at a cutoff time. */
export function AggregationDiagram() {
  return (
    <Figure
      title="From a child table to features of an entity"
      hoverReadout={false}
      caption="Each training row is a customer at a cutoff time t₀, the moment the prediction is made. Only the customer's orders from before t₀, and within a window of length w, are kept. They are grouped by customer and summarised by aggregation functions, and the result is joined back to the customer as one row of features. Repeating this for several windows and functions gives a block of features per related table."
    >
      <Diagram
        spec={spec}
        ariaLabel="Customers and a cutoff time per training row select the customer's orders inside a window before the cutoff; the orders are grouped by customer and aggregated into counts, sums, means and the latest value; the aggregates are joined back to the customer as one feature row"
      />
    </Figure>
  )
}
