import type { Specimen } from '@lab/specimen'
import { MomentumSpecimen, TimeSlicedSpecimen, DivergenceSpecimen, HeronSpecimen, SeekSpecimen } from './_trace/figures'

export const specimens: Specimen[] = [
  {
    module: 'part-1-mathematical-foundations/probability/stochastic-processes',
    title: 'Heavy-ball descent, traced',
    description:
      'A traced run of gradient descent with momentum on an ill-conditioned quadratic: scrub or play the steps, drag the step cursor on any chart, and inspect the state and the per-phase timing.',
    tags: ['trace', 'TracePanel', 'optimisation', 'profile'],
    render: () => <MomentumSpecimen />,
  },
  {
    module: 'part-1-mathematical-foundations/probability/stochastic-processes',
    title: 'Time-sliced random walks',
    description:
      'timeSliced runs eight random walks over animation frames and streams partial traces into the view; raising the step count calls extend, which computes only the new steps.',
    tags: ['timeSliced', 'extend', 'stream', 'random walk'],
    render: () => <TimeSlicedSpecimen />,
  },
  {
    module: 'part-1-mathematical-foundations/probability/stochastic-processes',
    title: 'Divergence is reported',
    description:
      'Gradient descent diverges once the learning rate passes 2/L; the state flags it and the trace stops with stopped = diverged.',
    tags: ['diverged', 'stop reason'],
    render: () => <DivergenceSpecimen />,
  },
  {
    module: 'part-1-mathematical-foundations/probability/stochastic-processes',
    title: 'Early stop on done',
    description: "Heron's square-root iteration stops as soon as done(state) holds, long before the step limit.",
    tags: ['done', 'Newton'],
    render: () => <HeronSpecimen />,
  },
  {
    module: 'part-1-mathematical-foundations/probability/stochastic-processes',
    title: 'seek from checkpoints',
    description:
      'seek(i) restarts from the nearest stored checkpoint (every 1,000 steps of 200,000) and returns exactly run(i), in a fraction of the time.',
    tags: ['seek', 'checkpoints', 'run'],
    render: () => <SeekSpecimen />,
  },
]
