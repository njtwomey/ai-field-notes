import { it } from 'vitest'
import { stream } from 'aifn/foundation/random'
import { webTraffic } from 'aifn-applied/data/synthetic'
import { milletRun } from 'aifn-applied/learning/weak-supervision'
it('scratch', () => {
  const t0 = performance.now()
  const train = webTraffic(stream('train'), { perClass: 30, classes: [0, 1, 5, 8], samplesPerDay: 24 })
  const test = webTraffic(stream('test'), { perClass: 15, classes: [0, 1, 5, 8], samplesPerDay: 24 })
  console.log('data ms', (performance.now() - t0).toFixed(0))
  for (const pooling of ['conjunctive', 'embedding'] as const) {
    const t1 = performance.now()
    const it = milletRun({ train, test: { ...test, discriminatory: test.discriminatory }, pooling, steps: 400, evaluate: 20, stepSize: 0.01 })
    let last
    for (const s of it) { last = s; if (s.step % 200 === 0) console.log(pooling, s.step, s.history.loss.at(-1)?.toFixed(3), s.history.trainAccuracy.at(-1)?.toFixed(2), s.history.testAccuracy.at(-1)?.toFixed(2), ((performance.now() - t1) / 1000).toFixed(1) + 's') }
    console.log(pooling, 'scores', JSON.stringify({ acc: last!.scores?.accuracy, aopcr: last!.scores?.aopcr, ndcg: last!.scores?.ndcg }), ((performance.now() - t1) / 1000).toFixed(1) + 's')
  }
}, 600000)
