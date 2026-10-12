import type { ReactNode } from 'react'
import { MathText } from 'aifn-render/diagram'
import { Tex } from 'aifn-render/layout'

/** A paragraph with inline `$…$` maths, optionally opening with a bold lead. */
function P({ children, lead }: { children: string; lead?: string }) {
  return (
    <p className="text-[15px] leading-relaxed text-foreground">
      {lead && <strong className="font-semibold">{lead} </strong>}
      <MathText text={children} />
    </p>
  )
}

function Eq({ children }: { children: string }) {
  return (
    <div className="my-1 overflow-x-auto">
      <Tex display>{children}</Tex>
    </div>
  )
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="space-y-3">
      <h3 className="text-base font-semibold tracking-tight">{title}</h3>
      {children}
    </section>
  )
}

const BENCH: [string, string, string, string][] = [
  [
    'smooth wave',
    'tie (GAM −0.229, ours −0.230); learned knots change nothing',
    'same as the GAM',
    'RVM 68%, SplineVM 83%; ours and GAM 93%',
  ],
  [
    'wave + sharp step',
    'ours, ν = 0.05 (−0.251 vs GAM −0.271; GAM with learned knots −0.262)',
    '5.7 vs GAM 7.0',
    'ours 88%, GAM 82% (85% with learned knots), RVM and SplineVM 61%',
  ],
  [
    'heavy-tailed noise',
    'ours with a Student-t loss (−0.84 vs −4.4 Gaussian)',
    'RVM overfits: 11',
    'ours and GAM 85–87%, RVM 73%',
  ],
  [
    '8% outliers',
    'ours with a Student-t loss (−0.72 vs −1.24 Gaussian)',
    'RVM overfits: 15',
    'ours and GAM 81–85%, RVM 64%',
  ],
  ['classification', 'tie with the GAM (−0.412)', 'same as the GAM', 'ours and GAM 98%, SplineVM 94%'],
  ['counts', 'tie with the GAM (−1.918)', '7.6 vs GAM 8.3', 'ours 89%, GAM 91%'],
]

/** The model behind the "variational GAM with adaptive smoothing" figure, written out. */
export function VbWriteup() {
  return (
    <article className="mx-auto max-w-3xl space-y-8 py-6">
      <header className="space-y-2">
        <h2 className="text-xl font-semibold tracking-tight">The model, in detail</h2>
        <P>
          {String.raw`A one-dimensional generalised additive model (GAM) term: a cubic spline $f(x)$ fed into any likelihood. Its smoothness is shared across the curve, as in a GAM, unless the data pay, in evidence, for one part of the curve to bend more freely than the rest. The model is fitted by variational Bayes, so every quantity it reports (the curve, its uncertainty, how much each knot matters) comes from one approximate posterior and one objective.`}
        </P>
      </header>

      <Section title="1 · What it is for">
        <P>
          {String.raw`The aim is a curve whose complexity follows the data: smooth where the data are smooth, sharper where they demand it, under whatever loss suits the response (squared error, a robust loss, a quantile loss, a classifier's log loss, a count model). It should also report how sure it is about the curve, and which parts of the curve the data actually determined.`}
        </P>
        <P>
          {String.raw`A flexible prior on where the curve may bend can explain the data trivially: put a bend wherever there is a residual. So the design keeps the plain GAM as the default and makes every departure from it pay. There is no prior over the number of knots, and by default the knots stay on a fixed grid, so adaptivity comes from how strongly each knot is shrunk. Knot locations can also be learned, but only on a leash around their grid points (section 4).`}
        </P>
      </Section>

      <Section title="2 · The model, top down">
        <P>
          {String.raw`The input is rescaled to $u = (x - x_{\min})/(x_{\max} - x_{\min}) \in [0, 1]$, and a grid of $M$ knots $t_1 < \dots < t_M$ is fixed evenly inside it. The curve is a cubic polynomial plus one truncated cubic per knot:`}
        </P>
        <Eq>{String.raw`f(u) = \sum_{k=0}^{3} c_k u^k + \sum_{j=1}^{M} w_j\,(u - t_j)_+^3, \qquad (z)_+ = \max(z, 0).`}</Eq>
        <P>
          {String.raw`The basis function $(u - t_j)_+^3$ is zero left of $t_j$ and cubic right of it, so its weight $w_j$ is the jump in the third derivative $f'''$ at $t_j$. A large $|w_j|$ is a sharp change of curvature at that knot; $w_j = 0$ means the curve passes through $t_j$ as a single cubic piece.`}
        </P>
        <P>{String.raw`The priors form a three-level hierarchy, and the likelihood is any loss read as a log-likelihood:`}</P>
        <Eq>
          {String.raw`\begin{aligned} \kappa &\sim \mathrm{Gamma}(a_\kappa,\, b_\kappa) && \text{the GAM's shared smoothing level} \\ \alpha_j \mid \kappa &\sim \mathrm{Gamma}(\nu,\, \nu\kappa) && \text{knot } j\text{'s relevance},\ \mathbb E[\alpha_j \mid \kappa] = 1/\kappa \\ w_j \mid \alpha_j &\sim \mathcal N(0,\, \alpha_j^{-1}) && \text{the jump in } f''' \text{ at } t_j \\ c_k &\sim \mathcal N(0,\, \alpha_c^{-1}) && \text{the cubic part, vague} \\ y_n \mid f &\sim p_\ell\big(y_n \mid f(u_n),\, s\big) && \text{the loss as a likelihood, scale } s \end{aligned}`}
        </Eq>
        <P>
          {String.raw`Read top down: the shared level $\kappa$ sets how large the weights are allowed to be in general; each knot's precision $\alpha_j$ is drawn around $1/\kappa$; each weight is drawn with that precision; the weights make the curve; the curve makes the data.`}
        </P>
      </Section>

      <Section title="3 · Why the GAM is the base, and what ν does">
        <P>
          {String.raw`If every knot had the same precision $\alpha$, the prior would be a Bayesian penalised spline: a ridge penalty on the jumps of $f'''$, with one smoothing parameter learned from the data. That is the GAM's smoothing term. The hierarchy keeps that as the centre and lets each knot depart from it.`}
        </P>
        <P>
          {String.raw`The shape $\nu$ prices the departure. As $\nu \to \infty$ the Gamma prior on $\alpha_j$ collapses onto $1/\kappa$ and the model is exactly the GAM. For small $\nu$ the $\alpha_j$ spread out: a knot can get a much smaller precision (freedom to bend) or a much larger one (pinned to zero), as in automatic relevance determination (ARD). Integrating $\alpha_j$ out shows what the prior does to a weight: $w_j$ is Student-$t$ with $2\nu$ degrees of freedom, so small $\nu$ means a sharp peak at zero and heavy tails, which is a sparsity-inducing prior. A knot moves away from the GAM's level only when its likelihood gain outweighs the prior's cost of being unusual, in the spirit of penalised-complexity priors, which shrink towards the simpler base model.`}
        </P>
        <P>
          {String.raw`The hyperprior on $\kappa$ is vague on a log scale, with its rate set from the data so that $\mathbb E[\kappa] = 1/\alpha_0$, where $\alpha_0$ is a rough size for the weights (a jump $w$ changes $f$ by about $w h^3$ over a knot spacing $h$). It must be scaled: $\kappa$ behaves like a prior variance of the weights and can be $10^7$ on this basis, where a textbook $\mathrm{Gamma}(10^{-3}, 10^{-3})$ would act as a wall.`}
        </P>
      </Section>

      <Section title="4 · Inference: variational Bayes">
        <P>
          {String.raw`The posterior over $(\mathbf w, \boldsymbol\alpha, \kappa)$, with $\mathbf w$ holding the cubic part and the knot weights, is approximated by a factorised distribution, and the loss scale $s$ is fitted as a point estimate:`}
        </P>
        <Eq>{String.raw`q(\mathbf w, \boldsymbol\alpha, \kappa) = \mathcal N(\mathbf w;\, \boldsymbol\mu, \boldsymbol\Sigma)\;\prod_{j=1}^{M} \mathrm{Gamma}(\alpha_j;\, a_j, b_j)\;\mathrm{Gamma}(\kappa;\, a, b).`}</Eq>
        <P>{String.raw`The objective is the evidence lower bound (ELBO): the expected fit minus the divergence of $q$ from the prior.`}</P>
        <Eq>{String.raw`\mathcal L(q) = \underbrace{\sum_{n=1}^{N} \mathbb E_q\big[\log p_\ell(y_n \mid f_n, s)\big]}_{\text{fit}} \;-\; \underbrace{\mathrm{KL}\big(q(\mathbf w, \boldsymbol\alpha, \kappa)\,\big\|\,p(\mathbf w, \boldsymbol\alpha, \kappa)\big)}_{\text{complexity}} \;\le\; \log p(\mathbf y).`}</Eq>
        <P lead="The relevances.">
          {String.raw`The Gamma factors have closed-form updates: $q(\alpha_j) = \mathrm{Gamma}\big(\nu + \tfrac12,\; \nu\,\mathbb E[\kappa] + \tfrac12\,\mathbb E[w_j^2]\big)$ and $q(\kappa) = \mathrm{Gamma}\big(a_\kappa + M\nu,\; b_\kappa + \nu \sum_j \mathbb E[\alpha_j]\big)$, with $\mathbb E[w_j^2] = \mu_j^2 + \Sigma_{jj}$. For large $\nu$ the two are so tightly coupled that alternating them moves $\mathbb E[\kappa]$ by about $1/(2\nu)$ per pass, so their joint fixed point is solved directly: it is the root of a function of $\mathbb E[\kappa]$ that increases monotonically, found by bisection.`}
        </P>
        <P lead="The weights, under any loss.">
          {String.raw`Under $q$, each $f_n = \boldsymbol\phi_n^\top \mathbf w$ is Gaussian, $f_n \sim \mathcal N(m_n, v_n)$ with $m_n = \boldsymbol\phi_n^\top\boldsymbol\mu$ and $v_n = \boldsymbol\phi_n^\top\boldsymbol\Sigma\boldsymbol\phi_n$. So the fit term is a sum of one-dimensional Gaussian expectations, computed by 24-point Gauss–Hermite quadrature for any loss. Their derivatives in $m_n$ need only loss values (Stein's identities, with $z$ standard normal):`}
        </P>
        <Eq>{String.raw`g_n = \frac{\partial}{\partial m_n}\mathbb E[\ell_n] = \frac{\mathbb E[z\,\ell_n]}{\sqrt{v_n}}, \qquad h_n = \frac{\partial^2}{\partial m_n^2}\mathbb E[\ell_n] = \frac{\mathbb E[(z^2 - 1)\,\ell_n]}{v_n}, \qquad \ell_n = \log p_\ell(y_n \mid m_n + \sqrt{v_n}\,z).`}</Eq>
        <P>
          {String.raw`Because the expectation smooths the loss over $q$'s spread, non-smooth losses (absolute error, pinball) work without special cases. The update for $q(\mathbf w)$ is the variational-Gaussian fixed point (Opper and Archambeau, 2009), a Newton step on the ELBO: $\boldsymbol\Sigma = \big(\mathbb E[\mathbf A] + \boldsymbol\Phi^\top \mathrm{diag}(-h_n)\,\boldsymbol\Phi\big)^{-1}$ and $\boldsymbol\mu \leftarrow \boldsymbol\mu + \boldsymbol\Sigma\big(\boldsymbol\Phi^\top \mathbf g - \mathbb E[\mathbf A]\boldsymbol\mu\big)$, halved until the ELBO does not fall. Here $\mathbb E[\mathbf A]$ is the diagonal of prior precisions. For the Gaussian loss one step is exact.`}
        </P>
        <P lead="The scale.">
          {String.raw`Where the loss has a scale (Gaussian, Student-$t$, Laplace, Huber, pinball), $s$ maximises the expected fit: in closed form for the Gaussian, $s^2 = \frac1N\sum_n \big[(y_n - m_n)^2 + v_n\big]$, and by a golden-section search on $\log s$ otherwise.`}
        </P>
        <P lead="The start.">
          {String.raw`The fit starts at the maximum a posteriori (MAP) point under the starting relevances, found by Newton's method, with $q(\mathbf w)$ the Laplace approximation there; kinked losses are smoothed at a third of the noise scale for this start only. Starting from the prior instead fails: $q(\mathbf w)$ is then so wide that the expected curvature of a logistic, Poisson or pinball loss is nearly zero, and the first step overshoots. From the start, the updates cycle (weights, relevances, scale) and the ELBO rises at every iteration until it stops changing.`}
        </P>
        <P lead="Knot locations (optional).">
          {String.raw`With "learn knot locations" on, each knot gets a leash prior $t_j \sim \mathcal N\big(g_j,\, (c\,h)^2\big)$ around its grid point $g_j$, with $h$ the grid spacing and $c$ the leash in spacings, and the locations are point-estimated on $\mathcal L(q) + \log p(\mathbf t)$. Only the fit term depends on $\mathbf t$, through the design. By the Bonnet and Price identities, $\partial \sum_n \mathbb E[\ell_n] / \partial \boldsymbol\phi_n = g_n \boldsymbol\mu + h_n \boldsymbol\Sigma \boldsymbol\phi_n$, and $\partial \phi_{nj} / \partial t_j = -3\,(u_n - t_j)_+^2$, so the gradient costs no more than the weight update. Each iteration takes one normalised step along it, and a trial position is scored with $q(\mathbf w)$ refitted to it (one Newton step, exact for the Gaussian loss). Scoring with the old weights would reject nearly every move, because those weights are tuned to the old basis. The leash keeps the knots from chasing single residuals. On the wave-plus-step data, knots slide towards the step and the plain GAM's fit there improves; on smooth data they barely move.`}
        </P>
      </Section>

      <Section title="5 · Reading the information">
        <P>
          {String.raw`A knot's weight starts with prior variance $1/\mathbb E[\alpha_j]$ and ends with posterior variance $\Sigma_{jj}$. The data have told us`}
        </P>
        <Eq>{String.raw`\text{bits}_j = \tfrac12 \log_2 \frac{1/\mathbb E[\alpha_j]}{\Sigma_{jj}} = -\tfrac12 \log_2(1 - \gamma_j), \qquad \gamma_j = 1 - \mathbb E[\alpha_j]\,\Sigma_{jj} \in [0, 1].`}</Eq>
        <P>
          {String.raw`$\gamma_j$ is the relevance vector machine's well-determinedness: near 1 when the data pin the weight down, near 0 when only the prior does. Its sum $\sum_j \gamma_j$ is the effective number of knots the curve uses. One bit corresponds to $\gamma_j = 0.75$; the figure calls a knot informative above half a bit ($\gamma_j > \tfrac12$).`}
        </P>
        <P>
          {String.raw`The ELBO puts these in one currency: the fit term pays for information in nats, the complexity term charges for it. A knot that departs from the GAM's level has to buy that departure with fit. The readout "ELBO − plain GAM's" compares the fitted model with a plain GAM on the same data: positive means the adaptivity paid for itself.`}
        </P>
      </Section>

      <Section title="6 · Losses and what the figure draws">
        <P>
          {String.raw`Gaussian (squared error), Student-$t$ with $\nu = 4$ (robust), Laplace (absolute error, the median), Huber ($\delta = 1.345$ in units of $s$), pinball (the $\tau$-quantile, as the asymmetric Laplace likelihood), logistic (Bernoulli with a logit link) and Poisson (log link). Each is a normalised log-likelihood, so the ELBO is comparable across models on the same data.`}
        </P>
        <P>
          {String.raw`The figure draws the curve through the loss's link (the identity, the class probability $\sigma(f)$, the rate $e^f$) with two bands: $\pm 2$ sd of $f$, the uncertainty about the curve, and for the Gaussian loss $\pm 2$ sd of a new $y$, which adds the noise $s^2$. The band about $f$ narrows roughly as $1/\sqrt{n}$; the band about $y$ does not, because the noise does not shrink with more data.`}
        </P>
      </Section>

      <Section title="7 · What it costs">
        <P>
          {String.raw`One iteration forms $\boldsymbol\Phi^\top \mathrm{diag}(-h)\,\boldsymbol\Phi$ in $O(N M^2)$, factors it in $O(M^3)$ and evaluates $N \times 24$ loss values. With $N = 200$ and $M = 30$ a fit takes tens of iterations and 50–600 ms in the browser; the relevance and scale updates are negligible. A sampler on the same model costs the same per step and needs thousands of steps.`}
        </P>
        <P>
          {String.raw`The truncated-cubic basis is badly conditioned (columns for knots near the right end are tiny), so every factorisation first rescales the matrix to unit diagonal, which is exact and changes only the rounding, and adds no jitter. A pivot floor of $10^{-12}$ of the diagonal was enough to distort $\boldsymbol\Sigma$ badly in an earlier version.`}
        </P>
      </Section>

      <Section title="8 · Assumptions and limits">
        <P lead="Mean field.">
          {String.raw`$q$ treats the weights and the relevances as independent. That ignores uncertainty in $\boldsymbol\alpha$ when computing $\boldsymbol\Sigma$, so the bands are somewhat too narrow, more so for small $\nu$: strongly shrunk knots carry almost no variance, the same over-confidence the relevance vector machine is known for.`}
        </P>
        <P lead="Knot placement.">
          {String.raw`By default placement is by selection: which grid knots get freedom, never where a knot moves. A feature narrower than the knot spacing (a step of width 0.01 against a spacing of 0.03 with 30 knots) cannot be represented, and no band from this basis covers it; that is most of the missing coverage on the default data. Learned locations help there, but they are point estimates: the bands carry no uncertainty about where the knots are, and maximising over $\mathbf t$ makes the ELBO optimistic. The readout comparing with the plain GAM subtracts $\log p(\mathbf t)$ so that both ELBOs are on the same footing.`}
        </P>
        <P lead="ν is set by hand.">
          {String.raw`It could be learned, by the ELBO or held-out likelihood, or given its own prior; for now the figure compares it with the plain GAM.`}
        </P>
        <P lead="Point-estimated scale.">
          {String.raw`The loss scale $s$ is a type-II estimate, not integrated over.`}
        </P>
      </Section>

      <Section title="9 · How it compares">
        <P>
          {String.raw`A small held-out test (6 cases, 5 seeds, $n = 200$ training and 800 test points; .scratch/splinevm/bench) against the plain GAM ($\nu = 1000$, same code), the engine's relevance vector machine with a radial basis kernel (lengthscale by evidence) and the type-II SplineVM of the first page. Test log predictive density per point, higher is better; differences under about 0.005 are within the noise of five seeds.`}
        </P>
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-sm">
            <thead>
              <tr className="border-b text-left">
                <th className="py-1.5 pr-3 font-semibold">case</th>
                <th className="py-1.5 pr-3 font-semibold">best test density</th>
                <th className="py-1.5 pr-3 font-semibold">effective knots Σγ</th>
                <th className="py-1.5 font-semibold">truth inside ± 2 sd of f</th>
              </tr>
            </thead>
            <tbody>
              {BENCH.map(([c, best, eff, cover]) => (
                <tr key={c} className="border-b border-border/50 align-top">
                  <td className="py-1.5 pr-3">{c}</td>
                  <td className="py-1.5 pr-3">{best}</td>
                  <td className="py-1.5 pr-3">{eff}</td>
                  <td className="py-1.5">{cover}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <P>
          {String.raw`The adaptive prior earns its keep where the curve has a local sharp feature: there it beats the plain GAM on test density, error and calibration with fewer effective knots. On smooth curves, classification and counts it settles on the GAM, as designed. The largest gains on noisy data come from the likelihood, not the prior: a Student-$t$ loss raises the test log density per point from −4.4 to −0.84 under heavy-tailed noise and from −1.24 to −0.72 under outliers. The relevance vector machine and the type-II SplineVM are over-confident (61–83% coverage), and the relevance vector machine over-fits outliers. Learned knot locations (leash of 3 spacings) recover about half of the adaptive prior's gain for a plain GAM on the step, add nothing on top of adaptive relevances (−0.259 against −0.257) and take 3 to 5 times as long.`}
        </P>
      </Section>
    </article>
  )
}
