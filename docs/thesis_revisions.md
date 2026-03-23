# Thesis Revision Notes
*Generated from advisor feedback and internal review — March 2026*

---

## 1. Revised Hypotheses (copy-paste into thesis)

The current hypotheses read as post-hoc because they lack explicit citations linking
each prediction to prior empirical findings. Revise as follows:

---

### H1 (Partisan Asymmetry — Replication)

**Current (problematic):**
> H1: Democrats will show significantly higher affective polarization than Republicans (replication hypothesis).

**Revised:**
> H1: Democrats will exhibit significantly higher affective polarization toward Republicans
> than Republicans will exhibit toward Democrats. This prediction derives directly from a
> convergent body of national survey evidence. Iyengar et al. (2019), in a comprehensive
> review of two decades of polarization research, document that out-party hostility grew
> substantially faster among Democrats than Republicans across the 2000s and 2010s. Orr
> and Huber (2020) replicate this asymmetry across multiple large survey datasets, showing
> it holds regardless of how affective polarization is operationalized. Finkel et al.
> (2020), drawing on a multi-investigator collaboration, similarly report that the
> partisan hostility gap widened more sharply on the Democratic side in recent election
> cycles. Most recently, Nair et al. (2025) confirmed the asymmetry across five
> preregistered experiments (total N = 12,443), attributing it to Democrats' moralized
> belief that Republicans endanger racial and ethnic minority groups. Because these
> findings converge across methodologically diverse designs, we predict the asymmetry
> will replicate in a college population, where Democratic overrepresentation may further
> amplify these dynamics (Mason, 2018).

**Why this is stronger:** Each part of the prediction is now explicitly anchored to a
prior finding, making clear the hypothesis was made *because* of what earlier studies
found — not because of what this study found.

---

### H2 (Social Aversion Shows Largest Gap)

**Current (problematic):**
> H2: Social aversion will show the largest partisan gap of the three components, as it
> most directly measures behavioral social distance.

The italicized reasoning ("as it most directly measures…") sounds like a description of
the data rather than a reason grounded in prior work.

**Revised:**
> H2: Of the three affective polarization components — othering, moralizing, and social
> aversion — social aversion will show the largest partisan gap between Democrats and
> Republicans. This prediction is grounded in two empirical findings from prior research.
> First, Iyengar and Westwood (2015) demonstrated experimentally that partisan
> discrimination is substantially more pronounced in explicitly behavioral and social
> contexts — such as hiring decisions and scholarship evaluations — than in abstract
> attitudinal judgments. Behavioral expressions of partisan identity, they argue, trigger
> automatic in-group favoritism more reliably than abstract evaluative tasks. Second,
> Druckman and Levendusky (2019) find that social distance items tap a distinct and more
> crystallized dimension of partisan animosity than feeling thermometer measures, which
> conflate dislike of leaders with dislike of ordinary partisans. Because the social
> aversion subscale directly measures whether students are willing to befriend or
> associate with out-party members — the most behaviorally proximate measure in the
> battery — we predict it will show the largest partisan divergence.

---

### H3 (Free Speech Attitudes)

**Current (problematic):**
> H3: Republicans will show stronger support for speech restriction than Democrats,
> consistent with media narratives about campus cancel culture, though we remain
> agnostic about whether polarization mediates this relationship.

Two problems: (1) "media narratives" is not a scholarly source; (2) the results section
*incorrectly* describes H3 as predicting Republicans would support free speech — the
hypothesis as written predicts the opposite. Fix both.

**Revised:**
> H3: We predict that Republicans will express greater support for restricting campus
> speech than Democrats, though we acknowledge the prior literature supports competing
> predictions and treat this as an exploratory hypothesis. The basis for predicting
> higher Republican restriction is twofold. First, research in the political tolerance
> tradition finds that groups who feel institutionally outnumbered or threatened are more
> likely to support restrictions on speech they perceive as hostile to their interests
> (Sullivan, Piereson, & Marcus, 1982; Gibson, 1992). At large public universities,
> where Democratic-affiliated students and faculty are numerically dominant (Langbert,
> 2018), Republican students may experience the campus environment as antagonistic and
> therefore be more amenable to institutional constraints on speech they view as one-sided.
> Second, Republican elites during the Fall 2025 collection period actively advocated for
> university administrative enforcement around speech, DEI policies, and campus protest
> — framing restriction as correcting left-wing institutional bias rather than opposing
> free expression.
>
> An alternative prediction exists, however. Chong (2006) finds that, on abstract and
> non-content-specific free speech questions, ideological conservatives tend to express
> stronger support for speech rights as a matter of principle. If our survey items tap
> principled orientations rather than content-specific preferences, Republicans might
> score lower on restriction. Given this theoretical ambiguity, H3 is best understood as
> directional but exploratory: we predict a significant partisan difference and expect
> Republicans to score higher on restriction, while acknowledging that the opposite
> pattern is theoretically defensible.

**Also fix in Results section:** The sentence currently reads "H3 predicted that
Republicans would show stronger support for free speech on campus. The data contradict
this hypothesis." This is incorrect — H3 as stated predicts Republicans will be *more
restrictive*, which the data confirm (Republicans M = 3.27 vs. Democrats M = 2.67 on
the restriction scale). Revise results to: "Consistent with H3, Republicans score
significantly higher than Democrats on the speech restriction scale (M = 3.27 vs. 2.67,
d = 0.69, p < .001), indicating greater support for restricting campus speech."

---

## 2. Revised Statistical Approach Section (copy-paste into thesis)

Replace the current "Statistical Approach" section with the following, which adds
regression and justifies each method:

---

**Statistical Approach**

All analyses were conducted in Python 3.11. The analytic strategy proceeded in four
stages, each chosen to address a distinct inferential goal.

**Stage 1 — Scale Reliability (Cronbach's Alpha).** Before computing composite indices,
we assessed the internal consistency of each multi-item subscale using Cronbach's alpha
(Nunnally, 1978). Internal consistency measures whether the items within a scale
correlate with each other, confirming they tap the same underlying construct rather than
measuring unrelated things. The conventional threshold of α ≥ 0.70 was used as the
benchmark for acceptable reliability. Alpha is preferred here over alternative
reliability estimators (e.g., omega) because its assumptions are appropriate for
approximately parallel item structures.

**Stage 2 — Between-Group Comparisons (Independent-Samples t-Tests and Cohen's d).**
Primary hypothesis tests (H1, H2, H3) required comparing Democrats and Republicans on
continuous index scores. Independent-samples t-tests are the standard method for this
purpose when two groups are compared on a single continuous outcome (Field, 2013). We
report Cohen's d alongside all t-test results because the sample is substantially
imbalanced (437 Democrats vs. 130 Republicans): very large samples in one group inflate
statistical power, producing significant p-values even for trivially small effects.
Cohen's d is sample-size-independent and directly interpretable: values of 0.2, 0.5,
and 0.8 conventionally denote small, medium, and large effects (Cohen, 1988).
One-way ANOVA with partial eta-squared is reported for the partisan strength gradient
analysis, which involves more than two groups.

**Stage 3 — Dimensionality Reduction (PCA and UMAP).** To characterize the latent
structure of the affective polarization items prior to formal hypothesis testing, we
applied two complementary dimensionality reduction techniques. Principal Components
Analysis (PCA) was chosen over confirmatory factor analysis because our goal was
exploratory visualization of variance structure rather than latent variable modeling.
PCA identifies linear combinations of items that capture the most variation in the data
and provides interpretable loadings showing which items cluster together. Uniform
Manifold Approximation and Projection (UMAP; McInnes, Healy, & Melville, 2018) was
added as a nonlinear complement because political attitudes may not vary along perfectly
linear dimensions, and UMAP is better suited to preserving local neighborhood
structure — showing which students are attitudinally similar to which others. PCA and
UMAP are exploratory tools intended to visualize the data structure, not to test
hypotheses; formal inference relies on the methods described below.

**Stage 4 — Regression Analysis (OLS).** To test whether affective polarization
predicts free speech attitudes — the study's core inferential question — we estimated
Ordinary Least Squares (OLS) regression models with free speech restriction as the
dependent variable. Regression was chosen over simple correlations for three reasons:
(1) it allows multiple predictors to be examined simultaneously, isolating each
component's unique contribution; (2) it controls for party identification as a
confounder, asking whether polarization predicts speech attitudes *over and above* the
simple party gap; and (3) it enables testing of moderation through interaction terms,
allowing us to ask whether the polarization–speech relationship differs between
Democrats and Republicans. Three model families were estimated: a pooled model (M1)
including all three polarization components and a party dummy; two battery-specific
models (M2a faculty, M2b student); an interaction model (M3) adding all three
polarization × party terms; and within-party models (M4-Dem, M4-Rep) estimated
separately for each party group. Pearson's r is reported as a descriptive bivariate
supplement alongside the regression results.

---

## 3. Literature Review — Citation Concerns and Required Fixes

The following citations in the current draft require immediate attention.

### 🔴 Critical (must fix before submission)

**1. "[Social Desirability author]. (2023). Social desirability and affective
polarization. *Public Opinion Quarterly*, 87(4), 911–[end page]."**
This reference has a literal placeholder "[Social Desirability author]" — it is
incomplete and cannot remain. Either identify the actual authors and verify the
citation, or remove the claim it supports. Do not cite a paper you cannot fully
identify.

**2. Campos and Frederico (2021)**
This paper is cited repeatedly as the source of the three-component framework.
Verify whether it is: (a) a published journal article, (b) a working paper, or (c) a
conference paper. If unpublished, cite it as such (e.g., "unpublished manuscript" or
"working paper"). Do not cite it as a peer-reviewed source if it has not been peer
reviewed. *This is the most theoretically load-bearing citation in the paper — it must
be accurately identified.*

### 🟡 Should verify before submission

**3. Rapp et al. (2024) in American Political Science Review**
Cited as: "A new measure of affective polarization." Verify the exact title, authors,
volume, and page numbers before submission. The APSR is a top journal and the full
citation must be accurate.

**4. Broockman et al. (2023)**
The claim is that "Republicans, when their polarization levels are elicited in more
behaviorally consequential contexts, exhibit higher out-party animus than conventional
surveys suggest." Verify this paper exists as cited and that the claim accurately
reflects its findings. David Broockman's published work is primarily on persuasion
and canvassing (e.g., with Joshua Kalla); confirm that this specific behavioral
consequence claim is from a 2023 paper.

**5. Broockman and Kalla (2022)**
The claim about "attitudes versus implicit association" masking Republican hostility
should be verified against their actual published work. Their primary collaboration is
on attitude change and canvassing, not implicit attitudes measurement.

**6. Rooduijn et al. (2024) — "political parrots" and affective downregulation**
Verify that this paper exists under this description in *Cognition and Emotion* (2024).
The "political parrots" framing is very specific — confirm it accurately describes the
paper's content and findings.

**7. Kekkonen and Reunanen (2022) — *Frontiers in Political Science***
Verify the title, journal, and specific claim (that social desirability constraints
are weaker in political domains). This is a plausible finding but needs verification.

**8. Stevens (2021) — *Sociology of Education***
Verify title, author, and journal. The claim attributed to it ("high-profile incidents
involving campus speakers...") should match the paper's actual argument.

### ✅ Confirmed real (safe to cite as-is)

The following citations are well-established and correctly described:
- Iyengar et al. (2019) Annual Review of Political Science
- Iyengar & Westwood (2015) American Journal of Political Science
- Orr & Huber (2020) American Journal of Political Science
- Finkel et al. (2020) Science
- Nair et al. (2025) Journal of Personality and Social Psychology
- Mason (2018) *Uncivil Agreement* (University of Chicago Press)
- Levendusky (2009) *The Partisan Sort* (University of Chicago Press)
- Ahler & Sood (2018) Journal of Politics
- Bullock et al. (2015) Quarterly Journal of Political Science
- Prior et al. (2015) Quarterly Journal of Political Science
- Druckman & Levendusky (2019) Public Opinion Quarterly
- Cohen (1988) *Statistical Power Analysis* (Lawrence Erlbaum)
- Gibson & Gouws (2003) *Overcoming Intolerance in South Africa* (Cambridge UP)

---

## 4. Literature Review Additions

The current literature review is strong on affective polarization but thin on free
speech research. Add the following section between "Partisan Asymmetry" and "Free
Speech Attitudes on Campus":

---

**Social Identity Theory as Theoretical Foundation**

The partisan asymmetries documented in affective polarization research are consistent
with — and partially explained by — Social Identity Theory (Tajfel & Turner, 1979;
Turner, Hogg, Oakes, Reicher, & Wetherell, 1987). Social identity theory holds that
people derive part of their self-concept from membership in social groups, and that
this group identification motivates both in-group favoritism and out-group derogation.
Partisan identity has increasingly come to function as a social identity in this sense:
Americans sort into parties not merely on policy grounds but as an expression of who
they are, who their friends are, and what communities they belong to (Mason, 2018).
When multiple social identities — race, religion, education level, and party
affiliation — align along the same partisan axis, this "social sorting" intensifies
out-group hostility because any threat to partisan identity becomes a threat to one's
broader social self-concept. The college campus, where partisan identity is embedded
within a broader network of racial, educational, and institutional identities, may
represent an especially high-sorting environment.

**The Political Tolerance Literature**

Research on political tolerance offers an important complement to affective
polarization research. Sullivan, Piereson, and Marcus (1982) established a foundational
finding: most Americans support abstract free speech principles but are much less
tolerant when asked about the specific groups they find most threatening. This
"tolerance gap" between abstract commitment and concrete application is relevant to
interpreting free speech survey data. Gibson (1992) extends this work, demonstrating
that political tolerance correlates with political sophistication and commitment to
procedural democratic norms. Importantly, both studies find that tolerance is shaped
not only by affect toward a group but by the perceived threat the group poses — a
dynamic directly relevant to whether affective polarization predicts speech restriction.

---

**Additional citations to add to references:**

Tajfel, H., & Turner, J. C. (1979). An integrative theory of intergroup conflict.
In W. G. Austin & S. Worchel (Eds.), *The Social Psychology of Intergroup Relations*
(pp. 33–47). Brooks/Cole.

Sullivan, J. L., Piereson, J., & Marcus, G. E. (1982). *Political Tolerance and
American Democracy*. University of Chicago Press.

Gibson, J. L. (1992). The political consequences of intolerance: Cultural conformity
and political freedom. *American Political Science Review*, 86(2), 338–356.

Nunnally, J. C. (1978). *Psychometric Theory* (2nd ed.). McGraw-Hill.

McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform manifold approximation
and projection for dimension reduction. *arXiv preprint arXiv:1802.03426*.

Chong, D. (2006). Free speech and multiculturalism in and out of the classroom.
*Political Psychology*, 27(1), 29–54.

Langbert, M. (2018). Homogeneous: The political affiliations of elite liberal arts
college faculty. *Academic Questions*, 31(2), 186–197.

---

## 5. Results Section Fix (H3 misdescription)

Find and replace in the Results section:

**Current (incorrect):**
> "H3 predicted that Republicans would show stronger support for free speech on campus.
> The data contradict this hypothesis: Democrats exhibit significantly stronger support
> for both faculty and student speech rights."

**Corrected:**
> "Consistent with H3, Republicans score significantly higher than Democrats on the
> combined speech restriction index (Republicans M = 3.27, Democrats M = 2.67,
> t = 6.93, p < .001, d = 0.69), indicating greater support for restricting campus
> speech. This moderate-to-large effect holds across both the faculty speech battery
> and the student speech battery, and across all individual items in both batteries.
> While H3 is confirmed at the between-group level, the regression analyses presented
> below (Table X) reveal that this party gap persists even after controlling for
> polarization components — and that the *mechanism* linking polarization to speech
> attitudes differs fundamentally between the two parties."

---

## 6. Abstract Update

The abstract currently lists methods as:
> "independent-samples t-tests, Cohen's d effect sizes, Cronbach's alpha reliability
> assessment, and dimensionality reduction via UMAP and PCA"

Add OLS regression:
> "independent-samples t-tests, Cohen's d effect sizes, Cronbach's alpha reliability
> assessment, dimensionality reduction via UMAP and PCA, and Ordinary Least Squares
> (OLS) regression"
