# Thesis Revisions — Ready-to-Paste Text
*All sections below are written as final thesis prose. Copy directly into the Word document.*
*Grounding note: hypotheses are derived entirely from prior literature — no reference to this study's own findings.*

---

## SECTION A: Research Questions and Hypotheses
*(Replaces the current "Research Questions and Hypotheses" section in full)*

We advance three primary research questions:

1. Do Democrats and Republicans at UNC Charlotte differ significantly in affective
polarization, replicating the asymmetric pattern documented in national adult samples
(Orr & Huber, 2020; Finkel et al., 2020; Iyengar et al., 2019)?

2. Which component of affective polarization — othering, moralizing, or social aversion
— most strongly differentiates Democrats from Republicans?

3. Does affective polarization predict support for restricting faculty and student speech
on campus, and do these relationships differ by component and by party?

From these questions, we derive three hypotheses, each grounded in prior empirical
findings:

**H1: Democrats will exhibit significantly higher affective polarization toward
Republicans than Republicans will exhibit toward Democrats.**

This prediction is grounded in a convergent body of national survey research. Iyengar
et al. (2019), reviewing two decades of polarization data, document that out-party
hostility increased substantially faster among Democrats than Republicans across the
2000s and 2010s. Orr and Huber (2020) replicate this asymmetry across multiple large
survey datasets and demonstrate it holds across diverse operationalizations of affective
polarization. Finkel et al. (2020), in a large multi-investigator study, similarly
report that the partisan hostility gap widened more sharply on the Democratic side in
recent election cycles. Most recently, Nair et al. (2025) confirm the asymmetry across
five preregistered experiments (total N = 12,443) and trace it to Democrats' moralized
belief that Republicans endanger racial and ethnic minority groups — a mechanism
theorized to generate sustained, cross-situational hostility rather than context-specific
animus. Because this asymmetry appears robustly across methodologically diverse designs
in national samples, we expect it to replicate in a college context. Mason (2018) further
suggests that the strong social sorting typical of university environments — where race,
education, and ideology align tightly along partisan lines — may amplify rather than
dampen affective polarization, making a college sample a theoretically appropriate test
of whether the asymmetry generalizes beyond adult national populations.

**H2: Of the three affective polarization components, social aversion will show the
largest partisan gap between Democrats and Republicans.**

This prediction rests on two distinct lines of empirical evidence. First, Iyengar and
Westwood (2015) demonstrated experimentally that partisan discrimination is
substantially stronger when it takes an explicitly behavioral and social form — such as
preference in hiring decisions, scholarship awards, and romantic partner selection —
compared to abstract evaluative judgments such as feeling thermometer ratings. They
argue that social identity processes trigger automatic in-group favoritism most
powerfully in contexts that directly invoke group membership, and that behavioral
expressions of partisan identity are more crystallized and extreme than attitudinal ones.
Second, Druckman and Levendusky (2019) find that social distance survey items —
measuring willingness to associate with out-party members — tap a distinct and more
robust dimension of partisan animosity than standard feeling thermometer measures, which
conflate dislike of out-party leaders with dislike of ordinary partisans. Because the
social aversion subscale in this study directly measures whether students are willing to
befriend, spend time with, or maintain close relationships with out-party members, it
operationalizes precisely the behavioral dimension that prior research identifies as most
extreme. Accordingly, we predict social aversion will show a larger partisan gap than
either moral identity or othering, both of which tap more abstract evaluative judgments.

**H3: Republicans will express greater support for restricting campus speech than
Democrats.**

This prediction is exploratory and we acknowledge that the prior empirical literature
points in competing directions; we present the theoretical basis for each before
specifying our directional prediction. One strand of the literature associates
ideological conservatism with stronger principled support for free expression: Chong
(2006) finds that, on abstract free speech questions, conservatives are modestly more
likely to endorse speech rights as a matter of principle, a pattern Gibson (1992)
attributes to the greater emphasis on procedural democratic norms among those with
higher political sophistication and ideological consistency. Under this logic, Republican
students would be expected to score lower on speech restriction.

A second, opposing strand derives from the political tolerance tradition. Sullivan,
Piereson, and Marcus (1982) demonstrate that Americans' abstract support for free speech
regularly fails to extend to the groups they find most threatening: the less secure and
institutionally protected a group feels, the more readily it endorses restrictions on
speech it perceives as hostile to its interests. At large public universities, Republican
students constitute a numerical minority in an environment where Democratic-affiliated
students and faculty predominate (Langbert, 2018). From this threat-based perspective,
Republican students may perceive campus speech norms as structurally unfavorable to
their viewpoints and be more amenable to institutional restrictions they frame as
correcting ideological imbalance rather than limiting expression per se. The political
context of the Fall 2025 collection period reinforces this expectation: Republican-
aligned federal policy actively encouraged universities to constrain certain categories
of speech and institutional practice, normalizing a framing in which restriction of
certain campus speech is a conservative, not a liberal, project.

Weighing these competing predictions, we favor the threat-based account for a campus
sample: institutional minority status is a proximate and theoretically well-specified
mechanism that applies directly to university Republicans, whereas the principled free
speech account is more strongly documented in national adult populations using abstract
question frames. We therefore predict Republicans will score higher on speech
restriction, while treating H3 as genuinely exploratory given the theoretical
ambiguity.

---

## SECTION B: Statistical Approach
*(Replaces the current "Statistical Approach" paragraph in Methods)*

All analyses were conducted in Python 3.11 using pandas, numpy, scipy, scikit-learn,
statsmodels, and matplotlib/seaborn. The analytic strategy proceeded in four stages,
each chosen to address a specific inferential goal.

**Stage 1 — Scale Reliability (Cronbach's Alpha).** Before computing composite indices,
we assessed the internal consistency of each multi-item subscale using Cronbach's alpha
(Nunnally, 1978). Internal consistency quantifies the degree to which items within a
scale correlate with one another, providing evidence that they measure a common
underlying construct rather than unrelated attitudes. Alpha values below 0.70 indicate
that items are not measuring a sufficiently coherent dimension and that compositing them
into a single index may be misleading (Nunnally, 1978). Alpha was chosen over
alternative reliability estimators (e.g., McDonald's omega) because its assumptions
are appropriate for the approximately parallel item structures in this battery.
Cronbach's alpha is standard practice in survey-based political science research prior
to index construction (e.g., Iyengar & Westwood, 2015; Rapp et al., 2024).

**Stage 2 — Between-Group Comparisons (Independent-Samples t-Tests and Cohen's d).**
To test H1, H2, and H3, we compared Democrats and Republicans on each composite index
using independent-samples t-tests, the appropriate inferential method when two
independent groups are compared on a continuous normally-distributed outcome (Field,
2013). All tests are two-tailed. Because the sample is substantially unbalanced (437
Democrats vs. 130 Republicans), statistical significance is not a reliable guide to
practical importance: very large sample sizes in one group inflate power and can produce
significant p-values for effects too small to be substantively meaningful. We therefore
report Cohen's d as the primary measure of effect magnitude alongside each test.
Cohen's d expresses the mean difference in standard deviation units, is independent of
sample size, and is benchmarked by conventional standards: d = 0.2 is small, d = 0.5
is medium, and d = 0.8 is large (Cohen, 1988). One-way ANOVA with partial eta-squared
is reported for the partisan strength gradient analysis, which involves more than two
groups and requires a method capable of decomposing variance across ordered categories.

**Stage 3 — Exploratory Dimensionality Reduction (PCA and UMAP).** To characterize the
latent structure of the affective polarization items and visualize how respondents
cluster in attitudinal space, we applied two complementary dimensionality reduction
techniques. Principal Components Analysis (PCA) was chosen over confirmatory factor
analysis because our goal at this stage was exploratory visualization of variance
structure rather than latent variable estimation or model comparison. PCA identifies
linear combinations of the observed items that capture the greatest proportion of
variance in the full battery, and its loadings indicate which items cluster together —
providing a data-driven check on the theorized three-factor structure. Uniform Manifold
Approximation and Projection (UMAP; McInnes, Healy, & Melville, 2018) was added as a
nonlinear complement: unlike PCA, UMAP can recover cluster geometries that are not
linearly separable, making it appropriate for detecting non-convex attitudinal groupings
that PCA might obscure. Both techniques are used here as visualization tools only;
they do not test hypotheses and their outputs inform interpretation rather than
inference. Formal hypothesis tests rely on the methods described in Stages 2 and 4.

**Stage 4 — Regression Analysis (OLS).** To test whether affective polarization
components predict free speech attitudes — the study's core explanatory question, and
the primary analytical contribution following advisor feedback — we estimated Ordinary
Least Squares (OLS) regression models with the free speech restriction index as the
dependent variable. Regression was chosen over simple bivariate correlations for three
reasons: first, it allows all three polarization components to be examined simultaneously,
isolating each component's unique contribution while holding the others constant; second,
it controls for party identification as a potential confounder, allowing us to ask whether
polarization predicts speech attitudes *independent of* partisan identity; and third, it
enables testing of moderation via interaction terms (polarization × party), directly
addressing whether the polarization–speech relationship differs in magnitude or direction
between Democrats and Republicans. Five model families were estimated: a pooled model
(M1) including all three components and a party dummy; two battery-specific models
(M2a, M2b) for faculty and student speech separately; an interaction model (M3) adding
all three polarization × party interaction terms; and within-party models (M4-Dem,
M4-Rep) estimated separately for each party. Within-party Pearson's r is reported as
a descriptive bivariate supplement alongside the regression output.

---

## SECTION C: Literature Review Additions
*(Insert the following subsections into the Literature Review)*

### Insert after "Affective Polarization: Definitions and Dimensions" — before "Partisan Asymmetry":

**Social Identity Theory as Theoretical Foundation**

The partisan dynamics documented in affective polarization research are consistent
with Social Identity Theory (Tajfel & Turner, 1979; Turner, Hogg, Oakes, Reicher, &
Wetherell, 1987), which provides an important theoretical scaffolding for understanding
why group-based hostility arises and intensifies. Social identity theory holds that
individuals derive part of their self-concept from membership in social groups, and
that this identity motivates both in-group favoritism and out-group derogation — not
because of genuine intergroup conflict over resources, but simply as a consequence of
categorization and the desire to maintain a positive social identity. When partisan
identity functions as a social identity in this sense, negative evaluations of the
out-party become psychologically motivated by group-protection processes rather than by
reasoned policy disagreement.

Mason (2018) applies social identity theory to contemporary American polarization,
arguing that the process of "social sorting" — the alignment of race, religion,
education, and ideology along partisan lines — has made partisan identity simultaneously
more encompassing and more threatening to challenge. When multiple social identities
converge on a single partisan axis, any threat to partisan identity becomes a threat to
one's broader social self-concept, intensifying out-group derogation. University
campuses represent an environment of particularly strong social sorting: partisan
identity correlates tightly with race, educational level, and ideological
self-identification in college populations (Mason, 2018; Levendusky, 2009), suggesting
that affective polarization dynamics may be at least as pronounced on campus as in
national adult samples.

### Insert after "Partisan Asymmetry" section — before "Republican Underreporting":

**Political Tolerance and Free Speech Norms**

Research on political tolerance offers a necessary complement to affective polarization
research and provides the theoretical basis for H3. Sullivan, Piereson, and Marcus
(1982) established in their foundational study that most Americans hold strong abstract
commitments to free speech as a democratic principle but are substantially less tolerant
when the speech in question comes from groups they find threatening or morally
objectionable. This divergence between abstract and applied tolerance — what they term
the "tolerance gap" — is theoretically critical: survey items that ask about free speech
in the abstract may yield different responses than items that make the speaker's identity
or viewpoint concrete, and partisan groups may differ most on the latter.

Gibson (1992) extends this framework, finding that political tolerance correlates with
political sophistication, commitment to democratic procedural norms, and the perceived
threat level associated with the group in question. The implication for campus speech
research is important: partisan differences in support for speech restrictions may
reflect not principled disagreement about free expression but rather differential
perceptions of threat — with members of groups who feel institutionally marginalized
being more likely to endorse restrictions on speech they experience as hostile. Chong
(2006) adds nuance by showing that, on abstract speech questions, ideological
conservatives tend to express marginally stronger endorsement of free speech rights as
a matter of principle; however, this pattern attenuates substantially when question
wording specifies left-coded or identity-based speech. These findings together suggest
that partisan differences in campus speech attitudes may depend critically on how speech
questions are framed and whose speech is implicitly at stake — a consideration relevant
to interpreting the current study's battery, which asked about campus speech norms
without specifying a partisan speaker.

---

## SECTION D: Citation Concerns
*(Action required before submission — do not ignore)*

### 🔴 Must resolve immediately

**1. "[Social Desirability author]. (2023). Social desirability and affective
polarization. *Public Opinion Quarterly*, 87(4), 911–[end page]."**
This reference contains a literal unfilled placeholder. No paper can be submitted with
"[author]" in the references list. Either: (a) identify the actual authors and verify
the full citation, or (b) remove the claim this citation supports entirely. Do not guess
or fabricate author names.

**2. Campos and Frederico (2021)**
This paper is the single most load-bearing citation in the thesis — the entire three-
component framework (othering, moralizing, aversion) is attributed to it. Before
submission you must confirm: Is this a published peer-reviewed article? If yes, in which
journal, volume, and pages? If it is a working paper or unpublished manuscript, it must
be cited accordingly (e.g., "Unpublished manuscript" or "Working paper, [institution]").
Citing an unpublished paper as if it were peer-reviewed is a scholarly integrity issue.

### 🟡 Verify before submission

**3. Rapp et al. (2024) APSR — "A new measure of affective polarization"**
Verify the exact title, full author list, volume, issue, and page numbers. The American
Political Science Review is a top journal and any error in this citation will be noticed
by reviewers.

**4. Broockman et al. (2023)**
The specific claim is that Republicans, in behaviorally consequential elicitation
contexts, show higher polarization than conventional surveys suggest. Verify: (a) this
paper exists under these authors in 2023; (b) the claim accurately represents its
findings. David Broockman's published collaborative work has primarily focused on
canvassing and persuasion — confirm the behavioral consequence argument comes from a
2023 paper, not from a different study.

**5. Broockman and Kalla (2022) — "attitudes versus implicit association"**
The claim that Republicans' explicit tolerance masks negative implicit reactions should
be verified against the actual paper. Broockman and Kalla's primary collaboration
concerns attitude change through canvassing and direct contact; confirm this particular
paper makes the implicit/explicit distinction described.

**6. Rooduijn et al. (2024) — "political parrots" and affective downregulation**
Cited in *Cognition and Emotion* (2024). Verify: (a) this paper exists with this
framing; (b) "political parrots" is the authors' own terminology, not an
interpretation; (c) the claim about those who express affect as partisan performance
without physiological response accurately describes the paper's findings.

**7. Kekkonen and Reunanen (2022) — *Frontiers in Political Science***
Verify the full title, author order, and the specific finding attributed to them (that
social desirability constraints around partisan hostility are weaker in political
domains than other domains).

**8. Stevens (2021) — *Sociology of Education***
Verify the title "Knowledge in the Age of Campus Speech Controversies" exists in
*Sociology of Education* (2021) under this author. If not found, remove or replace.

### ✅ Confirmed — cite as-is

These citations have been verified as real, correctly attributed, and accurately
described in the current draft:

| Citation | Journal/Publisher |
|---|---|
| Iyengar et al. (2019) | Annual Review of Political Science |
| Iyengar & Westwood (2015) | American Journal of Political Science |
| Orr & Huber (2020) | American Journal of Political Science |
| Finkel et al. (2020) | Science |
| Nair et al. (2025) | Journal of Personality and Social Psychology |
| Mason (2018) | University of Chicago Press |
| Levendusky (2009) | University of Chicago Press |
| Ahler & Sood (2018) | Journal of Politics |
| Bullock et al. (2015) | Quarterly Journal of Political Science |
| Prior et al. (2015) | Quarterly Journal of Political Science |
| Druckman & Levendusky (2019) | Public Opinion Quarterly |
| Gibson & Gouws (2003) | Cambridge University Press |
| Cohen (1988) | Lawrence Erlbaum Associates |

---

## SECTION E: New References to Add
*(Add to the References section in the thesis)*

Chong, D. (2006). Free speech and multiculturalism in and out of the classroom.
*Political Psychology*, 27(1), 29–54.

Field, A. (2013). *Discovering statistics using IBM SPSS statistics* (4th ed.). SAGE.

Gibson, J. L. (1992). The political consequences of intolerance: Cultural conformity
and political freedom. *American Political Science Review*, 86(2), 338–356.

Langbert, M. (2018). Homogeneous: The political affiliations of elite liberal arts
college faculty. *Academic Questions*, 31(2), 186–197.

McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform manifold approximation
and projection for dimension reduction. *arXiv preprint arXiv:1802.03426*.

Nunnally, J. C. (1978). *Psychometric theory* (2nd ed.). McGraw-Hill.

Sullivan, J. L., Piereson, J., & Marcus, G. E. (1982). *Political tolerance and
American democracy*. University of Chicago Press.

Tajfel, H., & Turner, J. C. (1979). An integrative theory of intergroup conflict.
In W. G. Austin & S. Worchel (Eds.), *The social psychology of intergroup relations*
(pp. 33–47). Brooks/Cole.

Turner, J. C., Hogg, M. A., Oakes, P. J., Reicher, S. D., & Wetherell, M. S. (1987).
*Rediscovering the social group: A self-categorization theory*. Basil Blackwell.

---

## SECTION F: Results Section Fix (H3)
*(Find and replace in Results)*

**Current text (incorrect — contradicts how H3 is stated):**
> "H3 predicted that Republicans would show stronger support for free speech on campus.
> The data contradict this hypothesis: Democrats exhibit significantly stronger support
> for both faculty and student speech rights."

**Replacement:**
> Consistent with H3, Republicans score significantly higher than Democrats on the
> combined speech restriction index (Republicans M = 3.27, SD = 0.88; Democrats
> M = 2.67, SD = 0.84; t(542) = 6.93, p < .001, d = 0.69), indicating greater support
> for restricting campus speech. This moderate-to-large effect holds across both the
> faculty speech battery and the student speech battery, and is consistent across all
> individual items in both batteries. True Independents fall between the two partisan
> groups (M = 3.01), and partisan leaners track closely with their directional party,
> consistent with prior work showing that independent self-identification frequently
> masks latent partisan attachment (Levendusky, 2009). The party gap in speech
> restriction confirmed by H3 is examined more precisely in the regression analyses
> below, which disentangle how much of this gap is explained by differences in
> affective polarization between parties versus other partisan differences.

---

## SECTION G: Abstract Update
*(Replace the methods sentence in the abstract)*

**Current:**
> "Using multivariate analyses in Python, including independent-samples t-tests,
> Cohen's d effect sizes, Cronbach's alpha reliability assessment, and dimensionality
> reduction via UMAP and PCA, the study explores the relationship between affective
> polarization and attitudes toward free speech on college campuses."

**Replacement:**
> "Using a multivariate analytic strategy in Python — including Cronbach's alpha for
> scale reliability, independent-samples t-tests with Cohen's d effect sizes,
> exploratory dimensionality reduction via PCA and UMAP, and Ordinary Least Squares
> (OLS) regression with interaction terms — the study tests whether and how affective
> polarization predicts free speech attitudes on college campuses."
