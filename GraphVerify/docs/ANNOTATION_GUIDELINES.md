# Claim-Level Annotation Guidelines

These guidelines are for human annotators labeling atomic claims against
retrieved evidence, producing the CSV files
`dataset/claim_annotation.py`/`experiments/compute_annotation_agreement.py`
consume. Two annotators label every item independently; a third adjudicates
disagreements (see `dataset/claim_annotation.py:adjudicate`).

## What you are labeling

You will see:
- A **claim**: one atomic, self-contained factual statement extracted from
  a generated answer.
- A set of **retrieved passages**: the evidence available to the system
  that generated the answer.

You are not judging whether the claim is true in the world. You are
judging whether the **retrieved passages** settle the claim, one way or
the other. A claim can be objectively true and still be labeled
Unsupported if the retrieved passages simply don't contain the relevant
information — that is a real, useful signal (it tells us the retriever
failed, not that the claim is false).

## The three labels

### Supported
The retrieved evidence entails the claim without requiring external
knowledge beyond ordinary coreference/entity resolution (e.g. resolving
"it" to the entity discussed two sentences earlier is fine; inferring an
unstated fact from general world knowledge is not).

### Unsupported
The retrieved evidence does not entail the claim and contains no direct
conflict with it. This covers three situations:
- the evidence is simply missing (no passage discusses the claim's topic
  at all),
- the evidence is present but too broad or indirect to entail the specific
  claim (e.g. the passage discusses the right entity but not the specific
  attribute the claim asserts),
- the claim itself is not verifiable from any retrievable evidence (e.g. a
  subjective claim, a prediction about the future).

### Contradictory
The retrieved evidence actively supports an incompatible value: a
different entity/value for a functional relation, a conflicting temporal
order or date, a numeric value outside a reasonable tolerance, a mutually
exclusive class label, or a stated relation that conflicts with the
claim's relation. Contradictory requires the evidence to say something
that cannot both be true alongside the claim — absence of support is
Unsupported, not Contradictory.

## What to mark alongside the label

- **Evidence span**: for Supported and Contradictory, copy the exact
  sentence(s) or passage id(s) that justify your label.
- **Path correctness** (only if you are also given the system's returned
  evidence path): mark `True`/`False` for whether that specific path is
  the correct justification for the label, independent of whether the
  verdict itself is correct. A system can get the right verdict via the
  wrong path (e.g. right conclusion, coincidentally connected through an
  unrelated intermediate entity) — record that as label-correct,
  path-incorrect.

## 10 positive (unambiguous) examples

1. Claim: *"The bridge was completed in 1937."* Evidence: *"Construction of
   the bridge finished in 1937."* → **Supported**.
2. Claim: *"The company was founded by Maria Chen."* Evidence: *"Maria
   Chen founded the company in 2004."* → **Supported**.
3. Claim: *"The novel was written by Jane Austen."* Evidence: no passage
   mentions the novel's author. → **Unsupported**.
4. Claim: *"The city has a population of 2 million."* Evidence: *"The city
   has a population of 1.2 million."* → **Contradictory** (numeric, well
   beyond tolerance).
5. Claim: *"The treaty was signed before the war began."* Evidence: *"The
   war began in 1914; the treaty was signed in 1916."* → **Contradictory**
   (temporal order reversed).
6. Claim: *"The award was given to the film in 2019."* Evidence: *"The
   film received the award in 2020."* → **Contradictory** (date conflict).
7. Claim: *"The study found no significant effect."* Evidence: *"The study
   found a statistically significant effect (p<0.01)."* → **Contradictory**
   (mutually exclusive finding).
8. Claim: *"The museum is located in Berlin."* Evidence: passages discuss
   the museum's collection and history but never state its city.
   → **Unsupported**.
9. Claim: *"The album sold over one million copies."* Evidence: *"The
   album sold 1.3 million copies in its first year."* → **Supported**
   (claim's threshold is satisfied by the evidence's value).
10. Claim: *"The senator voted against the bill."* Evidence: *"The senator
    voted in favor of the bill."* → **Contradictory** (mutually exclusive
    vote outcome).

## 10 edge-case examples

1. Claim: *"The bridge, designed in the 1920s, was completed in 1937."*
   Evidence only states the completion year, not the design decade.
   → Label **Supported** for the completion sub-fact only if the claim was
   already decomposed to just that sub-fact; if the claim as given bundles
   both facts and only one is verifiable, treat the whole claim as
   **Unsupported** and flag it for re-decomposition (note in the `notes`
   column) rather than guessing.
2. Claim: *"The company is one of the largest in its industry."* This is a
   vague, threshold-free comparative claim. Unless the evidence explicitly
   ranks it, label **Unsupported** — do not infer from adjacent size
   figures.
3. Claim: *"He was born in the city now known as Ho Chi Minh City."*
   Evidence: *"He was born in Saigon."* Since Saigon was renamed Ho Chi
   Minh City (ordinary entity resolution, not inference), label
   **Supported**.
4. Claim: *"The company's revenue grew by about 10%."* Evidence:
   *"Revenue grew from $100M to $109M."* That is 9%, within normal
   rounding of "about 10%" — label **Supported**, not a numeric conflict;
   use judgment on "about"/approximate language rather than the strict
   numeric-tolerance rule, and note your reasoning.
5. Claim: *"The actor won the award for Best Actor."* Evidence: *"The
   actor won the award for Best Supporting Actor."* Same award family,
   different specific category — label **Contradictory** (the specific
   value is wrong), not Unsupported.
6. Claim: *"The paper was retracted."* Evidence is from before any
   retraction could have occurred (older publication date) and is simply
   silent on retraction status. → **Unsupported**, not Contradictory —
   silence is not a denial.
7. Claim: *"Two of the three co-authors are affiliated with MIT."*
   Evidence lists affiliations for only two of the three authors, both at
   MIT. → **Unsupported** for the whole claim, since the third author's
   affiliation (needed to confirm "two of three", not "at least two") is
   unverifiable from what's given — unless the claim was decomposed so
   this ambiguity doesn't arise.
8. Claim: *"The device operates at a temperature of approximately 300K."*
   Evidence gives 27°C. Convert before judging (27°C ≈ 300K) —
   **Supported**. Unit conversion is ordinary resolution, not inference.
9. Claim: *"The show's third season was its best-reviewed."* Evidence
   gives review scores for seasons 1–3 and season 3 is indeed highest.
   → **Supported** — a comparative claim IS verifiable when the evidence
   contains all the compared values, unlike edge case #2 where no
   comparison set was given.
10. Claim: *"The bill passed by a narrow margin."* Evidence: *"The bill
    passed 51–49."* "Narrow" is a judgment call, but a 2-vote margin is
    reasonably called narrow — label **Supported**; reserve Unsupported
    for genuinely ungroundable subjective language (e.g. "the performance
    was widely praised" with no evidence about reception at all).

## Inter-annotator agreement

Report Cohen's kappa (two annotators) or Krippendorff's alpha (two or
more) for verdict labels and path-correctness labels **separately** — see
`experiments/compute_annotation_agreement.py`. A kappa/alpha below ~0.6
typically indicates the guideline needs sharpening before labeling more
data, not that the annotators need to "try harder."
