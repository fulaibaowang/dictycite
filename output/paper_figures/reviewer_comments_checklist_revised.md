# Reviewer comments (round 2) — `Yun_Discovery_Science_2026_revised_Gadi.pdf`

Reviewer: **gadi** · 74 annotations on the revised manuscript · line numbers refer to current `paper.tex`.

Legend: ✏️ = inline edit (delete struck text → insert caret text) · 💬 = comment · 🔖 = highlight only (revise flagged text) · ⭐ = substantive / cross-cutting.

> Note: Gadi stopped reviewing at the Appendix (p12: *"I ran out of time to review the rest. sorry."*), so coverage is pp. 1–12.

---

## Page 1 (Abstract + intro start)

- [x] ✅ **L74 (abstract)** — `curator-written` → `curator-generated` *(applied paper-wide, all 6 occurrences — see ⭐ recurring note)*
- [x] ✅ **L74 (abstract)** — `Reranking and gene-aware query expansion both improve…` → "**We report that** reranking and gene-aware query expansion improve retrieval selectively…" (added opener, dropped `both`)
- [x] ✅ **L74 (abstract)** — `while curated annotations…` → `whereas`
- [x] ✅ **L74 (abstract)** — `these cases remain harder` → `these cases are harder`
- [x] ✅ **L82 (intro)** — `However, much biological` → `However, a large amount of biological`

## Page 2 (Intro)

- [x] ✅ **L91** — `…asked to retrieve the paper cited as supporting evidence in a curated knowledge base?` → `…asked to retrieve a proper supporting reference from a curated knowledge base?`
- [x] ✅ **L93–94** — took Gadi's edit: colon list, dropped "more", and dropped the "rather than assumed to improve retrieval" over-claim → "In this setting, we evaluate two common strategies: semantic reranking, which reorders the retrieved articles using a selective model, and query expansion, which adds related terms to the search query." (the "not assumed, must be evaluated" point is carried by the Results/Discussion.)
- [x] ✅ **L98–99** — `studies of chemotaxis, motility, phago-…` → `chemotaxis, cell motility, phago-…`
- [x] ✅ **L101** — `curator-written notes` → `curator-generated` *(recurring; done globally in ca8aad6 — see L74/101/136/147/148/304)*
- [x] ✅ **L102–104** — `…retrieve the cited article from a domain-specific literature corpus.` → `…retrieve a proper citation to an article from a domain-specific literature corpus.`
- [ ] ⏸️ **L117** — `such cases remain challenging.` — "say 2–3 words about **why** it is challenging." **Deferred** (skipped for now).
- [x] ✅ **L120 + L384** — `supporting papers` → `supporting citations` (both occurrences, for consistency).

## Page 3 (Related work / Dataset)

- [x] ✅ **L123** — deleted `Rather than proposing a new architecture, we` → "…PubMed-scale retrieval. **We use** these standard components…" *(eyeball: this drops the "not a new architecture" scope disclaimer — revert if you want to keep that positioning.)*
- [x] ✅ **L101/136/147/148/302 (and L74)** — `curator-written` → `curator-generated` **everywhere**: all 6 occurrences (L74, 101, 136, 147, 148, 304) converted in ca8aad6; verified zero `curator-written` remain. Gadi's reasoning: *written* describes only part of the curator's work — they read the literature, make logical inferences, and decide on a suitable set of references.
- [x] ✅ **L159** — instead of defining the jargon, removed it: `removing residual markup, decoding HTML entities, and normalizing whitespace` → `removing formatting tags and normalizing whitespace` (less distracting, no IR jargon).

## Page 4 (Dataset)

- [x] ✅ **L167** — appended the magnitude only (author-edited wording): "…absent from the Europe PMC abstract corpus, removing about 2\% of the claims in total." (43/2,063 claims excluded ≈ 2%, computed from notebook 04 funnel 2,063→2,040→2,020.)
- [x] ✅ **L172** — added Gadi's non-biologist explanation (tightened to 2 sentences) after "…product description.": "Maintaining such alternative names is common practice, because the relevant literature does not always use the consensus gene name. For example, the gene \emph{pkaC} (Fig.~\ref{fig:dataset}A) encodes the protein PKA-C, …" — points to Fig. 1A, which shows the pkaC gene page (same running example as L165 / Fig. 1B).

## Page 5 (Evidence labels / counts)

- [x] ✅ **L180** — added a concrete example before "This gap creates an important distinction…": "For example, in the \emph{pkaC} instance shown in Fig.~\ref{fig:dataset}B, the cited abstract describes the study at a broad level, but not the specific expression pattern stated in the curator claim." Used the Fig. 1B running example instead of Gadi's Rosengarten wording; **reference deliberately not added** (Rosengarten still uncited at L165 and absent from ds_paper_revised.bib — left as-is per author).
- [x] ✅ **L192–195 (counts)** — made the parentheticals explicit: `654 (585)` → `654 (585 in the has-PDF subset)` and `387 (350)` → `387 (350 in the has-PDF subset)`.

## Page 6

- [x] ✅ **L180 (`This`)** — referent named: now reads "This gap creates an important distinction…", and the preceding sentence supplies the concrete pkaC example it refers back to.

## Page 7 (Experimental setup)

- [x] ✅ **L231** — `implementation details are given in the Appendix.` → `are provided in the Appendix.`
- [x] ✅ **L235** — `Recall@K for first-stage retrieval` → `Recall@K for the first-stage retrieval`

## Page 8 (Results / Table 1)

- [x] ✅ **L273 (Table 1 caption)** — `MRR@10 is mean reciprocal rank at cutoff 10, with higher values indicating better…` → `is the mean reciprocal rank at cutoff 10, where higher values indicate better…`
- [x] ✅ **L273 (Table 1 caption, `…confidence intervals.`)** — explained Δ: added "positive values indicate improvement over BM25" before the confidence-intervals clause (author-tightened wording).
- [x] ✅ **L260–262** — reworded: dropped redundant "by itself", inserted "alone" → "…but scores below BM25 at MRR@10, indicating that higher recall alone does not improve the top ranks." (author also dropped "automatically").

## Page 9 (Table 2)

- [ ] ✏️ **L310/caption — Table 2 title** — strike `Worked examples` (reconsider the framing).
- [ ] 💬🔖 **L317** — `…The competing article shares location vocabulary…` — "**why is it 'competing'?** do you mean the next article in the ranked list?" (define/relabel).
- [ ] ⭐💬🔖 **L308–335 (Table 2 overall)** — Gadi (3 separate notes): "It is hard to follow this example. I still think this table format is **not suitable**… I don't understand the **arrow** notation and I don't fully understand what the **ranks** mean. Please **reconsider your position on using this table**." On Example 3: "Here your intention is a bit clearer because there are only two elements… but you already lost me as a sympathetic reader." → **reconsider Table 2** (arrows `$\rightarrow$`, rank semantics).
- [ ] ⭐💬🔖 **L308 (`Table 2.`)** — "If you insist on using this table, please **move it after Figure 3**. It is hard to jump back and forth since the main text is not aligned with the order of these display items."

## Page 10 (Fig. 3 + full-text results)

- [x] ✅ **L348/354 (Fig. 3 / QE text, `curator claim`)** — unified to "original query" (matching the legend): bridged once at first mention (L342: "the original query (the curator claim)"), then "original query" in the Fig. 3 caption and prose (L349/353/355). Kept "curator claim" only where it contrasts with the cited abstract.
- [x] ✅ **L348 (Fig. 3 panel `(d)`)** — defined the y-axis in the **caption** instead of re-rendering (author chose not to touch the figure): "(d) The y-axis shows the change in MRR@10 (in MRR@10 units) relative to the original-query baseline…". Note: panel (d) is in raw MRR@10 units, not percentage points (deltas computed as v−body on the 0–1 scale in query_expansion_sweeping.py), so "percentage points" was deliberately avoided.
- [x] ✅ **L360** — `…limiting for claims whose cited abstracts…` → `claims where the cited abstracts…` *(this is the same "whose" Gadi's two overlapping marks both point at)*
- [x] ✅ **L~ (full-text)** — L359 subsection title: "…evidence is missing from abstracts" → "…missing from **the** abstracts".
- [x] ✅ **L372** — `…already close to ceiling…` → "close to **the** ceiling".
- [x] ✅ **L371/373/377 (`buckets`)** — replaced all 5 occurrences with **`groups`** (Gadi: "too casual; use 'groups' or 'cases'").

## Page 11 (Discussion)

- [x] ✅ **L377** — `provide their largest benefit` → "provide **the** largest benefit" (dropped "their").
- [x] ✅ **L~ (discussion)** — inserted `a` at L385: "a realistic form of **a** niche biological search" (author edit).
- [x] ✅ **L386 (`a costly reranker`)** — resolved $-vs-compute ambiguity by dropping the word "costly": "…because **a reranking stage that adds substantial computation** is useful only if it improves over a strong lexical baseline on the target retrieval task." (author wording — cost = computation.)
- [x] ✅ **L389 (`low-cost`)** — `low-cost` → **`low-overhead`**: "Gene-aware query expansion provides a complementary, low-overhead way to improve the pipeline." (author wording.)

## Page 12 (Discussion / Appendix)

- [x] ✅ **L403 (`…available to index.` / full-text coverage)** — addressed partial-coverage concern: noted it could in principle bias retrieval (full-text articles contribute many chunks vs one for abstract-only), but the full-dataset columns of Table~\ref{tab:s2_evidence_haspdf_full} show the effect is not dominant — mixed abstract/full-text retrieval still improves evidence finding under partial coverage. (author wording.)
- [ ] 🚫 **L412 (`Appendix`)** — "I ran out of time to review the rest. sorry." → **no action** (reviewer stopped here).

---

## Recurring themes (cross-cutting)

1. ✅ **`curator-written` → `curator-generated` everywhere** — DONE (all 6 occurrences). Gadi's explicit, repeated request — *written* understates the curator's inference and reference-selection work.
2. ⭐ **Table 2 still not working** — arrow notation and rank semantics are unclear; Gadi asks us to reconsider the table entirely, and at minimum **move it after Fig. 3** and align the main-text order with the display items.
3. **Define "cost"** (compute time vs. dollars) for "costly reranker" / "low-cost expansion" (L386, L389).
4. **Jargon / clarity for biologists:** define "residual markup" (L159), explain the Δ in Table 1 (L273), replace casual "buckets" with "groups"/"cases", label the Fig. 3(d) y-axis.
5. **Add missing numbers / examples:** count of excluded claims (L167); pkaC/cAMP non-biologist gene-name explanation (L172); Rosengarten-abstract worked example + **its reference** (L180).
6. **De-novo full-text guidance** — recommend defaulting to full text on a new problem, since abstract gene-name absence is ambiguous (L403).
7. **Grammar nits (low-effort):** whereas / are / where / provided / "the" insertions, "much" → "a large amount of", colon-list rewrite of the two-strategies sentence.

---

## Open figure-asset tasks (need re-render, not a .tex edit)

- Fig. 3 panel **(d): add a y-axis label** in the image.
