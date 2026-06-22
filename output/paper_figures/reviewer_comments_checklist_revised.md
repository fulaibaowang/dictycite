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
- [ ] ✏️ **L101** — `curator-written notes` → `curator-generated` *(recurring)*
- [x] ✅ **L102–104** — `…retrieve the cited article from a domain-specific literature corpus.` → `…retrieve a proper citation to an article from a domain-specific literature corpus.`
- [ ] ⏸️ **L117** — `such cases remain challenging.` — "say 2–3 words about **why** it is challenging." **Deferred** (skipped for now).
- [x] ✅ **L120 + L384** — `supporting papers` → `supporting citations` (both occurrences, for consistency).

## Page 3 (Related work / Dataset)

- [x] ✅ **L123** — deleted `Rather than proposing a new architecture, we` → "…PubMed-scale retrieval. **We use** these standard components…" *(eyeball: this drops the "not a new architecture" scope disclaimer — revert if you want to keep that positioning.)*
- [ ] ⭐💬🔖 **L101/136/147/148/302 (and L74)** — `curator-written`: **"Please change this everywhere to 'generated'."** Gadi's reasoning: *written* describes only part of the curator's work — they read the literature, make logical inferences, and decide on a suitable set of references. He did not change it throughout; we must (6 occurrences: L74, 101, 136, 147, 148, 302).
- [ ] 💬🔖 **L159** — `removing residual markup` — **define** "residual markup".

## Page 4 (Dataset)

- [ ] 💬🔖 **L167** — `…cited articles were absent from the Europe PMC abstract corpus.` — **say how many claims were excluded.** Likely small, but readers/reviewers need the number; a large number could indicate systematic error in the procedure.
- [ ] ⭐💬🔖 **L172 (product description)** — add a non-biologist explanation of why query expansion uses gene synonyms/products. Gadi's suggested text: *"This is a common practice in such databases because the relevant literature does not always include the consensus gene name. For example, the pkaC gene encodes a protein named PKA-C that performs the activity of a protein kinase regulated by the small second messenger cyclic adenosine monophosphate; the literature refers to the gene, protein, and activity using the full name or the common initials cAMP."*

## Page 5 (Evidence labels / counts)

- [ ] ⭐💬✏️ **L180** — after `This gap creates an important distinction…` add a concrete example. Gadi's text: *"For example, the abstract of the article by Rosengarten et al. 2015 does not mention the gene name pkaC or any of its common synonyms [ref# — please add to the reference list]."* (ties to the pkaC example already at L165; **add the reference**.)
- [ ] ✏️ **L192–195 (counts)** — make the parentheticals explicit: `654 (585)` → `654 (585 **in the has-PDF subset**)` and `387 (350)` → `387 (350 **in the has-PDF subset**)`

## Page 6

- [ ] 🔖 **L180 (`This`)** — highlight only, no comment. Likely another bare `This` sentence-opener flag (recurring) — name the referent.

## Page 7 (Experimental setup)

- [x] ✅ **L231** — `implementation details are given in the Appendix.` → `are provided in the Appendix.`
- [x] ✅ **L235** — `Recall@K for first-stage retrieval` → `Recall@K for the first-stage retrieval`

## Page 8 (Results / Table 1)

- [x] ✅ **L273 (Table 1 caption)** — `MRR@10 is mean reciprocal rank at cutoff 10, with higher values indicating better…` → `is the mean reciprocal rank at cutoff 10, where higher values indicate better…`
- [ ] 💬🔖 **L273 (Table 1 caption, `…confidence intervals.`)** — **explain what the Δ means** (i.e., a larger difference means that…).
- [ ] ✏️ **L260–262** — `used as a recall-oriented candidate generator, but by itself scores below BM25 at MRR@10, indicating that higher recall does not automatically improve…` → reword (strike `by itself scores below` / `generator, recall does`) → insert `alone` → "…higher recall **alone** does not automatically improve the top ranks."

## Page 9 (Table 2)

- [ ] ✏️ **L310/caption — Table 2 title** — strike `Worked examples` (reconsider the framing).
- [ ] 💬🔖 **L317** — `…The competing article shares location vocabulary…` — "**why is it 'competing'?** do you mean the next article in the ranked list?" (define/relabel).
- [ ] ⭐💬🔖 **L308–335 (Table 2 overall)** — Gadi (3 separate notes): "It is hard to follow this example. I still think this table format is **not suitable**… I don't understand the **arrow** notation and I don't fully understand what the **ranks** mean. Please **reconsider your position on using this table**." On Example 3: "Here your intention is a bit clearer because there are only two elements… but you already lost me as a sympathetic reader." → **reconsider Table 2** (arrows `$\rightarrow$`, rank semantics).
- [ ] ⭐💬🔖 **L308 (`Table 2.`)** — "If you insist on using this table, please **move it after Figure 3**. It is hard to jump back and forth since the main text is not aligned with the order of these display items."

## Page 10 (Fig. 3 + full-text results)

- [ ] 💬🔖 **L348/354 (Fig. 3 / QE text, `curator claim`)** — "the legend defines it as **'original query'**, so this is confusing." → unify wording between caption legend and prose ("original query" vs "original curator claim").
- [ ] 💬🔖 **L348 (Fig. 3 panel `(d)`)** — "**Define the y-axis in the image**" (figure asset — re-render fig5/fig3 with a y-axis label on panel d).
- [x] ✅ **L360** — `…limiting for claims whose cited abstracts…` → `claims where the cited abstracts…` *(this is the same "whose" Gadi's two overlapping marks both point at)*
- [ ] ✏️ **L~ (full-text)** — `Full-text chunks … abstracts` → insert `the` ("**the** abstracts")
- [ ] ✏️ **L372** — `…already close to ceiling…` → minor caret `the` near "to ceiling"
- [x] ✅ **L371/373/377 (`buckets`)** — replaced all 5 occurrences with **`groups`** (Gadi: "too casual; use 'groups' or 'cases'").

## Page 11 (Discussion)

- [ ] ✏️ **L377** — `provide their largest benefit` → strike `their largest` *(reword)*
- [ ] ✏️ **L~ (discussion)** — `benchmark of niche…` → insert `a` ("**a** benchmark of niche…")
- [ ] ⭐💬🔖 **L386 (`a costly reranker`)** — "could you explain this term? why is one reranker more costly than another? Is it the actual cost in **$** or measured in **compute time**?" → define cost.
- [ ] 💬🔖 **L389 (`low-cost`)** — "same as above re. cost" → define alongside L386.

## Page 12 (Discussion / Appendix)

- [ ] ⭐💬🔖 **L403 (`…available to index.` / full-text coverage)** — de-novo problem: when you approach a **new** problem you can't know whether a gene name is absent because the paper isn't about that gene, uses a synonym, or has a general abstract. Gadi's recommendation: **"you should always use the full text when you approach a new problem. You should address this issue here."**
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
