# Reviewer comments — `Yun_Discovery_Science_2026_Gadi.pdf`

Reviewer: **gadshaulsky** · 59 annotations · line numbers refer to `paper.tex`.

Legend: ✏️ = inline edit (delete struck text → insert caret text) · 💬 = comment · 🔖 = highlight only (revise flagged text).

---

## Page 1 (Abstract)

- [x] ✏️ **L73–74 (abstract):** strike `1` → insert `One` *(spell out the number)*
- [x] ✅ **L74 (abstract)** — `metadata helps`: resolved by replacing "metadata" with **"gene annotations / curated annotations"** throughout (count noun → "annotations help"), which also addresses the plural-verb note. All 18 "metadata" mentions swapped paper-wide.

## Page 2 (Intro)

- [ ] 🔖 **L91** — `This raises...`: "Don't start sentences with **'This'** — too ambiguous. Flagged in places but **change throughout the paper**."
- [x] ✅ **L93:** strike `the` → insert `a` — *done ("the target is a specific paper")*
- [ ] 💬 **L97** — `reranking or generic query expansion`: "Define **'reranking'** and **'query expansion'** — not trivial to a biologist."
- [ ] 💬 **L113** — `when reranking improves over strong lexical retrieval,`: "Confusing — rephrase."
- [x] ✅ **L120:** delete `both` (in "improve both recall and final ranking") — *done*
- [ ] ✏️ **L125:** edit "Textpresso ... semantic" — strike `a` → insert `-` *(hyphenation/typo fix)*

## Page 3 (Dataset)

- [ ] 💬 **L144** — `Europe PMC abstract corpus.`: "Using the Europe DB can be misleading. Why not use all available literature?"
- [ ] 💬 **L145** — `DOI 10.5281/zenodo.20308282.`: "Link throws an error in Safari. Please revise."

## Page 4 (Dataset)

- [ ] 💬 **L173** — `gold`: "Don't know what **'gold'** means. Define here or use a more intuitive term."
- [ ] 💬 **L175** — `cited articles were absent from the Europe PMC abstract corpus.`: "How many were absent? Would using PubMed directly have solved it?"

## Page 5 (Pipeline)

- [ ] 🔖 **L191** — `This creates...`: flagged 'This' sentence-opener (revise).
- [x] ✅ **L191:** delete `in principle` — *done ("some queries can be answered from abstracts alone")*
- [ ] 🔖 **L202–203** — `gold`: flagged again (same define-the-term issue).
- [ ] 💬 **L206** — `the full dataset contains 816/654/387 pairs ... respectively; ... 707/585/350.`: "Avoid 'respectively'. Rewrite as 'X pairs fall into category A, Y into category B...' **or put the data in a small table**."
- [ ] 💬 **L215** — `fusion parameters`: "What does this mean?"
- [ ] ✏️ **L217:** edit "retrieval pipeline" — strike `e` → insert `article-retrieval` *(→ "article-retrieval pipeline")*
- [ ] 💬 **L220** — `dense`: "Capitalized in the figure — should it be capitalized here too?"
- [ ] 💬 **L220** — `Our` (Section: two-stage design): "Don't understand the terminology. Why a **re**-ranker and not a ranker? Is there a hidden rank?"
- [ ] 💬 **L220 (Text note)** — "Explain in more detail what **BM25, MRR@K, RRF** etc. are — still didn't understand them after finishing the paper."

## Page 6 (Pipeline / Setup)

- [ ] 💬 **L231 (§ heading)** — `Gene-aware`: "The query is already gene-aware (defined by the gene); awareness here is to synonyms/products. Reconsider the term **'gene-aware'**."
- [ ] 💬 **L235** — `Chunked`: "Don't know the term **'chunked'** — define it."
- [ ] 💬 **L239** — `Because...`: "Didn't understand this paragraph — need to understand **'first-occurrence pooling'**."
- [ ] 💬 **L242 (§5 heading)** — `Experimental setup`: "Didn't understand anything in Section 5. Add a **text box explaining the setup in lay terms**."
- [ ] 💬 **L239 (Text note)** — "Still didn't understand the terms after finishing the paper."

## Page 7 (Results)

- [ ] 💬 **L267** — `Table 2 grounds`: "Tables/figures must be referenced in order — **Table 2 can't be mentioned before Table 1**."

## Page 8 (Results / Table 1)

- [ ] 💬 **L284** — `The external Ragnarok BM25+RankZephyr reference remains below BM25, further...`: "Why describe this out of order vs. the table?"
- [ ] 💬 **L288 (Table 1 caption)** — `Table`: "Don't understand the last three columns — explain them in the table footnotes."
- [ ] 💬 **L307** — `Table 2 illustrates a failure mode ... preserves the task-relevant lexical signal.`: "Seems out of place — Table 2 hasn't been described yet."

## Page 9 (Table 2 / Fig. 3)

- [ ] 💬 **L317 (Table 2 caption)** — `Table 2.`: "Had to work hard to read it. This table format isn't suitable — **incorporate the three examples into the text** with more explicit narratives."
- [ ] 💬 **L361 (Fig. 3 caption)** — `Fig. 3.`: "1. define **K** in panels a–c; 2. label the **y-axis in d**; 3. don't understand panel **d** — text doesn't explain it."

## Page 10 (Full-text / Fig. 4)

- [x] ✅ **L373:** strike `use` → insert `used` — *done*
- [ ] 💬 **L380 (Fig. 4 caption)** — `Fig. 4. Retrieval`: "Reorder columns — put **'abstract insufficient' first** (discussed first; most interesting effect)."
- [ ] 💬 **L380 (Fig. 4 caption)** — `Columns correspond to abstract supports detail (n=707)...`: "Change all colors to **black** — no need for three colors when already split into three columns."

## Page 11 (Discussion)

- [x] ✅ **L386:** `(0.16 →0.39)` → `from 0.16 to 0.39` — *done (consistent with "rises from 0.78 to 0.97" above)*
- [ ] 💬 **L386** — `chunked setting.`: "Elaborate — 'abstract insufficient' may have two subgroups (abstract insufficient but full text sufficient vs. both insufficient). **Curator may have given a wrong reference in some cases.**"
- [ ] 🔖 **L397** — `Its` (value sentence): "Revise — **'Its'** as a sentence-opener is as bad as 'This'."
- [ ] 💬 **L397** — `not only as a new dataset, but also`: "Avoid 'not only X but also Y' — say the value is a combination of the new dataset and a test case for..."
- [x] ✅ **L397:** `expressions` → `terms` ("organism-specific terms") — *done*
- [ ] 🔖 **L397** — `This helps explain...`: "Revise" (This sentence-opener).
- [ ] 🔖 **L399** — `This has a practical implication...`: flagged 'This' opener.
- [ ] 💬 **L399** — `default upgrade.`: "Don't understand — upgrade of what?"
- [ ] 💬 **L399** — `For` (in "For niche biological search..."): "Revise — starting with **'For'** seems odd."
- [ ] 💬 **L405** — `not only by`: "Rephrase."
- [ ] 🔖 **L405** — `This suggests that...`: flagged 'This' opener.
- [ ] 💬 **L386 (Text note)** — "Reconsider: leave the text as is here, but **add a paragraph to the Discussion** noting curators may have made mistakes. Offer: reviewer could examine claim–reference pairs still problematic after full text."

## Page 12 (Discussion / Appendix)

- [ ] 💬 **L414** — `Since open-access and machine-readable full-text coverage is uneven in practice, complete full-text indexing cannot be assumed.`: "Don't understand this sentence."
- [x] ✅ **L419:** delete `still` (in "can still improve evidence finding") — *done*
- [ ] 💬 **L437 (§ Appendix)** — `Appendix`: "Too technical — did not review it."
- [ ] 📝 **L517 (Table S2, `tab:s2_evidence_haspdf_full`)** *(added manually)* — Table S2 is **shifted** (misaligned on the page) and **too wide**; fix placement/width.

---

## Recurring themes (cross-cutting)

1. **Define jargon for biologists:** reranking, query expansion, BM25, MRR@K, RRF, gold, chunked, fusion parameters, first-occurrence pooling. Consider a lay-terms text box for §5.
2. **Sentence openers:** stop starting sentences with *This / Its / For*; rephrase throughout.
3. **Avoid "respectively" and "not only X but also Y"** constructions.
4. **Table/figure ordering:** reference Table 1 before Table 2; describe rows in table order.
5. **Two broken/queried references:** Zenodo DOI `10.5281/zenodo.20308282`; reconsider Europe PMC vs. full PubMed/literature.
6. **Figures:** define K, label axes, recolor Fig. 4 to black, reorder Fig. 4 columns; consider folding Table 2 examples into prose.
7. **Reviewer offer:** willing to examine still-problematic claim–reference pairs (note on p11).
