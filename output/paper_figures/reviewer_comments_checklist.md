# Reviewer comments — `Yun_Discovery_Science_2026_Gadi.pdf`

Reviewer: **gadshaulsky** · 59 annotations · line numbers refer to `paper.tex`.

Legend: ✏️ = inline edit (delete struck text → insert caret text) · 💬 = comment · 🔖 = highlight only (revise flagged text).

---

## Page 1 (Abstract)

- [x] ✏️ **L73–74 (abstract):** strike `1` → insert `One` *(spell out the number)*
- [x] ✅ **L74 (abstract)** — `metadata helps`: resolved by replacing "metadata" with **"gene annotations / curated annotations"** throughout (count noun → "annotations help"), which also addresses the plural-verb note. All 18 "metadata" mentions swapped paper-wide.

## Page 2 (Intro)

- [x] ✅ **L91** — `This raises...`: → "**Such differences raise** a practical question…" (named the referent). *Being selective: only ambiguous bare "This/Its + verb" openers were changed, not every "This".*
- [x] ✅ **L93:** strike `the` → insert `a` — *done ("the target is a specific paper")*
- [x] ✅ **L97** — `reranking or generic query expansion`: glossed inline at first body use → "two common strategies---semantic reranking, which reorders the retrieved articles using a more selective model, and query expansion, which adds related terms to the search query---should be evaluated…". Full definitions remain in Related Work (L128) and Methods/Figs 1–2.
- [x] ✅ **L113** — `when reranking improves over strong lexical retrieval,`: rephrased → "**whether reranking adds value beyond strong keyword-based retrieval**" (clearer, and parallels the Results-section phrasing)."
- [x] ✅ **L120:** delete `both` (in "improve both recall and final ranking") — *done*
- [x] ✏️ **L125:** edit "Textpresso ... semantic" — strike `a` → insert `-` *(hyphenation/typo fix)*

## Page 3 (Dataset)

- [x] 🚫 **L144** — `Europe PMC abstract corpus.`: **Won't fix.** Rests on a misconception — Europe PMC indexes all PubMed/MEDLINE abstracts (EPMC ⊇ PubMed), so the source choice doesn't restrict coverage; absences come only from the topical corpus snapshot. (Gadi is a friendly reviewer, not the gatekeeper.)
- [x] ⏸️ **L145 (Zenodo DOI)** — **Deferred.** Link will resolve once the Zenodo record is published; leaving as-is for now.

## Page 4 (Dataset)

- [x] ✅ **L173** — `gold`: replaced jargon with intuitive terms in prose (no "ground-truth" — too strong, since citations aren't exhaustive). L171 first use now: "...used as the **reference article for evaluation**. We treat such curator-cited articles as **known positive references**, not as an exhaustive set...". Targeted swaps elsewhere: gold cited article→known cited article, gold PubMed articles→cited, gold PMID→cited PMID, non-gold→non-cited competitor, gold source→cited source, gold abstract→cited abstract. "goldset" now appears **once**, at L452, as a findability anchor: "the public dataset (released as the dictyBase citation **goldset**) are available at…". This matches the public-facing name (GitHub README, Zenodo, `dictycite_goldset.jsonl`) while keeping the rest of the prose jargon-free.
- [x] 🚫 **L175** — `cited articles were absent from the Europe PMC abstract corpus.`: **Won't fix** (same misconception as L144 — absences are from the topical corpus snapshot, not EPMC vs PubMed coverage).

## Page 5 (Pipeline)

- [x] ✅ **L191** — `This creates...`: → "**This gap creates** an important distinction…"
- [x] ✅ **L191:** delete `in principle` — *done ("some queries can be answered from abstracts alone")*
- [x] ✅ **L202–203** — `gold`: resolved with L173 (now "cited PubMed articles" / "cited PMID").
- [x] ✅ **L206** — `816/654/387 ... respectively; ... 707/585/350`: rephrased to attach each count to its label inline → "*abstract supports detail* accounts for 816 pairs (707 in the has-PDF subset), *abstract supports core* for 654 (585), and *abstract insufficient* for 387 (350)." Removes the double slash-triple and "respectively"; no table needed (only 6 numbers).
- [x] 💬 **L215** — `fusion parameters`: "What does this mean?"
- [x] ✏️ **L217:** edit "retrieval pipeline" — strike `e` → insert `article-retrieval` *(→ "article-retrieval pipeline")*
- [ ] 💬 **L220** — `dense`: "Capitalized in the figure — should it be capitalized here too?"
- [x] ✅ **L220** — `Our` (re-ranker terminology): clarified at first use → "Stage 2 then reranks this pool: a cross-encoder reorders the candidate articles produced by Stage 1 rather than searching the whole corpus again." Makes the "hidden rank" (Stage 1's ranking) explicit without a heavy definition.
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
- [x] ✅ **L397** — `Its` + `not only...but also` (value sentence): both resolved → "**The benchmark's value is a combination of** a new dataset **and** a test case for retrieval behavior…"
- [x] ✅ **L397:** `expressions` → `terms` ("organism-specific terms") — *done*
- [x] ✅ **L397** — `This helps explain...`: → "**These cues help explain** why BM25 remains strong…"
- [x] ✅ **L399** — `This has a practical implication...`: → "**This finding has** a practical implication…"
- [ ] 💬 **L399** — `default upgrade.`: "Don't understand — upgrade of what?" *(still open — define/clarify)*
- [x] ✅ **L399** — `For` (in "For niche biological search..."): → "**In** niche biological search…"
- [ ] 💬 **L405** — `not only by`: "Rephrase." *(still open — also a content issue: pool is fixed, so crediting "more candidates" is contradictory; needs recast)*
- [x] ✅ **L405** — `This suggests that...`: → "**These gains suggest** that curated gene annotations can enrich…"
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
