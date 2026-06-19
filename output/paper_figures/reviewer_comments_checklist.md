# Reviewer comments — `Yun_Discovery_Science_2026_Gadi.pdf`

Reviewer: **gadshaulsky** · 59 annotations · line numbers refer to `paper.tex`.

Legend: ✏️ = inline edit (delete struck text → insert caret text) · 💬 = comment · 🔖 = highlight only (revise flagged text).

---

## Page 1 (Abstract)

- [x] ✅ **L73–74 (abstract):** strike `1` → insert `One` *(spell out the number)*
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
- [x] ✅  **L215** — `fusion parameters`: "What does this mean?"
- [x] ✅  **L217:** edit "retrieval pipeline" — strike `e` → insert `article-retrieval` *(→ "article-retrieval pipeline")*
- [x] ✅ **L220** — `dense`: reworded L218 to "keyword-based BM25 retrieval and **dense semantic retrieval**" (lowercase descriptive phrasing, echoes the intro's "keyword-based"). Capitalized "Dense" elsewhere is only inside the system label **"BM25+Dense"** (a config name, proper-noun-like), so it stays — prose is lowercase, label is capitalized; that distinction is the answer to "should it be capitalized here too?" (no).
- [x] ✅ **L220** — `Our` (re-ranker terminology): clarified at first use → "Stage 2 then reranks this pool: a cross-encoder reorders the candidate articles produced by Stage 1 rather than searching the whole corpus again." Makes the "hidden rank" (Stage 1's ranking) explicit without a heavy definition.
- [x] ✅ **L220 (Text note)** — "Explain BM25, MRR@K, RRF etc.": added a `\paragraph{Retrieval terminology.}` at the start of §4 defining BM25, dense retrieval, RRF, reranking, Recall@K, and MRR@K in one place. Removed the now-duplicate metric definitions at L247 (kept the caveat) and simplified the L218 reranking clause (terminology paragraph carries it). L97 intro gloss kept (also covers query expansion).

## Page 6 (Pipeline / Setup)

- [x] 🚫 **L231 (§ heading)** — `Gene-aware`: **Won't fix (leave term).** "X-aware" = the method leverages gene-level knowledge (synonyms/aliases/products); defensible and idiomatic. Appears 14× (abstract ×2, two headings, contribution, figure caption) and is already operationally defined at first use (L234). Renaming risks drift across paper/figures/artifacts for a borderline point.
- [x] ✅ **L235** — `Chunked`: defined inline at first definitional use (L238) → "each article is **split into shorter passages (*chunks*)** — an abstract chunk and several body-text chunks — that are indexed and retrieved individually." (Standard IR term, but jargon to a biologist; one-clause gloss, no rename.)
- [x] ✅ **L239** — `Because...` (first-occurrence pooling): rewritten to drop the jargon term entirely and state the *why* plainly → "Because an article is split into several chunks, the same article can appear multiple times in a chunk-level ranking… each article takes the rank of its highest-ranked chunk."
- [x] 🚫 **L242 (§5 heading)** — `Experimental setup` / "add a lay text box": **Declined.** §5 is standard methods (metrics, depths, significance, systems); its jargon is now defined upstream in the §4 terminology paragraph. A boxed lay-summary would be heavy and out of place. (Gadi not the gatekeeper.)
- [x] ✅ **L239 (Text note)** — "still didn't understand the terms": resolved by the §4 `Retrieval terminology` paragraph (same complaint as the L220 text note).

## Page 7 (Results)

- [x] ✅ **L267** — `Table 2 grounds`: flipped the roadmap sentence so Table 1 is referenced first → "The aggregate trends in Table~\ref{tab:ranker_comparison} and Figs… are illustrated by three query-level examples in Table~\ref{tab:qual_examples}." No table floats moved (chose "mention Table 1 earlier").

## Page 8 (Results / Table 1)

- [x] 🚫 **L284** — `The external Ragnarok BM25+RankZephyr reference remains below BM25, further...`: **Won't fix.** The current narrative order is logical (rerankers first, then the external reference as context); keeping it.
- [x] ✅ **L288 (Table 1 caption)** — last three columns defined **in the caption text** (no footnote): *Model size* (approx. params for neural rerankers; "--" = non-neural), *MRR@10* (mean reciprocal rank at cutoff 10, higher better), *Δ vs. BM25* (MRR@10 difference from BM25 in pp; brackets = 95% paired-bootstrap CI).
- [x] ✅ **L307** — `Table 2 illustrates a failure mode ...`: reworded to lead with the finding and demote the table to a concrete case → "One query-level failure mode is that a general reranker can promote articles sharing phenotype or location terms… Example 1 in Table~\ref{tab:qual_examples} gives a concrete case." Removes the premature "Table illustrates" forward-reference.

## Page 9 (Table 2 / Fig. 3)

- [x] ✅ **L317 (Table 2 caption)** — `Table 2.`: **Kept the table, reframed it as worked examples of the dataset/evaluation** (more useful to a biologist than "qualitative examples"; declined dissolving into prose). Changes: (a) new caption explaining claim→query, cited article=evaluation target, rank change=mechanism, examples↔3 result sections (kept "illustrative not quantitative" caveat); (b) row labels → **Curator claim / Evidence contrast / Ranking change / Interpretation**; (c) cells made readable — "Cited article"/"Competing article", "rank" added (BM25 rank 2 → …), PMIDs demoted to inline, quotes kept; (d) example titles point only to the **§section** (dropped Table/Fig refs, which the section already cites).
- [x] ✅ **L361 (Fig. 3 caption)** — `Fig. 3.`: caption rewritten. (1) **K defined** as the cutoff rank on the x-axis of (a)–(c); (3) **panel (d) expanded** — change in MRR@10 vs. original-query baseline, two bars per reranker = the two expansion variants, positive = improvement. (2) **y-axis label in (d)** = not added — the quantity is already named in the caption/panel title, so a separate label would be redundant (no figure edit). The "panel design makes it harder" point is a figure-asset matter, not changed.

## Page 10 (Full-text / Fig. 4)

- [x] ✅ **L373:** strike `use` → insert `used` — *done*
- [x] 🚫 **L380 (Fig. 4) — reorder columns:** **Declined.** Current order (detail → core → insufficient) matches how the three labels are introduced/counted everywhere else (dataset stats 816/654/387, §4, the n-counts 707/585/350); reordering only this figure would break that consistency. Text already emphasizes "abstract insufficient" verbally.
- [x] ✅ **L380 (Fig. 4) — recolor:** Done in the plotting script (`notebooks/report_7a.py`, `_plot_retrieval_rerank_overlay`): the per-evidence-level color is redundant with the separate columns, so all lines now use a single **dark grey (#333333)** matching the legend (used dark grey, not pure black). baseline/chunked stay distinguished by line style. **PNG must be regenerated** by running the notebook (figure asset not rebuilt here).

## Page 11 (Discussion)

- [x] ✅ **L386:** `(0.16 →0.39)` → `from 0.16 to 0.39` — *done (consistent with "rises from 0.78 to 0.97" above)*
- [x] ✅ **L386** — `chunked setting.` (+ Text note below): addressed as a **dataset limitation** (per Gadi's steer: leave Results as-is, note dataset imperfection). Expanded the limitations paragraph: "...not exhaustive or definitive relevance judgments... some difficult cases may reflect **claim–citation mismatch rather than retrieval failure**... an inline citation may... occasionally point to a paper that does not clearly support the extracted statement." Covers both the two-subgroups idea and the curator-wrong-reference case.
- [x] ✅ **L397** — `Its` + `not only...but also` (value sentence): both resolved → "**The benchmark's value is a combination of** a new dataset **and** a test case for retrieval behavior…"
- [x] ✅ **L397:** `expressions` → `terms` ("organism-specific terms") — *done*
- [x] ✅ **L397** — `This helps explain...`: → "**These cues help explain** why BM25 remains strong…"
- [x] ✅ **L399** — `This has a practical implication...`: → "**This finding has** a practical implication…"
- [x] ✅ **L399** — `default upgrade.`: clarified → "should not be treated as a default upgrade **to the retrieval pipeline**." (answers "upgrade of what?")
- [x] ✅ **L399** — `For` (in "For niche biological search..."): → "**In** niche biological search…"
- [x] ✅ **L405** — `not only by`: recast to fix the logic (fixed pool can't show "more candidates") → "we fix the candidate pool and vary only the reranker query, **isolating a second route**: beyond retrieving more candidates at the first stage, it also gives the reranker better query-side context." Also drops the "not only…but also" construction.
- [x] ✅ **L405** — `This suggests that...`: → "**These gains suggest** that curated gene annotations can enrich…"
- [x] ✅ **L386 (Text note)** — resolved together with L386 above (dataset-imperfection caveat added to the limitations paragraph). Gadi's offer to examine still-problematic claim–reference pairs is optional future work, not needed now.

## Page 12 (Discussion / Appendix)

- [x] ✅ **L414** — `Since open-access...cannot be assumed.`: rephrased in plain language → "In practice, not every article has open-access, machine-readable full text, so a system cannot assume that complete full text is available to index."
- [x] ✅ **L419:** delete `still` (in "can still improve evidence finding") — *done*
- [x] 🚫 **L437 (§ Appendix)** — `Appendix`: **No action.** Reviewer just noted he skipped the (intentionally technical) implementation appendix; nothing to change.
- [x] ✅ **L517 (Table S2)** *(added manually)* — width fix: reduced `\tabcolsep` 8pt→4pt and wrapped the tabular in `\resizebox{\textwidth}{!}{…}` so it fits the text width and centers. **Eyeball after compiling** — if the scaled font looks too small, we can drop the `\resizebox` and rely on the smaller `\tabcolsep` alone.

---

## Recurring themes (cross-cutting)

1. **Define jargon for biologists:** reranking, query expansion, BM25, MRR@K, RRF, gold, chunked, fusion parameters, first-occurrence pooling. Consider a lay-terms text box for §5.
2. **Sentence openers:** stop starting sentences with *This / Its / For*; rephrase throughout.
3. **Avoid "respectively" and "not only X but also Y"** constructions.
4. **Table/figure ordering:** reference Table 1 before Table 2; describe rows in table order.
5. **Two broken/queried references:** Zenodo DOI `10.5281/zenodo.20308282`; reconsider Europe PMC vs. full PubMed/literature.
6. **Figures:** define K, label axes, recolor Fig. 4 to black, reorder Fig. 4 columns; consider folding Table 2 examples into prose.
7. **Reviewer offer:** willing to examine still-problematic claim–reference pairs (note on p11).
