# Phase 5b: a second fresh batch (frozen before any of its data exists)

Why. The first fresh batch (600 tasks, PHASE5_PLAN.md) confirmed P1 (separability) and P3 (editor ranking) against the
most-trained colluder but not P2 (safety with five honeypots: Granite small minus ordinary, +2.8 points, 95% interval
[-0.6, +7.5]). It had 66 own attacks. This batch replicates the test on tasks that appear in neither earlier set, with more
own attacks, and nothing changes: no editor, cutoff, filter, threshold, prompt, adapter or analysis code is touched
(the two retrained most-trained adapters stay as they were for batch 1).

Tasks: 1,000 from the APPS test file, seed 6, excluding the 600 of batch 1, the 800 original tasks and any problem whose first
200 characters match a train problem. Pipeline exactly as PHASE5_PLAN.md (`modal_phase5.py`, `paper/fresh_analysis.py`).

Hypotheses (most-trained colluder, mean over its three seeds; flagged honeypots; task-level bootstrap), tested
separately on batch 2 and then pooled over batches 1 and 2 (1,600 tasks):
- P1: separability(Granite small) < separability(ordinary); 95% interval of the difference excludes 0.
- P2: safety with five honeypots, Granite small minus ordinary, is positive with a 95% interval excluding 0.
- P3: Granite has the lowest separability of the three editors' small-edit honeypots.
The pooled P2 interval is the number the paper reports for the safety gain; the batch-2 result is the clean replication.
P2 failing on both would mean the paper states that honeypots add no detectable safety against a trained colluder.
Everything is reported whatever it shows.
