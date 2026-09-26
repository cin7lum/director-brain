1. Model used for generation is the same Doubao instance doing the evaluation
   design. Evaluation uses deterministic field comparison against pre-written
   human references, not model self-scoring. This mitigates but does not
   eliminate circularity. A formal eval should use a separate model or human
   review for final scoring.
2. ollama had no models downloaded; external API not used. The POC establishes
   capability upper bound with a strong model; cheaper model performance is
   not measured.
3. Benchmark size is 32 (meets >=30 requirement). Larger benchmark would
   increase confidence.
4. negative_constraint_recall = 93.8%: SD-28 (conflicting constraints) the
   reference lists "conflicting_constraints" as a negative constraint but the
   model expresses it via status=CONFLICTING_CONSTRAINTS instead. This is a
   representation difference, not a semantic miss.
5. DirectorDecision schema is POC-only. Formal EXTEND into production requires
   Chief decision and integration with existing DirectorBrief.
