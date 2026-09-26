1. **ollama has no model pulled**: Default model llama3.2:3b is not downloaded.
   Real-model smoke test is skipped. Adapter code is correct but untested
   against a live model in this environment. POC used Doubao for generation.

2. **cv2 not installed**: 5 existing tests + 6 collection errors pre-exist.
   Not related to Phase-1A. Requires `pip install opencv-python` in venv.

3. **Heuristic physical rename deferred**: Role is documented as
   SHOT_SELECTION_SPECIALIST but class name remains HeuristicDirectorReasoner.

4. **DirectorBrief integration deferred**: DirectorDecision and DirectorBrief
   are separate. Future integration requires Chief decision.

5. **Context parameterizer not built**: NEEDS_CONTEXT stops here. Next phase.

6. **Single model path**: No model router/fallback. Primary path only, as
   specified.
