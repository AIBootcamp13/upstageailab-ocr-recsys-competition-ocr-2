# Transform Optimization Summary

I've completed profiling and implementing transform optimizations. Summary of findings and recommended next steps below.

## Profiling results

| Transform        | Time (ms) | % of Total |
|------------------|-----------:|-----------:|
| Normalize        | 3.150      | 87.84%     |
| PadIfNeeded      | 0.265      | 7.39%      |
| LongestMaxSize   | 0.162      | 4.51%      |
| ToTensorV2       | 0.010      | 0.26%      |

Key finding: **Normalization dominates (87.84%) of transform time.**

## Performance comparison

| Configuration                          | Time                 | Relative |
|----------------------------------------|----------------------|---------:|
| Baseline (no caching)                  | 2m38.868s (158.9s)   | -        |
| Image caching only                     | 2m21.620s (141.6s)   | +10.8%   |
| Image caching + ConditionalNormalize   | 2m28.026s (148.0s)   | +6.8%    |

Image caching (Phase 6B) yields the largest immediate gain (~10.8%).

## Why pre-normalization didn't work

- Complex integration: requires propagating float32 normalized images through the whole pipeline.
- Memory overhead: float32 images consume ~4× RAM (e.g., 404 images × ~1.5MB ≈ 600MB vs ~150MB for uint8).
- Marginal benefit: normalization runs on CPU in parallel with GPU inference, so it isn't fully blocking.
- Implementation issues: encountered PIL handling errors and validation-step bugs when pre-normalizing.

## Recommendation

Keep Phase 6B (image caching) in place and move to a more holistic solution for larger gains:

- Proceed to Phase 6A — WebDataset: expected 2–3× speedup, lower effort/risk.
- Or skip to Phase 7 — DALI: maximum performance potential (5–10×), higher implementation effort.
- Also: document current state and create a handover for the next session.

Which path would you like to take?
