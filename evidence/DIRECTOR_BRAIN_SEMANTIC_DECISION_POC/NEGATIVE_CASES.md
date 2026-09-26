# Negative Cases Results

| ID | Input | Expected | Result |
|----|-------|----------|--------|
| SD-04 | 声音别抢在画面前面 | No audio_precedes_picture | PASS — must_avoid=[audio_lead] |
| SD-08 | 不要让下一句声音提前 | No audio_precedes_picture | PASS — must_avoid=[audio_lead] |
| SD-16 | 不要改原来的镜头顺序 | No reorder_story_beat | PASS — must_preserve=[shot_order] |
| SD-30 | 不要延长这个反应 | No extend_visible_duration | PASS — must_avoid=[duration_extension] |

All 4 negative cases correctly avoid producing the positive action.
