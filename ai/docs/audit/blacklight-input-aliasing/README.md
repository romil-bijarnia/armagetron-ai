# Blacklight enemy brake-reserve observation aliasing

The original standalone simulator probe completed successfully. Its TemporaryDirectory source and executable were automatically removed. This folder recreates the exact source and preserves the observed output; the experiment has not been rerun.

Compile and run, if desired:

```sh
clang++ -O2 -std=c++17 -I /Users/romilbijarnia/Documents/armagetronad/ai/sim /tmp/blacklight-aliasing-evidence-qz5k6xey/probe.cpp /Users/romilbijarnia/Documents/armagetronad/ai/sim/tron.cpp /Users/romilbijarnia/Documents/armagetronad/ai/sim/observe.cpp -o /tmp/blacklight-aliasing-evidence-qz5k6xey/probe
/tmp/blacklight-aliasing-evidence-qz5k6xey/probe
```

Observed original stdout:

```text
identical_maps=1 identical_96_scalars=1 identical_legal_mask=1
reserved_scalars_88_95_all_zero=1
after_identical_controls_seconds=0.5 enemy_speed_full_reserve=14.6311 enemy_speed_empty_reserve=19.9838
```

This compares two constructed simulator states with different enemy brake reserves. It proves current-observation aliasing and a different subsequent transition under identical controls. It does not establish identical historical trajectories or different optimal actions. No production weights or running engine were used.
