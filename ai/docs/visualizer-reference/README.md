# Inspector design references

`decision.png` and `network.png` are the selected full-screen design references (1536×1024). They are documentation, not assets rendered as UI.

The brief: a readable dark scientific tool for understanding real Armagetron decisions and the loaded network. Charcoal surfaces, fine dividers, cyan for the cycle and player branch, amber for opponents, lilac for arena attention; sans headings and monospaced measurements. Decision, Network and Training share one toolbar, checkpoint identity and playback controls. No decorative network animation or invented performance metrics.

Tokens: background `#101316`, surface `#151b20`, control `#1b242c`, border `#2d3a44`, text `#dce5ee`, muted `#94a8ba`, cyan `#38d8e8`, lilac `#bd9aef`, amber `#f5ac49`. Flat bordered panels, 3–5 px radii, 12–18 px gaps, 20 px section titles and 25 px app title. Training extends the same system with rows of actual metrics and history plots. Mobile stacks panels and retains all controls.

Runtime data replaces every illustrative number. Intentional functional refinements: preserve the arena’s physical aspect ratio; expose both downsampling and both fusion stages separately; include the positional embedding in the first attention stage’s parameter count; show signed activations and legal-action masks; use a native accessible history slider; page all feature channels; identify original-tensor statistics separately from display scaling. Auxiliary and ownership branches remain explicit and are generated from the loaded model. The viewer also supports the smaller v1/v2 layouts.

V4 extends the existing interface with six controls, a third spatial branch, a shared opponent branch, three cross-branch context stages and attack forecasts. The graph scrolls horizontally to keep all five branches readable; its order follows the actual progressive stem and residual computations. These additions reflect the implemented architecture in [V4.md](../V4.md).
