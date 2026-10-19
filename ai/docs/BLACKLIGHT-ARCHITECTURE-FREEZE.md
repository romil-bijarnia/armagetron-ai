# Blacklight architecture freeze

Selected on 19 October 2026. Keep this brain and continue improving its training.
No further architecture changes are planned.

The selected design is V4's existing close, nearby and arena encoders, exact
numeric sensors and masked opponent attention, followed by a persistent 256-unit
GRU. Its 775 inputs combine the 768 decision features with the previous executed
six-action control and elapsed time. A residual projection feeds the policy,
value and combat forecasts. The policy retains all six steering/braking actions.

The network has **8,155,373 parameters**. The canonical checkpoint is
`runs/blacklight/latest.pt`, at update **11,280**, with network version 4,
observation schema 2 and memory schema 1. It receives 108 scalar measurements and
current/previous maps. Wire protocol remains version 4.

Migration preserved the existing learned tensors and decisions. The memory
residual projection and the newly added opponent-resource connections remain
zero in this checkpoint. Those additions are installed but untrained; they will
learn during subsequent training.

The recurrent pipeline has passed mechanical, numerical-parity and bounded
learning/resume checks. This selects a working design; it does not establish
that Blacklight is already a strong fighter or that this is the best possible
network. Combat qualification remains open.

## Training work before the long run

- Exercise varied strong native bot profiles, including close-combat situations.
  An explicit named profile pool is implemented in recording and imitation. This does not
  add stock bots to simulator RL, which still uses self-play, Hunter and exploiter
  opponents.
  It is opt-in while fighting strength is being qualified; the initial aggressive
  presets that ran straight into walls were screened and revised in native play.
- Cover braking, compound controls and recovery from states reached after the
  student's mistakes. Track their actual examples and outcomes.
- Complete a short learning pilot, then evaluate frozen weights on held-out
  fights. Compare wins, surviving kills, deaths and stalled rounds, and retain
  regression checks as training continues.
- Verify native applied-action cadence and latency, memory resets and frame
  times in the intended game mode. Check value and combat forecasts against
  outcomes before relying on them for search.

Opponents, scenarios, demonstrations, rewards, losses, sampling and runtime
efficiency can improve without rebuilding the network. A weaker training run is
a reason to inspect those inputs and learning choices, not to replace the brain.

Reopen the architecture only when a reproducible failure demonstrates that the
observation representation, available actions or memory capacity cannot express
the behavior needed. Record that evidence before proposing a structural change.
