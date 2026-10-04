# Armagetron AI

Armagetron Advanced with a deep reinforcement learning opponent that learns the game by playing against itself inside the real engine.

This repository is the Armagetron Advanced source (0.4 trunk) plus two additions. The engine gains a neural network bridge (`src/tron/gNeural.cpp`) that lets an external process drive chosen AI cycles, and a headless lockstep mode in which the dedicated server plays rounds without any connected human, as fast as the network can answer. The `ai/` folder holds the Python side: the policy network, a self-play PPO trainer that runs dozens of engines in parallel, evaluation tools, and a play mode where you join a local server from the normal game client and race the trained network.

Because training happens in the game itself, the agent learns the real physics (rubber, wall acceleration, turn delay, brakes) rather than an approximation, and it controls its cycle through exactly the same turn and brake commands a human uses.

## Getting started

Build the dedicated server with the bridge, then train or play. The details, including the full configure line, are in [ai/README.md](ai/README.md).

```bash
cd ai && uv run arma-train
```

```bash
cd ai && uv run arma-play
```

With the play server running, open Armagetron, go to Play Game, Multiplayer, Custom Connect, and connect to `127.0.0.1` on port 4534.

## Documentation

- [ai/README.md](ai/README.md): building, training, playing and measuring.
- [ai/docs/DESIGN.md](ai/docs/DESIGN.md): the system design, covering the engine bridge, observations, network, training, evaluation, measured performance, planned changes and the test plan.
- The original Armagetron Advanced documentation is unchanged: `README`, `README-DEVELOPER`, `INSTALL` and `documentation/`.

## License

Armagetron Advanced is free software under the GNU General Public License, version 2 or later; see `COPYING.txt`. The engine changes here are released under the same license.
