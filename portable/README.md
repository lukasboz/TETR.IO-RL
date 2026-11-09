# Portable placement model

`model.json` contains the actual learned network exported from `runs/initial/latest.pt`. `model.mjs` runs its 49-feature encoding and inference in a browser or Node.js, without PyTorch or npm dependencies. `model.rs` is the same exported network for Rust, with no external crate dependencies; `features.rs` provides its board encoder.

```powershell
python -m tetrio_rl export --checkpoint runs/initial/latest.pt --output portable/model.json
python -m tetrio_rl export --checkpoint runs/initial/latest.pt --format rust --output portable/model.rs
python -m unittest discover -s tests -p test_portable.py -v
node --test portable/test/*.test.mjs
```

JavaScript integration:

```javascript
import {PlacementModel} from './model.mjs';
const payload = await fetch('./model.json').then(response => response.json());
const policy = new PlacementModel(payload);
const {index, scores} = policy.rankCandidates(candidates);
// Execute the engine-owned input sequence associated with candidates[index].
```

Each candidate is `{board, lines, held, next, landing_height}`. `board` is a numeric, **top-first** 10-column grid with 20–80 rows; nonzero cells are occupied. The encoder uses the bottom 24 rows, padding above shorter boards. `lines` is the clear count for that placement, `held` is null or an uppercase piece symbol, `next` starts with the piece that becomes active after this placement, and `landing_height` is the mean locked-cell height above the floor **before** the clear. The optional fields default to zero/empty values, but accurate values are needed to match the training environment.

The external game must enumerate legal moves using its own engine, produce the corresponding board after locking/clearing, and retain an executable input sequence for every candidate. Avoid applying simulated moves to the live board. This library scores placements; it does not implement an external game's controller, turn timing, versus attacks or garbage resolution.

The export includes the checkpoint SHA-256, training-step count and feature schema. Tests compare JavaScript encoding and inference with Python/PyTorch on actual simulator states. Floating-point scores can vary slightly between runtimes; parity checks use numeric tolerances.
