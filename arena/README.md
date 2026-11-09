# Local versus arena

Open [../runs/arena.html](../runs/arena.html) in a browser and select **Start match**. The HTML file contains the complete game and the trained 10,000-placement DQN model. It does not fetch scripts, fonts, model weights or other resources, and works without a server or internet connection.

From PowerShell in the project directory:

```powershell
# Regenerate the game with the current checkpoint.
python -m tetrio_rl arena --checkpoint runs/initial/latest.pt --output runs/arena.html
Start-Process runs/arena.html
```

Choose **You vs DQN**, **Watch DQN vs heuristic**, or **Watch DQN vs DQN**. Set the bot's placements per second with the speed slider; the speed can be changed during a round. Gravity controls the human board. Bots make timed placement decisions at the chosen speed, matching the placement-based policy used during training.

| Control | Action |
| --- | --- |
| Left / Right arrows | Move; hold to repeat |
| Up arrow / X | Rotate clockwise |
| Z | Rotate counterclockwise |
| A | Rotate 180° using two clockwise rotations |
| C / Shift | Hold once per placed piece |
| Down arrow | Soft drop |
| Space | Hard drop |
| P / Escape | Pause or resume |
| R | Start a new round |
| Enter | Start from the ready or game-over screen |

Touch buttons are shown on small screens and touch devices. Losing browser focus pauses an active game and releases held controls. Wins are kept in memory between rounds and reset when the match mode changes; refreshing the page resets the session.

## Rules

Both boards use the same seeded seven-bag sequence, a 10×20 visible field, four hidden spawn rows, hold, SRS 90° kicks and ghost pieces. The human board has gravity, a 500ms lock delay and a 15-reset limit. The 180° action is two legal clockwise turns, committed together; it is not a TETR.IO SRS+ 180° kick table.

Attacks are local rules inspired by modern versus games:

- Singles send zero base garbage; doubles 1, triples 2, and quads 4.
- T-spins use a three-corner test and send 2/4/6 for single/double/triple. Mini distinctions and all-spin scoring are not implemented.
- Consecutive difficult clears receive a one-line back-to-back bonus. Combos add one line per two additional clears, capped at four.
- A perfect clear adds 10 attack lines.
- Attacks first cancel the sender's pending garbage. Remaining attacks travel to the opponent for 700ms.
- Ready garbage rises only after a placement that clears no lines, up to eight rows per placement. A packet shares one random gap column.
- Overflow or a blocked spawn ends the round.

This is an independent arena built around our existing simulator and model. It is not Tetr Online, and does not claim exact TETR.IO attack or timing parity. The initial model was trained for local stacking; versus performance will benefit from further training against garbage and opponents.

## Source and validation

`dist/engine.mjs` owns game rules and candidate generation. `dist/app.mjs` owns controls and rendering. The model scorer comes from `../portable/model.mjs`; the `arena` command copies that scorer and the selected exported model into `dist/` and packages a single HTML file. Generated weights and runtime files are ignored by Git. No Node package installation is required.

```powershell
python -m tetrio_rl arena
python -m unittest discover -s tests -v
node --test arena/test/*.test.mjs portable/test/*.test.mjs live/test/*.test.mjs
```

Validation covers cross-language placement parity, feature and neural-score parity, holds, seven-bag generation, rotations, attacks, cancellation, garbage overflow, pause and game-over handling. A real trained-model versus match is simulated through the same engine. The generated HTML is executed with a DOM/canvas test double to check keyboard controls, rematches, watch mode and the absence of external fetches. This verifies application logic, not browser layout or pixels. Interactive browser preview was unavailable because the managed environment rejected the browser launch.

The separately prepared Tetr Online integration remains under `../integrations/tetr-online/`; its actual upstream controller is not connected because source downloads were unavailable.
