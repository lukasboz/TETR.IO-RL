# Tetr Online integration status

Target: [xiyan128/tetr_online](https://github.com/xiyan128/tetr_online), a Rust/Bevy game with local versus and an interchangeable controller architecture.

The trained policy has been exported into [../../portable](../../portable) for both JavaScript and Rust inference. **The actual Tetr Online controller is not connected yet.** This environment could not clone the public repository, and the browsing tool could retrieve its README but not its implementation files. No game API or controller trait signature has been assumed from the README.

To supply the source on a machine with GitHub access, run from this project directory:

```powershell
git clone https://github.com/xiyan128/tetr_online.git integrations/tetr-online/upstream
```

Then inspect that checkout's `AGENTS.md` instructions and actual `PlayerController`, candidate-generation and AI catalog interfaces. The intended integration is to register the exported Rust scorer as a placement policy, use the game's own engine to enumerate candidates, encode their post-placement boards using `portable/features.rs`, and execute the chosen candidate through the game's normal controller. The trained model must not bypass the game's move legality or garbage handling.

Remaining validation requires compiling the actual game, confirming state/coordinate conversions against its engine, and playing or simulating human-versus-model and bot-versus-bot rounds. The portable export alone is not a completed Tetr Online integration.
