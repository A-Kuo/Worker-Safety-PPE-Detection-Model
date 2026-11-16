# Security

## Reporting

Please report vulnerabilities privately through GitHub's "Report a vulnerability" button on the
repository's Security tab, not in a public issue. Include the version, platform and steps to reproduce.

## Scope and things to know

- **Model files.** `ppe fetch-model` installs a file only if its size and SHA-256 match the values
  packaged with the code. Loading an ONNX model from any other source with `--weights` is at your own
  risk. The optional Torch path (`.pt` files, `PPE_ALLOW_TORCH=1`) uses pickle-based checkpoints, which
  can execute code when loaded: only load checkpoints you trust.
- **The web demo (`app/`).** It binds to `127.0.0.1` by default and has no authentication. Do not
  expose it to a network you do not control. It accepts uploaded images and video.
- **Camera and recordings.** The live window reads a camera only when started, and writes clips or
  snapshots only on request (`R`, `--record`, `--snapshot`), to the project directory (the current directory
  for an installed package).
- **Not a safety system.** Detection misses and false alarms are documented in `docs/MODEL_CARD.md`.
  A missed detection is expected behaviour of a beta model, not a vulnerability.

Only the latest release is supported.
