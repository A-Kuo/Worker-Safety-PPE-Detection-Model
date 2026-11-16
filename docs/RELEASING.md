# Releasing

Model weights are not in git and cannot be built in CI, so a release has one manual step: attaching
the two ONNX files. Everything else is checked by automation.

## Before tagging

1. `version` in `pyproject.toml`, `release` and the URLs in `src/ppe/models.json`, and the hashes in
   `docs/MODEL_CARD.md` must agree: `python scripts/check_release.py --tag vX.Y.Z`.
2. If a model file changed, re-run the evidence (`ppe eval`, `ppe verify-npu`, `ppe bench`,
   `scripts/summarize_evidence.py`) and update `models.json` with the new size and SHA-256.
3. CI is green on `main`.

## Cut the release

```bash
git tag vX.Y.Z && git push origin vX.Y.Z
```

The `Release` workflow tests, builds the wheel and sdist, runs the wheel smoke and creates a **draft**
GitHub release with the wheel, sdist, `SHA256SUMS.txt` and the model card. Nothing is public yet.

Attach the model files from the machine that holds them, then publish the draft:

```bash
sha256sum vest_specialist.onnx vest_specialist.int8.onnx   # must match src/ppe/models.json
gh release upload vX.Y.Z vest_specialist.onnx vest_specialist.int8.onnx
gh release edit vX.Y.Z --draft=false
```

## After publishing

Run the **Post-release check** workflow with the tag. On Ubuntu and Windows it installs the released
wheel with no source checkout and no torch, runs `ppe fetch-model --all` (which rejects any file whose
hash differs from the packaged registry) and runs both models.

On the NPU machine:

```powershell
python -m pip install "ppe-X.Y.Z-py3-none-any.whl[npu,gui]"
ppe fetch-model --all
ppe verify-npu --providers openvino
ppe window --strict-npu --providers openvino
```

## If something is wrong after publishing

Do not replace an asset in place: `ppe fetch-model` in already-installed copies pins the old hash and
would reject the new file. Cut a new patch version instead.
