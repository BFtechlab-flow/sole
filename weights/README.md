# Trained weights

No supervised checkpoint has been published yet: `manifest.json` lists no
files. This folder will describe the five fold checkpoints used for the final
submission once they exist.

Why: the organizers must be able to reproduce the submitted predictions from
the public repository alone. Checkpoints (~100 MB each) do not belong in git,
so they are attached to a GitHub Release and listed here with their SHA-256.

## Publishing (after training)

1. Write the manifest (the URLs point to the release you are about to create):

   ```bash
   python tools/weights.py manifest \
     --checkpoints checkpoints/fold{0,1,2,3,4}_best.pt \
     --base-url https://github.com/BFtechlab-flow/sole/releases/download/v1.0/ \
     --config configs/filanet_tuned.yaml
   ```

2. On GitHub: *Releases → Draft a new release*, tag `v1.0`, attach the five
   `fold*_best.pt` files, publish.
3. Commit `weights/manifest.json` and `configs/filanet_tuned.yaml`.

## Reproducing the submission

```bash
python tools/weights.py fetch --dest checkpoints     # download + SHA-256 check
python scripts/infer.py --config configs/filanet_tuned.yaml \
  --weights checkpoints/fold{0,1,2,3,4}_best.pt --output submission.csv
```
