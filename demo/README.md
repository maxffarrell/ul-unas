# Audio comparison demo

Ten noisy recordings from the GTCRN demo, with RNNoise, GTCRN, and newly generated UL-UNAS outputs. The DNS3 test set also includes clean references. No objective benchmark scores are claimed.

Open `index.html` in a browser, or from the repository root run:

```sh
python3 -m http.server 8000 --directory demo
```

Visit http://localhost:8000. The page is static and can be hosted on GitHub Pages or any static host.

## Regenerate UL-UNAS outputs

Use a Python 3.12 virtual environment and the repository's existing model and DNS3 checkpoint:

```sh
python -m pip install -r demo/requirements.txt
python demo/scripts/generate_results.py
```

Inference uses the unmodified `ULUNAS.forward` on CPU in float32, evaluation mode, and inference mode. All inputs are mono at 16 kHz. Output is WAV PCM_16, matching `inference.py`, without normalization. `results.json` records source revisions, model and checkpoint hashes, input/output hashes, durations, and runtime versions.

The 35 baseline audio files and original page/CSS come from https://github.com/Xiaobin-Rong/gtcrn_demo at `0e738b64fc7130557f870a3165db68b5ba1d7a64`, under CC0 (`LICENSE-GTCRN`). Baseline audio files are unchanged. UL-UNAS results use this repository's checkpoint/model at `00f7c700da43d38347f30a6ccebd86fcbc798e07`, under its MIT license.
