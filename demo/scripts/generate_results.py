"""Regenerate the ten UL-UNAS comparisons using the unmodified upstream model."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO))
from ulunas import ULUNAS

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

torch.set_num_threads(4)
torch.manual_seed(0)
checkpoint = REPO / 'checkpoints/model_trained_on_dns3.tar'
model = ULUNAS().cpu().eval()
model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['model'], strict=True)
inputs = sorted((ROOT / 'samples/DNS3_test_set').glob('*_noisy.wav'))
inputs += sorted(p for p in (ROOT / 'samples/DNS3_blind_test_set').glob('*.wav')
                 if not p.stem.endswith(('_GTCRN', '_RNNoise', '_UL-UNAS')))
assert len(inputs) == 10
results = []
for path in inputs:
    audio, rate = sf.read(path, dtype='float32')
    assert rate == 16000 and audio.ndim == 1, (path, rate, audio.shape)
    with torch.inference_mode():
        enhanced = model(torch.from_numpy(audio).unsqueeze(0)).squeeze(0).numpy()
    assert enhanced.shape == audio.shape and np.isfinite(enhanced).all()
    assert not np.array_equal(audio, enhanced)
    stem = path.stem.removesuffix('_noisy')
    output = path.with_name(stem + '_UL-UNAS.wav')
    # Match upstream inference.py's soundfile WAV default: PCM_16, no normalization.
    sf.write(output, enhanced, rate, subtype='PCM_16')
    results.append(dict(input=str(path.relative_to(ROOT)), output=str(output.relative_to(ROOT)),
                        sample_rate=rate, frames=len(audio), duration_seconds=len(audio)/rate,
                        input_sha256=digest(path), output_sha256=digest(output),
                        peak_before_pcm_encoding=float(np.abs(enhanced).max())))
    print(f'{path.name}: {len(audio)/rate:.2f}s -> {output.name}', flush=True)
manifest = dict(model='UL-UNAS', checkpoint='model_trained_on_dns3.tar',
                checkpoint_sha256=digest(checkpoint), model_source_sha256=digest(REPO/'ulunas.py'),
                ul_unas_revision='00f7c700da43d38347f30a6ccebd86fcbc798e07',
                gtcrn_demo_revision='0e738b64fc7130557f870a3165db68b5ba1d7a64',
                inference='Upstream ULUNAS.forward, CPU float32, eval, inference_mode; no normalization',
                torch_version=torch.__version__, numpy_version=np.__version__, soundfile_version=sf.__version__,
                results=results)
(ROOT / 'results.json').write_text(json.dumps(manifest, indent=2)+'\n')
