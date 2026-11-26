#!/usr/bin/env python3
"""Classify a photo with the DeiT-tiny .nb on the Radxa NPU (run ON THE BOARD; needs numpy + opencv).

  python3 classify.py photo.jpg --nb models/deit_tiny_int8pcq.nb --labels config.json --bench ~/nb_bench/nb_bench

Preprocessing: resize the short side to 224, centre-crop 224x224, RGB, raw uint8 (mean/std are inside the .nb).
"""
import argparse, json, os, subprocess, tempfile
import numpy as np, cv2
ap = argparse.ArgumentParser(); ap.add_argument('image'); ap.add_argument('--nb', required=True); ap.add_argument('--labels', required=True)
ap.add_argument('--bench', default=os.path.expanduser('~/nb_bench/nb_bench')); a = ap.parse_args()
im = cv2.imread(a.image); h, w = im.shape[:2]; s = 224 / min(h, w)
im = cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA); h, w = im.shape[:2]
y, x = (h - 224) // 2, (w - 224) // 2; rgb = cv2.cvtColor(im[y:y + 224, x:x + 224], cv2.COLOR_BGR2RGB)
labels = json.load(open(a.labels))['id2label']
with tempfile.TemporaryDirectory() as t:
    rgb.tofile(t + '/in.bin'); env = dict(os.environ, LD_LIBRARY_PATH=os.path.expanduser('~/awnn_inference/install/lib'))
    subprocess.run([a.bench, '-m', a.nb, '-n', '1', '-i0', t + '/in.bin', '-o', t + '/out.bin'], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    logits = np.fromfile(t + '/out.bin', np.float32)
p = np.exp(logits - logits.max()); p /= p.sum()
for i in np.argsort(-logits)[:5]: print(f'{p[i]*100:5.1f}%  {labels[str(i)]}')
