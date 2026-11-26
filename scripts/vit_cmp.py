"""Compare board outputs with the float ONNX reference saved by vit_ref.py.
   python3 scripts/vit_cmp.py board_out/pcq board_out/i16      (each dir = output of scripts/board_eval.sh for one .nb)"""
import numpy as np, sys, glob, os
n = len(glob.glob('vit/test/v_*.ref.npy'))
for d in sys.argv[1:]:
    t1 = t5 = 0; c = []
    for k in range(n):
        r = np.load(f'vit/test/v_{k:02d}.ref.npy'); l = np.fromfile(f'{d}/v_{k:02d}.bin', np.float32)
        t1 += l.argmax() == r.argmax(); t5 += r.argmax() in np.argsort(-l)[:5]; c.append(np.corrcoef(r, l)[0, 1])
    print(f'{d}: top-1 agrees with float {t1}/{n}, float top-1 inside quantised top-5 {t5}/{n}, logit correlation mean {np.mean(c):.3f} min {min(c):.3f}')
