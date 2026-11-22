# Dog gait model (CC BY-NC 4.0)

`dog-gait-model.json` is the per-gait, dimensionless gait model that `--dog-style` plans to
(`trajgen/dogstyle.py`): stride timing, duty, touchdown offsets, swing paths, footholds and trunk
waveforms of a real dog, as functions of Froude speed. `dog-gait-model.txt` is its human-readable
summary. Both were fitted by `tools/dog/dog_gait_model.py` from the source data below.

## Source and licence

**AI4Animation MANN dog motion capture**: H. Zhang, S. Starke, T. Komura, J. Saito,
"Mode-Adaptive Neural Networks for Quadruped Motion Control", *ACM Transactions on Graphics* 37(4),
SIGGRAPH 2018. Data from https://github.com/sebastianstarke/AI4Animation (`MotionCapture.zip`,
https://starke-consult.de/AI4Animation/SIGGRAPH_2018/MotionCapture.zip, 51 BVH captures).

The motion capture is licensed under **Creative Commons Attribution-NonCommercial 4.0
International (CC BY-NC 4.0)**, https://creativecommons.org/licenses/by-nc/4.0/. The files in this
directory are a derived work, adapted from it: strides segmented from the captures and summarised as
fitted lines and harmonics (method in `tools/dog/dog_gait_model.py`). They are distributed under the
same CC BY-NC 4.0 licence, **not** under the Apache-2.0 licence of the rest of this repository.

Use them for non-commercial work only. Motions planned with `--dog-style`, and anything trained on
those motions, derive from this data and carry the same restriction. Everything else in trajopt
works without it.

## Rebuilding the model

```bash
mkdir -p out/assets/ai4animation_mann && cd out/assets/ai4animation_mann
curl -LO https://starke-consult.de/AI4Animation/SIGGRAPH_2018/MotionCapture.zip
unzip -q MotionCapture.zip -d extracted && cd -       # the BVH files must end up directly in extracted/
.venv/bin/python tools/dog/gait.py                   # out/dog/gait_stats.json (the dog's leg-root height h)
.venv/bin/python tools/dog/dog_gait_model.py         # data/dog/dog-gait-model.json and .txt
```

`MANN_DIR=/path/to/bvh/files` points the tools at another copy of the captures. Rebuilt from the
51 captures, the model matches the shipped file to 2e-12 (relative).

The model is dimensionless (lengths / the dog's leg-root height h = 0.3446 m, speeds as
s = v / sqrt(g h)). `trajgen.pipeline solve walk --dog-style data/dog/dog-gait-model.json` scales it to
the robot's own leg-root height and hip radius, measured on its stand pose. The `.txt` summary and the
model's `selection` speeds are in units of a reference robot with a 0.40 m leg-root height
(`h0_robot`).
