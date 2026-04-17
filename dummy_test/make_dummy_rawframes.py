from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random

data_dir = 'dummy_test/rawframes_dummy'

root = Path(data_dir)
classes = ["class0", "class1"]
clips_per_class = 3
frames_per_clip = 16
size = (160, 120)
root.mkdir(parents=True, exist_ok=True)

def mk_clip(cls, idx):
    clip_dir = root / f"{cls}_clip{idx}"
    clip_dir.mkdir(parents=True, exist_ok=True)
    color = (random.randrange(256), random.randrange(256), random.randrange(256))
    for fidx in range(1, frames_per_clip+1):
        img = Image.new("RGB", size, color)
        d = ImageDraw.Draw(img)
        d.text((10, 10), f"{cls} #{idx} f{fidx}", fill=(255,255,255))
        img.save(clip_dir / f"img_{fidx:05d}.jpg", quality=85)
    return clip_dir

clips = []
for ci, cls in enumerate(classes):
    for j in range(clips_per_class):
        clips.append((mk_clip(cls, j), ci))

# split: 2 train clips per class, 1 val per class
train_list = []
val_list = []
for ci, cls in enumerate(classes):
    cls_clips = [c for c in clips if str(c[0]).endswith(cls)]
    # simpler: iterate clips and bucket by class name
for (p, lab) in clips:
    name = p.name
    line = f"{name} {frames_per_clip} {lab}\n"
    if name.endswith("0") or name.endswith("1"):  # 2/3 train
        train_list.append(line)
    else:
        val_list.append(line)

(root/"train.txt").write_text("".join(train_list))
(root/"val.txt").write_text("".join(val_list))
print("Wrote:", root)
