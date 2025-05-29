import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from transformers import XCLIPProcessor, XCLIPModel
from decord import VideoReader, cpu as decord_cpu

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型与处理器
processor = XCLIPProcessor.from_pretrained("microsoft/xclip-base-patch32")
model = XCLIPModel.from_pretrained("microsoft/xclip-base-patch32")
model = torch.compile(model)  # PyTorch ≥ 2.0
model.to(device)
model.eval()

# 使用 decord 读取视频帧
def extract_frames(video_path, num_frames=8):
    try:
        vr = VideoReader(video_path, ctx=decord_cpu(0))
        total_frames = len(vr)
        indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
        frames = [Image.fromarray(vr[i].asnumpy()) for i in indices]
        return frames
    except Exception as e:
        print(f"[Warning] Failed to read {video_path}: {e}")
        return None

# 加载数据集
ds_train = load_dataset("smpchallenge/SMP-Video", "videos", split="train")
ds_test = load_dataset("smpchallenge/SMP-Video", "videos", split="test")
print(f"Train videos: {len(ds_train)}, Test videos: {len(ds_test)}")

# 收集视频路径信息
video_infos = []
for sample in ds_train:
    video_infos.append(("train", sample["uid"], sample["vid"]))
for sample in ds_test:
    video_infos.append(("test", sample["uid"], sample["vid"]))

# 单视频推理
embeddings = []
for split, uid, vid in tqdm(video_infos):
    video_path = f"dataset/{split}/{uid}/{vid}.mp4"
    frames = extract_frames(video_path)
    if frames is None:
        continue

    inputs = processor(videos=frames, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model.get_video_features(**inputs)
    embeddings.append(output.cpu().numpy())

# 保存嵌入向量
final_embeddings = np.vstack(embeddings)
np.save("smp_video_embeddings.npy", final_embeddings)
print(f"Saved {final_embeddings.shape[0]} embeddings to smp_video_embeddings.npy")
