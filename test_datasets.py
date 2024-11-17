from torch.utils.data import DataLoader
from utils.data_multifiles import MultiTeacherAlignedEmbeddingDataset
import torch
teacher_paths = "./Embeddings"
dataset = MultiTeacherAlignedEmbeddingDataset(teacher_paths)
batch_size = 64

dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=6,
        timeout=10000,
        prefetch_factor=512,
    )
for k, batch in enumerate(dataloader):
        if k >= 10:
            break
        img, teacher_emb = batch
        print(img, teacher_emb)