import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import open_clip
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
from torchvision import transforms


model_name = "ViT-L-14-336"
pretrained = "/home/user/model/open_clip_model.safetensors" # "openai"
agiqa_csv_path = "/home/user/datasets1/agiqa/AGIQA_label_updated.csv"
agiqa_root = "/home/user/datasets1/agiqa/AGIQA-3K"
tid2013_txt_path = "datasets1/tid2013/mos_with_names.txt"  
tid2013_csv_path = "/home/user/datasets1/tid2013/tid2013_label_updated.csv"
tid2013_root = "datasets1/tid2013/distorted_images"
koniq_csv_path = "/home/user/datasets1/koniq/koniq10k_scores_and_distributions/koniq10k_scores_and_distributions.csv"
koniq_root = "/home/user/datasets1/koniq/koniq10k/imgs"
csiq_csv_path = "/home/user/datasets1/csiq/CSIQ/CSIQ_label_updated.csv"
csiq_root = "/home/user/datasets1/csiq/CSIQ/All_Image"
live_csv_path = "datasets1/live/labeledLIVE/live_merged_with_path.csv"
live_root = "datasets1/live/labeledLIVE"
livec_csv_path="/home/user/datasets1/clive/ChallengeDB_release/image_analysis_results_with_mos.csv"
livec_root="/home/user/datasets1/clive/ChallengeDB_release/Images"
epochs = 60
batch_size = 8
lr = 1e-4
dataset = "tid2013" # datsset_name = {"agiqa", "tid2013", "koniq", "csiq","live", "livec"}
model_this = "iqa_clip_cross_attention_double_12"  
trial = 0 # trial = 1 for debug, otherwise 0


# ================================
# 1. 数据集类
# ================================
# ================================
# 1. 统一的双文本数据集类
# ================================
class DualTextImageDataset(Dataset):
    def __init__(self, dataframe, image_root, preprocess, dataset_name):
        self.data = dataframe.reset_index(drop=True)
        self.image_root = image_root
        self.preprocess = preprocess
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 图像路径：优先 "image"，其次 "图像名"，再尝试 "path"（适配 LIVE）
        image_key = "image" if "image" in row else ("图像名" if "图像名" in row else "path")
        image_path = os.path.join(self.image_root, row[image_key])

        # 文本1：依次尝试 content1 → 内容1 → ContentDescription（LIVE）
        text1 = row.get("content1", row.get("内容1", row.get("ContentDescription", "")))

        # 文本2：依次尝试 content2 → 内容2 → QualityPerspective（LIVE）
        text2 = row.get("content2", row.get("内容2", row.get("QualityPerspective", "")))

        # 标签：依次尝试 mos_quality → dmos → MOS（覆盖 LIVE 的 dmos）
        label = float(
            row.get("mos_quality", 
                   row.get("dmos", 
                          row.get("MOS", 0.0)))
        )
        if dataset == 'live':
            label = label / 100.0
        if dataset == 'livec':
            label = label / 100.0

        image = self.preprocess(Image.open(image_path).convert("RGB"))
        return image, text1, text2, label
    


# ================================
# 2. CLIP融合模型
# ================================
import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    """Cross-Attention: visual queries attend to text keys/values."""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, vis_tokens, txt_tokens):
        B, N_v, D = vis_tokens.shape
        N_t = txt_tokens.shape[1]

        q = self.q_proj(vis_tokens).view(B, N_v, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(txt_tokens).view(B, N_t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(txt_tokens).view(B, N_t, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, N_v, D)
        return self.out_proj(out)



    
class CLIPBranchWithCrossAttention(nn.Module):
    def __init__(self, model, tokenizer, vis_pre_layers=18, txt_pre_layers=11, cross_attn_heads=8, cross_attn_dropout=0.3):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.vis_pre_layers = vis_pre_layers
        self.txt_pre_layers = txt_pre_layers

        self.visual_width = model.visual.transformer.width      # 1024
        self.text_width = model.transformer.width               # 768
        self.output_dim = model.visual.output_dim               # 768

        # Text → Visual projection
        self.text_to_visual = nn.Linear(self.text_width, self.visual_width, bias=False)

        # Cross Attention
        self.cross_attn = CrossAttention(
            dim=self.visual_width,
            num_heads=cross_attn_heads,
            dropout=cross_attn_dropout
        )
        self.cross_attn_norm = nn.LayerNorm(self.visual_width)

        # 🔒 Freeze all
        for p in self.model.parameters():
            p.requires_grad = False

        # 🔓 Unfreeze last 2 visual blocks + last 2 text blocks
        for blk in self.model.visual.transformer.resblocks[vis_pre_layers:]:
            for p in blk.parameters():
                p.requires_grad = True
        for blk in self.model.transformer.resblocks[txt_pre_layers:]:
            for p in blk.parameters():
                p.requires_grad = True

        # 🔓 Unfreeze custom modules
        for m in [self.text_to_visual, self.cross_attn, self.cross_attn_norm]:
            for p in m.parameters():
                p.requires_grad = True

    def forward(self, images, texts):
        device = images.device
        B = images.size(0)

        # ===== Encode Text (partial) =====
        text_tokens = self.tokenizer(texts).to(device)
        with torch.set_grad_enabled(True):  # allow grad for unfrozen layers
            x_txt = self.model.token_embedding(text_tokens)
            x_txt = x_txt + self.model.positional_embedding[:x_txt.size(1)]
            x_txt = x_txt.permute(1, 0, 2)  # [L, B, D]

            for i in range(self.txt_pre_layers):
                x_txt = self.model.transformer.resblocks[i](x_txt)

        x_txt = x_txt.permute(1, 0, 2)  # [B, L, D]
        txt_for_attn = self.text_to_visual(x_txt)  # [B, L, 1024]

        # ===== Encode Image (partial) =====
        x = self.model.visual.conv1(images)
        x = x.reshape(B, x.shape[1], -1).permute(0, 2, 1)
        cls_token = self.model.visual.class_embedding.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.model.visual.positional_embedding
        x = self.model.visual.ln_pre(x)

        for i in range(self.vis_pre_layers):
            x = self.model.visual.transformer.resblocks[i](x)

        # ===== Cross Attention =====
        residual = x
        x = residual + self.cross_attn(x, txt_for_attn)
        x = self.cross_attn_norm(x)

        # ===== Finish Image Encoder =====
        for i in range(self.vis_pre_layers, len(self.model.visual.transformer.resblocks)):
            x = self.model.visual.transformer.resblocks[i](x)

        img_feat = self.model.visual.ln_post(x[:, 0])
        img_feat = img_feat @ self.model.visual.proj  # [B, 768]

        # ===== Finish Text Encoder =====
        x_txt = x_txt.permute(1, 0, 2)
        for i in range(self.txt_pre_layers, len(self.model.transformer.resblocks)):
            x_txt = self.model.transformer.resblocks[i](x_txt)
        x_txt = x_txt.permute(1, 0, 2)
        x_txt = self.model.ln_final(x_txt)
        eot_idx = text_tokens.argmax(dim=-1)
        txt_feat = x_txt[torch.arange(B), eot_idx]  # [B, 768]

        return img_feat, txt_feat

class DualCLIPCrossAttentionModel(nn.Module):
    def __init__(self, clip_model_1, clip_model_2, tokenizer_1, tokenizer_2, output_dim=768):
        super().__init__()
        self.branch1 = CLIPBranchWithCrossAttention(clip_model_1, tokenizer_1)
        self.branch2 = CLIPBranchWithCrossAttention(clip_model_2, tokenizer_2)

        # Regressor on concatenated features
        self.regressor = nn.Sequential(
            nn.Linear(output_dim * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

        # Learnable alignment weights
        self.log_k1 = nn.Parameter(torch.tensor(np.log(0.1)))
        self.log_k2 = nn.Parameter(torch.tensor(np.log(0.1)))

    def forward(self, images, texts1, texts2):
        img1, txt1 = self.branch1(images, texts1)
        img2, txt2 = self.branch2(images, texts2)

        fused = torch.cat([img1, txt1, img2, txt2], dim=1)
        pred = self.regressor(fused).squeeze(-1)
        return pred, (img1, txt1), (img2, txt2)


def hybrid_loss(pred, y, img_feat, txt_feat, k):
    mse_pred = F.mse_loss(pred, y)
    cos_sim = F.cosine_similarity(img_feat, txt_feat, dim=1)
    total_loss = mse_pred + k * (1 - cos_sim).mean()
    return total_loss, mse_pred.item(), (1 - cos_sim).mean().item()

def dual_hybrid_loss(pred, y, feats1, feats2, k1, k2):
    mse_pred = F.mse_loss(pred, y)
    img1, txt1 = feats1
    img2, txt2 = feats2
    cos1 = F.cosine_similarity(img1, txt1, dim=1)
    cos2 = F.cosine_similarity(img2, txt2, dim=1)
    align_loss = k1 * (1 - cos1).mean() + k2 * (1 - cos2).mean()
    return mse_pred + align_loss, mse_pred.item(), align_loss.item()

# ================================
# 3. 评估函数
# ================================
def evaluate(model, dataloader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for images, texts1, texts2, y in tqdm(dataloader, desc="Eval"):
            images = images.to(device)
            y = y.to(device, dtype=torch.float32).view(-1)
            pred, _, _ = model(images, texts1, texts2)
            preds.extend(pred.cpu().numpy())
            labels.extend(y.cpu().numpy())
    preds, labels = np.array(preds), np.array(labels)
    sp = spearmanr(preds, labels)[0]
    pr = pearsonr(preds, labels)[0]
    mse = np.mean((preds - labels) ** 2)

    savedir = f"{model_this}_eval_{dataset}"
    os.makedirs(savedir, exist_ok=True)
    return sp, pr, mse

# ================================
# 4. 训练函数（只训练 text_proj）
# ================================
def train(model, train_loader, val_loader, device, epochs=5, lr=1e-4, model_save_path="best_dual.pth", dataset_name="default"):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    history = {'loss': [], 'spearman': [], 'pearson': [], 'mse': [], 'k1': [], 'k2': []}
    best_sp = -1

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, texts1, texts2, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = images.to(device)
            y = y.to(device, dtype=torch.float32).view(-1)

            optimizer.zero_grad()
            pred, feats1, feats2 = model(images, texts1, texts2)
            k1 = torch.exp(model.log_k1)
            k2 = torch.exp(model.log_k2)
            loss, mse_pred, align_loss = dual_hybrid_loss(pred, y, feats1, feats2, k1, k2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        sp, pr, mse = evaluate(model, val_loader, device)
        scheduler.step(sp)

        history['loss'].append(avg_loss)
        history['spearman'].append(sp)
        history['pearson'].append(pr)
        history['mse'].append(mse)
        history['k1'].append(k1.item())
        history['k2'].append(k2.item())

        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val SP: {sp:.4f} | PR: {pr:.4f}")

        if sp > best_sp:
            best_sp = sp
            torch.save(model.state_dict(), model_save_path)
            print("✅ Saved best model.")
        
    save_dir = f"{model_this}_train_procedure_{dataset_name}"
    os.makedirs(save_dir, exist_ok=True)

    dataframe = pd.DataFrame(history)
    dataframe.to_csv(os.path.join(save_dir, 'history.csv'), index=False)

    for metric in ['loss', 'spearman', 'pearson', 'mse', 'k1', 'k2']:
        plt.figure()
        plt.plot(range(1, epochs + 1), history[metric], label=metric)
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.xticks(range(1, epochs + 1, 5))  # 设置横轴刻度间隔为5
        plt.legend()
        plt.savefig(os.path.join(save_dir, f'{metric}.png'))
        plt.close()

def generate_model_path(base_path, dataset_name):
    """
    根据基础路径和数据集名称生成唯一的模型保存路径。
    
    参数:
    - base_path: 模型保存的基础目录。
    - dataset_name: 数据集的名称。
    
    返回:
    - 唯一的模型保存路径。
    """
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    return os.path.join(base_path, f'{model_this}_{dataset_name}_linear.pth')

# ================================
# 5. 主函数
# ================================
def main(dataset_name="default"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load TWO independent CLIP models
    model1, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model2, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)  # ← 独立实例！
    tokenizer1 = open_clip.get_tokenizer(model_name)
    tokenizer2 = open_clip.get_tokenizer(model_name)

    net = DualCLIPCrossAttentionModel(
        clip_model_1=model1,
        clip_model_2=model2,
        tokenizer_1=tokenizer1,
        tokenizer_2=tokenizer2,
        output_dim=model1.visual.output_dim
    ).to(device)

    # ... rest unchanged: data loading with DualTextImageDataset, train, evaluate ...


    if dataset_name == "agiqa":
        df = pd.read_csv(agiqa_csv_path)
    elif dataset_name == "tid2013":
        df = pd.read_csv(tid2013_csv_path)
    elif dataset_name == "koniq":
        df = pd.read_csv(koniq_csv_path)
    elif dataset_name == "csiq":
        df = pd.read_csv(csiq_csv_path)
        # 删除有空值的行
        df = df.dropna().reset_index(drop=True)
    elif dataset_name == "live":
        df = pd.read_csv(live_csv_path)
    elif dataset_name == "livec":
        df = pd.read_csv(livec_csv_path)

    
    

    # ---- 划分训练/验证/测试集 ----
    train_df, temp_df = train_test_split(df, test_size=0.2)
    val_df, test_df = train_test_split(temp_df, test_size=0.5)

    if trial == 1:
        train_df = train_df.sample(frac=0.1, random_state=42)
        val_df = val_df.sample(frac=0.1, random_state=42)
        test_df = test_df.sample(frac=0.1, random_state=42)

        # 在 main() 中替换数据集构建部分
    if dataset_name in ["agiqa", "tid2013", "csiq", "live", "livec"]:
        train_dataset = DualTextImageDataset(train_df, image_root=agiqa_root if dataset_name=="agiqa" else (tid2013_root if dataset_name=="tid2013" else (csiq_root if dataset_name=="csiq" else (live_root if dataset_name=="live" else livec_root))), preprocess=preprocess, dataset_name=dataset_name)
        val_dataset = DualTextImageDataset(val_df, image_root=agiqa_root if dataset_name=="agiqa" else (tid2013_root if dataset_name=="tid2013" else (csiq_root if dataset_name=="csiq" else (live_root if dataset_name=="live" else livec_root))), preprocess=preprocess, dataset_name=dataset_name)
        test_dataset = DualTextImageDataset(test_df, image_root=agiqa_root if dataset_name=="agiqa" else (tid2013_root if dataset_name=="tid2013" else (csiq_root if dataset_name=="csiq" else (live_root if dataset_name=="live" else livec_root))), preprocess=preprocess, dataset_name=dataset_name)
    elif dataset_name == "koniq":
        train_dataset = DualTextImageDataset(train_df, image_root=koniq_root, preprocess=preprocess, dataset_name="koniq")
        val_dataset = DualTextImageDataset(val_df, image_root=koniq_root, preprocess=preprocess, dataset_name="koniq")
        test_dataset = DualTextImageDataset(test_df, image_root=koniq_root, preprocess=preprocess, dataset_name="koniq")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"Dataset: {dataset_name} | Train size: {len(train_dataset)} | Val size: {len(val_dataset)} | Test size: {len(test_dataset)}")

           
    # 展示前两张图片
    # for i in range(2):
    #     img, text1, text2, label = train_dataset[i]
    #     img_np = img.permute(1, 2, 0).numpy()
    #     # 将像素值从 [-1,1] 或 [0,1] 转换到 [0,1] 以便正确显示（CLIP preprocess 后是标准化的）
    #     img_vis = np.clip((img_np * [0.26862954, 0.26130258, 0.27577711] + [0.48145466, 0.4578275, 0.40821073]), 0, 1)
    #     plt.imshow(img_vis)
    #     plt.title(f"Text1: {text1}\nText2: {text2}\nLabel: {label:.3f}")
    #     plt.axis('off')
    #     plt.show()

    model_saved_path = generate_model_path(model_this, dataset_name)  # 获取模型保存路径
    print(model_saved_path)

        # # # ---- 训练 ----
    train(net, train_loader, val_loader, device, epochs=epochs, lr=lr, model_save_path=model_saved_path, dataset_name=dataset_name)

        # ---- 测试 ----
    net.load_state_dict(torch.load(model_saved_path))
    sp, pr, mse = evaluate(net, test_loader, device)
    print(f"✅ Test Spearman: {sp:.4f} | Pearson: {pr:.4f} | MSE: {mse:.4f}")

if __name__ == "__main__":
    main(dataset_name=dataset) 
