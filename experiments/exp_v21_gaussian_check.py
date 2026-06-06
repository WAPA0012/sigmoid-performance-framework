"""
exp_v21_gaussian_check.py — 高斯性检验
========================================

检验CLIP和训练模型的embedding是否满足高斯假设。
用kurtosis和Shapiro-Wilk检验。
"""

import sys, os
sys.path.insert(0, '/Users/wp/ZCodeProject/PRISM')

import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats
from transformers import CLIPModel, CLIPProcessor
from torchvision import datasets, transforms

device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f"Device: {device}")
CLIP_LOCAL = '/Users/wp/ZCodeProject/PRISM/data/clip_model/clip-vit-base-patch32'


def analyze_features(features, name):
    """分析特征的高斯性"""
    features = features.numpy() if torch.is_tensor(features) else features
    n_samples, n_dims = features.shape
    print(f"\n{'='*60}")
    print(f"  {name}: {n_samples} samples x {n_dims} dims")
    print(f"{'='*60}")

    # 1. 全局统计
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    print(f"  Mean norm: {np.linalg.norm(mean):.4f}")
    print(f"  Mean std per dim: {std.mean():.6f}")
    print(f"  Std of stds: {std.std():.6f}")

    # 2. Kurtosis (高斯=3)
    # 逐维度kurtosis
    kurt_per_dim = stats.kurtosis(features, axis=0, fisher=False)  # Pearson kurtosis (normal=3)
    excess_kurt = kurt_per_dim - 3.0
    print(f"\n  Kurtosis (normal = 3.0):")
    print(f"    Mean: {kurt_per_dim.mean():.4f}")
    print(f"    Median: {np.median(kurt_per_dim):.4f}")
    print(f"    Std: {kurt_per_dim.std():.4f}")
    print(f"    Excess kurtosis (mean): {excess_kurt.mean():.4f}")
    print(f"    |excess kurt| < 1: {(np.abs(excess_kurt) < 1).mean()*100:.1f}% of dims")
    print(f"    |excess kurt| < 2: {(np.abs(excess_kurt) < 2).mean()*100:.1f}% of dims")

    # 3. Skewness (高斯=0)
    skew_per_dim = stats.skew(features, axis=0)
    print(f"\n  Skewness (normal = 0.0):")
    print(f"    Mean: {skew_per_dim.mean():.4f}")
    print(f"    |skew| < 0.5: {(np.abs(skew_per_dim) < 0.5).mean()*100:.1f}% of dims")

    # 4. Shapiro-Wilk (在子集上，因为SW有5000样本限制)
    np.random.seed(42)
    n_test = min(5000, n_samples)
    idx = np.random.choice(n_samples, n_test, replace=False)
    subset = features[idx]

    # 逐维度SW检验 (取前10个维度)
    n_dims_test = min(20, n_dims)
    p_values = []
    for d in range(n_dims_test):
        _, p = stats.shapiro(subset[:, d])
        p_values.append(p)
    p_values = np.array(p_values)
    n_normal = (p_values > 0.05).sum()
    print(f"\n  Shapiro-Wilk (first {n_dims_test} dims, α=0.05):")
    print(f"    Normal: {n_normal}/{n_dims_test} ({n_normal/n_dims_test*100:.0f}%)")
    print(f"    Mean p-value: {p_values.mean():.4f}")

    # 5. 多元正态性: Mahalanobis距离 → chi-square
    cov = np.cov(features.T)
    try:
        cov_inv = np.linalg.pinv(cov)
        diff = features - mean
        mahal = np.sum(diff @ cov_inv * diff, axis=1)

        # chi-square test
        chi2_stat, chi2_p = stats.kstest(mahal, 'chi2', args=(n_dims,))
        print(f"\n  Mahalanobis distance → χ²({n_dims}) test:")
        print(f"    KS statistic: {chi2_stat:.4f}")
        print(f"    p-value: {chi2_p:.4f}")
        if chi2_p > 0.05:
            print(f"    → PASS: consistent with multivariate Gaussian")
        else:
            print(f"    → Reject multivariate Gaussian (but univariate margins may still be OK)")
    except:
        print("  Mahalanobis test failed (singular covariance)")

    # 6. 各类条件下的分析
    return {
        'name': name,
        'n_samples': n_samples,
        'n_dims': n_dims,
        'mean_kurtosis': kurt_per_dim.mean(),
        'mean_excess_kurt': excess_kurt.mean(),
        'mean_skew': skew_per_dim.mean(),
        'pct_normal_dims': n_normal / n_dims_test,
    }


def main():
    print("Loading CLIP...")
    model = CLIPModel.from_pretrained(CLIP_LOCAL, local_files_only=True)
    processor = CLIPProcessor.from_pretrained(CLIP_LOCAL, local_files_only=True)
    model = model.to(device)
    model.eval()

    # 加载CIFAR-10
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])
    cifar10 = datasets.CIFAR10(root='/Users/wp/ZCodeProject/PRISM/data/cifar10',
                                train=False, download=True, transform=transform)

    np.random.seed(42)
    selected = []
    for c in range(10):
        idx = [i for i, (_, l) in enumerate(cifar10) if l == c]
        selected.extend(np.random.choice(idx, 100, replace=False))
    subset = torch.utils.data.Subset(cifar10, selected)
    loader = torch.utils.data.DataLoader(subset, batch_size=64, shuffle=False)

    # 提取特征
    print("Extracting features...")
    all_img_feats = []
    all_labels = []

    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
    text_prompts = [f"a photo of a {c}" for c in cifar10_classes]

    with torch.no_grad():
        text_inputs = processor(text=text_prompts, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        text_out = model.text_model(input_ids=text_inputs['input_ids'],
                                     attention_mask=text_inputs.get('attention_mask'))
        text_features = model.text_projection(text_out.pooler_output).cpu()

        for images, labels in loader:
            images = images.to(device)
            vision_out = model.vision_model(pixel_values=images)
            img_feats = model.visual_projection(vision_out.pooler_output)
            all_img_feats.append(img_feats.cpu())
            all_labels.extend(labels.numpy())

    image_features = torch.cat(all_img_feats, dim=0)
    print(f"Image features: {image_features.shape}")
    print(f"Text features: {text_features.shape}")

    # 分析原始特征
    results = []
    r1 = analyze_features(image_features, "CLIP Image Features (raw)")
    results.append(r1)

    r2 = analyze_features(text_features, "CLIP Text Features (raw)")
    results.append(r2)

    # 分析L2归一化后的特征
    img_norm = F.normalize(image_features, dim=-1)
    txt_norm = F.normalize(text_features, dim=-1)
    r3 = analyze_features(img_norm, "CLIP Image Features (L2-normalized)")
    results.append(r3)
    r4 = analyze_features(txt_norm, "CLIP Text Features (L2-normalized)")
    results.append(r4)

    # 每类条件下的图像特征
    labels_arr = np.array(all_labels)
    for c in range(3):  # 前3类
        mask = labels_arr == c
        class_feats = image_features[mask]
        r = analyze_features(class_feats, f"CLIP Image Features - Class '{cifar10_classes[c]}' only")
        results.append(r)

    # 总结
    print("\n" + "=" * 70)
    print("SUMMARY: Gaussian Assumption Check")
    print("=" * 70)
    print(f"\n  {'Feature Set':>45s} {'Kurt':>6s} {'Excess':>7s} {'Skew':>6s} {'Normal%':>8s}")
    for r in results:
        print(f"  {r['name']:>45s} {r['mean_kurtosis']:>6.2f} {r['mean_excess_kurt']:>7.2f} "
              f"{r['mean_skew']:>6.2f} {r['pct_normal_dims']*100:>7.1f}%")

    print(f"\n  INTERPRETATION:")
    print(f"  - Normal distribution: kurtosis ≈ 3.0, excess kurtosis ≈ 0, skew ≈ 0")
    print(f"  - If excess kurtosis < 1 and |skew| < 0.5 for most dims → approximately Gaussian")
    print(f"  - This validates (or rejects) the I = d·log(ratio) approximation")

    # 保存
    import json
    out = '/Users/wp/ZCodeProject/PRISM/paper/gaussian_check_results.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out}")


if __name__ == '__main__':
    main()
