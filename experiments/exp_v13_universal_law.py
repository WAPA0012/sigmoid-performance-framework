"""
PRISM v13 — 寻找第一条定量规律
================================

假设: 跨模态检索准确率 = f(δ_intra / δ_inter)
      这个函数的形状对所有模态对相同

实验设计:
  1. 在 MNIST+FSDD 上, 不同训练阶段(epoch 1-30)测量:
     - δ_intra: 同类跨模态平均距离
     - δ_inter: 不同类跨模态平均距离
     - ratio = δ_inter / δ_intra
     - 检索准确率(cosine + VQ)
  2. 在 CIFAR-10+FSDD 上重复
  3. 看两条曲线是否重合

如果重合 → 第一条定量规律
如果不重合 → 至少学到函数形状是否依赖数据集
"""

import sys, os, struct, gzip, glob, pickle
sys.path.insert(0, '/Users/wp/ZCodeProject/PRISM')

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

device = torch.device('mps') if torch.backends.mps.is_available() else (
    torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print(f"Device: {device}")
BASE_DIR = '/Users/wp/ZCodeProject/PRISM'


# ============================================================
# Data Loading
# ============================================================

def load_mnist():
    def read_img(fn):
        with gzip.open(fn, 'rb') as f:
            magic, n, r, c = struct.unpack('>IIII', f.read(16))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r*c).astype(np.float32) / 255.0
    def read_lbl(fn):
        with gzip.open(fn, 'rb') as f:
            magic, n = struct.unpack('>II', f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)
    X = read_img(os.path.join(BASE_DIR, 'data/mnist/train-images-idx3-ubyte.gz'))
    Y = read_lbl(os.path.join(BASE_DIR, 'data/mnist/train-labels-idx1-ubyte.gz'))
    Xt = read_img(os.path.join(BASE_DIR, 'data/mnist/t10k-images-idx3-ubyte.gz'))
    Yt = read_lbl(os.path.join(BASE_DIR, 'data/mnist/t10k-labels-idx1-ubyte.gz'))
    return X, Y, Xt, Yt


def load_cifar10():
    CIFAR_DIR = os.path.join(BASE_DIR, 'data/cifar10/cifar-10-batches-py')
    train_X, train_Y = [], []
    for i in range(1, 6):
        with open(os.path.join(CIFAR_DIR, f'data_batch_{i}'), 'rb') as f:
            d = pickle.load(f, encoding='bytes')
        train_X.append(d[b'data']); train_Y.extend(d[b'labels'])
    X = np.concatenate(train_X).astype(np.float32).reshape(-1, 3, 32, 32) / 255.0
    Y = np.array(train_Y, np.int64)
    # 只用数字对应的类 (0-9都是数字类别，CIFAR没有，但我们可以随机取10类或用全部)
    # 用前5000样本的子集
    idx = np.random.permutation(len(X))[:5000]
    return X[idx].reshape(len(idx), -1), Y[idx]


def load_fsdd(max_len=8000, n_mels=32):
    import scipy.io.wavfile as wav
    FSDD_DIR = os.path.join(BASE_DIR, 'data/fsdd/free-spoken-digit-dataset-master/recordings')
    files = sorted(glob.glob(os.path.join(FSDD_DIR, '*.wav')))
    specs, labels = [], []
    for fp in files:
        digit = int(os.path.basename(fp).split('_')[0])
        rate, data = wav.read(fp)
        data = data.astype(np.float32) / 32768.0
        if len(data) < max_len:
            data = np.pad(data, (0, max_len - len(data)))
        else:
            data = data[:max_len]
        win_len, hop, n_fft = 256, 128, 256
        n_frames = 1 + (max_len - win_len) // hop
        stft = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float32)
        for j in range(n_frames):
            frame = data[j*hop:j*hop+win_len] * np.hanning(win_len)
            stft[:, j] = np.abs(np.fft.rfft(frame, n_fft))
        mel_spec = np.zeros((n_mels, n_frames), dtype=np.float32)
        freqs = np.linspace(0, rate/2, n_fft//2+1)
        mel_f = 2595 * np.log10(1 + freqs/700)
        mel_edges = np.linspace(mel_f[0], mel_f[-1], n_mels+1)
        hz_edges = 700 * (10**(mel_edges/2595) - 1)
        for m in range(n_mels):
            lo, hi = np.searchsorted(freqs, hz_edges[m]), np.searchsorted(freqs, hz_edges[m+1])
            if hi > lo: mel_spec[m] = stft[lo:hi].mean(0)
        specs.append(np.log1p(mel_spec * 1000))
        labels.append(digit)
    specs = np.array(specs, np.float32)
    labels = np.array(labels, np.int64)
    specs = (specs - specs.mean()) / (specs.std() + 1e-8)
    return specs.reshape(len(specs), -1), labels


def make_pairs(vis_X, vis_Y, aud_X, aud_Y, n):
    vis, aud, lbl = [], [], []
    for d in range(10):
        mi = np.where(vis_Y == d)[0]
        fi = np.where(aud_Y == d)[0]
        if len(mi) == 0 or len(fi) == 0:
            continue
        for j in range(n // 10):
            vis.append(vis_X[mi[j % len(mi)]])
            aud.append(aud_X[fi[j % len(fi)]])
            lbl.append(d)
    vis = np.array(vis, np.float32)
    aud = np.array(aud, np.float32)
    lbl = np.array(lbl, np.int64)
    p = np.random.permutation(len(vis))
    return vis[p], aud[p], lbl[p]


# ============================================================
# Encoders
# ============================================================

class Encoder(nn.Module):
    def __init__(self, d_in, d_shared=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, d_shared),
        )
    def forward(self, x):
        return self.net(x)


def info_nce_loss(z1, z2, temperature=0.07):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = z1 @ z2.T / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


# ============================================================
# Metrics (measured at each epoch)
# ============================================================

def measure_metrics(emb_vis, emb_aud, labels):
    """测量 δ_intra, δ_inter, ratio, 检索准确率."""
    n = len(labels)
    n_sample = min(300, n)

    # 跨模态距离采样
    same_dists = []
    diff_dists = []
    idx = np.random.choice(n, n_sample, replace=False)
    for i in idx:
        for j in idx:
            if i == j: continue
            d = np.linalg.norm(emb_vis[i] - emb_aud[j])
            if labels[i] == labels[j]:
                same_dists.append(d)
            else:
                diff_dists.append(d)

    if not same_dists or not diff_dists:
        return None

    delta_intra = np.mean(same_dists)
    delta_inter = np.mean(diff_dists)
    ratio = delta_inter / delta_intra if delta_intra > 0 else 0

    # σ_out / σ_in (类可分性)
    classes = np.unique(labels)
    intra_spreads = []
    inter_dists_list = []
    centers_v = []
    centers_a = []
    for c in classes:
        mask = labels == c
        cv = emb_vis[mask].mean(0)
        ca = emb_aud[mask].mean(0)
        centers_v.append(cv)
        centers_a.append(ca)
        intra_spreads.append(np.mean(np.linalg.norm(emb_vis[mask] - cv, axis=1)))
        intra_spreads.append(np.mean(np.linalg.norm(emb_aud[mask] - ca, axis=1)))
    sigma_in = np.mean(intra_spreads)
    centers = np.array(centers_v)
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            inter_dists_list.append(np.linalg.norm(centers[i] - centers[j]))
    sigma_out = np.min(inter_dists_list) if inter_dists_list else 0
    sep_ratio = sigma_out / sigma_in if sigma_in > 0 else 0

    # 余弦检索准确率
    emb_vis_n = emb_vis / (np.linalg.norm(emb_vis, axis=1, keepdims=True) + 1e-10)
    emb_aud_n = emb_aud / (np.linalg.norm(emb_aud, axis=1, keepdims=True) + 1e-10)
    n_query = min(200, n)
    q_idx = np.random.choice(n, n_query, replace=False)
    cos_correct = 0
    for i in q_idx:
        sims = emb_aud_n @ emb_vis_n[i]
        nearest = sims.argmax()
        if labels[nearest] == labels[i]:
            cos_correct += 1
    cos_acc = cos_correct / n_query

    return {
        'delta_intra': delta_intra,
        'delta_inter': delta_inter,
        'ratio': ratio,
        'sigma_out': sigma_out,
        'sigma_in': sigma_in,
        'sep_ratio': sep_ratio,
        'cos_acc': cos_acc,
    }


# ============================================================
# Main Experiment
# ============================================================

def run_experiment(name, vis_X, vis_Y, aud_X, aud_Y, n_train=2000, n_test=500, epochs=30):
    """训练并对每个epoch测量指标."""

    print(f"\n{'='*70}")
    print(f"实验: {name}")
    print(f"{'='*70}")

    vis_tr, aud_tr, lbl_tr = make_pairs(vis_X, vis_Y, aud_X, aud_Y, n_train)
    vis_te, aud_te, lbl_te = make_pairs(vis_X, vis_Y, aud_X, aud_Y, n_test)

    d_vis = vis_tr.shape[1]
    d_aud = aud_tr.shape[1]
    d_shared = 64

    enc_vis = Encoder(d_vis, d_shared).to(device)
    enc_aud = Encoder(d_aud, d_shared).to(device)
    cls_vis = nn.Linear(d_shared, 10).to(device)
    cls_aud = nn.Linear(d_shared, 10).to(device)

    params = list(enc_vis.parameters()) + list(enc_aud.parameters()) + \
             list(cls_vis.parameters()) + list(cls_aud.parameters())
    opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-5)

    train_ld = DataLoader(
        TensorDataset(torch.tensor(vis_tr), torch.tensor(aud_tr), torch.tensor(lbl_tr)),
        batch_size=128, shuffle=True)

    records = []

    for ep in range(epochs):
        # Train
        enc_vis.train(); enc_aud.train()
        for vb, ab, lb in train_ld:
            vb, ab, lb = vb.to(device), ab.to(device), lb.to(device)
            zv = enc_vis(vb); za = enc_aud(ab)
            l_con = info_nce_loss(zv, za)
            l_ce = F.cross_entropy(cls_vis(zv), lb) + F.cross_entropy(cls_aud(za), lb)
            loss = l_con + 0.5 * l_ce
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()

        # Measure
        enc_vis.eval(); enc_aud.eval()
        with torch.no_grad():
            ev = enc_vis(torch.tensor(vis_te).to(device)).cpu().numpy()
            ea = enc_aud(torch.tensor(aud_te).to(device)).cpu().numpy()

        m = measure_metrics(ev, ea, lbl_te)
        if m:
            m['epoch'] = ep + 1
            records.append(m)
            if (ep + 1) % 5 == 0:
                print(f"  Ep {ep+1:2d}: ratio={m['ratio']:.3f}, "
                      f"sep={m['sep_ratio']:.3f}, cos_acc={m['cos_acc']:.3f}")

    return records


def main():
    print("=" * 70)
    print("PRISM v13 — 寻找第一条定量规律")
    print("假设: cos_acc = f(ratio), 形状对所有数据集相同")
    print("=" * 70)

    np.random.seed(42)
    torch.manual_seed(42)

    # --- Experiment 1: MNIST + FSDD ---
    print("\n[1] 加载 MNIST + FSDD...")
    mn_X, mn_Y, mn_Xt, mn_Yt = load_mnist()
    fsdd_X, fsdd_Y = load_fsdd()
    print(f"  MNIST: {mn_X.shape}, FSDD: {fsdd_X.shape}")

    records_mnist = run_experiment("MNIST + FSDD", mn_X, mn_Y, fsdd_X, fsdd_Y,
                                   n_train=2000, n_test=500, epochs=30)

    # --- Experiment 2: CIFAR-10 + FSDD ---
    # CIFAR-10没有数字0-9的对应，我们用随机投影把CIFAR映射到低维
    # 然后用前10类(正好是10类)
    print("\n[2] 加载 CIFAR-10 + FSDD...")
    cifar_X, cifar_Y = load_cifar10()
    # 随机投影降维
    rp = np.random.randn(cifar_X.shape[1], 256).astype(np.float32) / np.sqrt(cifar_X.shape[1])
    cifar_rp = cifar_X @ rp
    cifar_rp = (cifar_rp - cifar_rp.mean(0)) / (cifar_rp.std(0) + 1e-8)
    print(f"  CIFAR-10: {cifar_rp.shape}")

    records_cifar = run_experiment("CIFAR-10 + FSDD", cifar_rp, cifar_Y, fsdd_X, fsdd_Y,
                                   n_train=2000, n_test=500, epochs=30)

    # --- Experiment 3: MNIST + FSDD with different hyperparameters ---
    # 用更大的temperature (更弱的对齐)
    print("\n[3] MNIST + FSDD (weak alignment, temperature=0.5)...")
    # 这个在run_experiment里没法改，单独写
    vis_tr, aud_tr, lbl_tr = make_pairs(mn_X, mn_Y, fsdd_X, fsdd_Y, 2000)
    vis_te, aud_te, lbl_te = make_pairs(mn_X, mn_Y, fsdd_X, fsdd_Y, 500)

    enc_vis3 = Encoder(mn_X.shape[1], 64).to(device)
    enc_aud3 = Encoder(fsdd_X.shape[1], 64).to(device)
    cls_vis3 = nn.Linear(64, 10).to(device)
    cls_aud3 = nn.Linear(64, 10).to(device)
    params3 = list(enc_vis3.parameters()) + list(enc_aud3.parameters()) + \
              list(cls_vis3.parameters()) + list(cls_aud3.parameters())
    opt3 = torch.optim.Adam(params3, lr=1e-3)
    train_ld3 = DataLoader(
        TensorDataset(torch.tensor(vis_tr), torch.tensor(aud_tr), torch.tensor(lbl_tr)),
        batch_size=128, shuffle=True)

    records_weak = []
    for ep in range(30):
        enc_vis3.train(); enc_aud3.train()
        for vb, ab, lb in train_ld3:
            vb, ab, lb = vb.to(device), ab.to(device), lb.to(device)
            zv = enc_vis3(vb); za = enc_aud3(ab)
            l_con = info_nce_loss(zv, za, temperature=0.5)  # 弱对齐
            l_ce = F.cross_entropy(cls_vis3(zv), lb) + F.cross_entropy(cls_aud3(za), lb)
            loss = l_con + 0.5 * l_ce
            opt3.zero_grad(); loss.backward(); opt3.step()
        enc_vis3.eval(); enc_aud3.eval()
        with torch.no_grad():
            ev = enc_vis3(torch.tensor(vis_te).to(device)).cpu().numpy()
            ea = enc_aud3(torch.tensor(aud_te).to(device)).cpu().numpy()
        m = measure_metrics(ev, ea, lbl_te)
        if m:
            m['epoch'] = ep + 1
            records_weak.append(m)
            if (ep + 1) % 5 == 0:
                print(f"  Ep {ep+1:2d}: ratio={m['ratio']:.3f}, cos_acc={m['cos_acc']:.3f}")

    # --- Analysis ---
    print("\n" + "=" * 70)
    print("分析: cos_acc vs ratio 的普适性")
    print("=" * 70)

    def print_records(name, records):
        print(f"\n  [{name}]")
        print(f"  {'Epoch':>5} {'ratio':>8} {'sep':>8} {'cos_acc':>8}")
        for r in records:
            print(f"  {r['epoch']:5d} {r['ratio']:8.3f} {r['sep_ratio']:8.3f} {r['cos_acc']:8.3f}")

    print_records("MNIST + FSDD", records_mnist)
    print_records("CIFAR-10 + FSDD", records_cifar)
    print_records("MNIST + FSDD (weak)", records_weak)

    # 拟合: cos_acc = sigmoid(a * (ratio - b))
    from scipy.optimize import curve_fit
    def sigmoid(x, a, b):
        return 1.0 / (1.0 + np.exp(-a * (x - b)))

    def fit_and_report(name, records):
        ratios = np.array([r['ratio'] for r in records])
        accs = np.array([r['cos_acc'] for r in records])
        try:
            popt, _ = curve_fit(sigmoid, ratios, accs, p0=[5, 1.0], maxfev=5000)
            a, b = popt
            preds = sigmoid(ratios, a, b)
            residuals = np.mean((preds - accs)**2)
            print(f"\n  [{name}] sigmoid拟合: cos_acc = σ({a:.2f}·(ratio - {b:.2f}))")
            print(f"    拟合MSE: {residuals:.6f}")
            return a, b
        except Exception as e:
            print(f"\n  [{name}] 拟合失败: {e}")
            return None, None

    print("\n" + "-" * 50)
    a1, b1 = fit_and_report("MNIST + FSDD", records_mnist)
    a2, b2 = fit_and_report("CIFAR-10 + FSDD", records_cifar)
    a3, b3 = fit_and_report("MNIST + FSDD (weak)", records_weak)

    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)

    if a1 and a2 and a3:
        params = [(a1, b1), (a2, b2), (a3, b3)]
        a_mean = np.mean([p[0] for p in params])
        b_mean = np.mean([p[1] for p in params])
        a_spread = np.std([p[0] for p in params])
        b_spread = np.std([p[1] for p in params])

        print(f"\n  Sigmoid参数分布:")
        print(f"    a (slope): mean={a_mean:.2f}, std={a_spread:.2f}")
        print(f"    b (center): mean={b_mean:.2f}, std={b_spread:.2f}")

        if a_spread / (a_mean + 1e-10) < 0.3 and b_spread / (b_mean + 1e-10) < 0.3:
            print(f"\n  ✓ 参数跨数据集稳定 (变异系数 < 30%)")
            print(f"    普适公式: cos_acc ≈ σ({a_mean:.1f}·(ratio - {b_mean:.1f}))")
            print(f"    这是一条定量规律!")
        else:
            print(f"\n  ✗ 参数跨数据集不稳定 (变异系数 > 30%)")
            print(f"    函数形状依赖数据集, 不是普适规律")
            print(f"    但函数形式(sigmoid)可能仍然普适, 只是参数不同")


if __name__ == '__main__':
    main()
