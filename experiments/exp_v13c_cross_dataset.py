"""
PRISM v13c — 跨数据集验证普适规律
==================================

已发现: cos_acc = σ((c/T + d) · (ratio - b))
  MNIST+FSDD: a = 1.32/T + 1.64, b = 1.06, R²=0.99

现在验证:
  1. CIFAR-10 + FSDD (不同视觉域)
  2. MNIST + FSDD, 不同共享维度 (d=32, 128)
  3. MNIST + FSDD, 不同类别数 (5类, 10类)

如果 c, d, b 在跨数据集后仍然稳定 → 普适规律确认
如果不稳定 → 至少知道函数形式普适但参数依赖数据
"""

import sys, os, struct, gzip, glob, pickle
sys.path.insert(0, '/Users/wp/ZCodeProject/PRISM')

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

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
    return X, Y


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


def load_cifar10_subset(n=3000):
    CIFAR_DIR = os.path.join(BASE_DIR, 'data/cifar10/cifar-10-batches-py')
    train_X, train_Y = [], []
    for i in range(1, 6):
        with open(os.path.join(CIFAR_DIR, f'data_batch_{i}'), 'rb') as f:
            d = pickle.load(f, encoding='bytes')
        train_X.append(d[b'data']); train_Y.extend(d[b'labels'])
    X = np.concatenate(train_X).astype(np.float32).reshape(-1, 3072) / 255.0
    Y = np.array(train_Y, np.int64)
    # 随机投影到256维
    rp = np.random.randn(3072, 256).astype(np.float32) / np.sqrt(3072)
    X_rp = X @ rp
    X_rp = (X_rp - X_rp.mean(0)) / (X_rp.std(0) + 1e-8)
    idx = np.random.permutation(len(X_rp))[:n]
    return X_rp[idx], Y[idx]


def make_pairs(vis_X, vis_Y, aud_X, aud_Y, n, n_classes=10):
    vis, aud, lbl = [], [], []
    for d in range(n_classes):
        mi = np.where(vis_Y == d)[0]
        fi = np.where(aud_Y == d)[0]
        if len(mi) == 0 or len(fi) == 0:
            continue
        for j in range(n // n_classes):
            vis.append(vis_X[mi[j % len(mi)]])
            aud.append(aud_X[fi[j % len(fi)]])
            lbl.append(d)
    if not vis:
        return np.array([]), np.array([]), np.array([])
    vis = np.array(vis, np.float32)
    aud = np.array(aud, np.float32)
    lbl = np.array(lbl, np.int64)
    p = np.random.permutation(len(vis))
    return vis[p], aud[p], lbl[p]


# ============================================================
# Encoder + Training + Metrics
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


def measure_metrics(emb_vis, emb_aud, labels):
    n = len(labels)
    n_sample = min(300, n)
    same_dists, diff_dists = [], []
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

    emb_vis_n = emb_vis / (np.linalg.norm(emb_vis, axis=1, keepdims=True) + 1e-10)
    emb_aud_n = emb_aud / (np.linalg.norm(emb_aud, axis=1, keepdims=True) + 1e-10)
    n_query = min(200, n)
    q_idx = np.random.choice(n, n_query, replace=False)
    cos_correct = sum(1 for i in q_idx if labels[(emb_aud_n @ emb_vis_n[i]).argmax()] == labels[i])
    return {'ratio': ratio, 'cos_acc': cos_correct / n_query}


def sigmoid(x, a, b):
    return 1.0 / (1.0 + np.exp(-a * (x - b)))


def run_single(temperature, vis_tr, aud_tr, lbl_tr, vis_te, aud_te, lbl_te,
               d_shared=64, n_classes=10, epochs=30):
    enc_vis = Encoder(vis_tr.shape[1], d_shared).to(device)
    enc_aud = Encoder(aud_tr.shape[1], d_shared).to(device)
    cls_vis = nn.Linear(d_shared, n_classes).to(device)
    cls_aud = nn.Linear(d_shared, n_classes).to(device)
    params = list(enc_vis.parameters()) + list(enc_aud.parameters()) + \
             list(cls_vis.parameters()) + list(cls_aud.parameters())
    opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-5)
    train_ld = DataLoader(
        TensorDataset(torch.tensor(vis_tr), torch.tensor(aud_tr), torch.tensor(lbl_tr)),
        batch_size=128, shuffle=True)

    records = []
    for ep in range(epochs):
        enc_vis.train(); enc_aud.train()
        for vb, ab, lb in train_ld:
            vb, ab, lb = vb.to(device), ab.to(device), lb.to(device)
            zv = enc_vis(vb); za = enc_aud(ab)
            l_con = info_nce_loss(zv, za, temperature=temperature)
            l_ce = F.cross_entropy(cls_vis(zv), lb) + F.cross_entropy(cls_aud(za), lb)
            loss = l_con + 0.5 * l_ce
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        enc_vis.eval(); enc_aud.eval()
        with torch.no_grad():
            ev = enc_vis(torch.tensor(vis_te).to(device)).cpu().numpy()
            ea = enc_aud(torch.tensor(aud_te).to(device)).cpu().numpy()
        m = measure_metrics(ev, ea, lbl_te)
        if m:
            m['epoch'] = ep + 1
            records.append(m)

    ratios = np.array([r['ratio'] for r in records])
    accs = np.array([r['cos_acc'] for r in records])
    try:
        popt, _ = curve_fit(sigmoid, ratios, accs, p0=[5, 1.0], maxfev=5000)
        a, b = popt
        preds = sigmoid(ratios, a, b)
        mse = np.mean((preds - accs)**2)
        return {'a': a, 'b': b, 'mse': mse, 'n_points': len(records)}
    except:
        return {'a': None, 'b': None, 'mse': None, 'n_points': len(records)}


def run_condition(name, vis_X, vis_Y, aud_X, aud_Y, temperatures, n_train=2000, n_test=500,
                  d_shared=64, n_classes=10, epochs=30):
    """在一个条件下跑多个temperature."""
    print(f"\n{'='*60}")
    print(f"  条件: {name}")
    print(f"{'='*60}")

    results = {}
    for temp in temperatures:
        np.random.seed(42)
        torch.manual_seed(42)

        vis_tr, aud_tr, lbl_tr = make_pairs(vis_X, vis_Y, aud_X, aud_Y, n_train, n_classes)
        vis_te, aud_te, lbl_te = make_pairs(vis_X, vis_Y, aud_X, aud_Y, n_test, n_classes)

        if len(vis_tr) == 0:
            print(f"    T={temp}: 数据不足, 跳过")
            continue

        r = run_single(temp, vis_tr, aud_tr, lbl_tr, vis_te, aud_te, lbl_te,
                       d_shared, n_classes, epochs)
        results[temp] = r
        if r['a'] is not None:
            print(f"    T={temp:.3f}: a={r['a']:.2f}, b={r['b']:.3f}, MSE={r['mse']:.5f}")
        else:
            print(f"    T={temp:.3f}: 拟合失败")

    # 拟合 a = c/T + d
    valid = [(t, r) for t, r in results.items() if r['a'] is not None]
    if len(valid) >= 3:
        temps = np.array([v[0] for v in valid])
        a_vals = np.array([v[1]['a'] for v in valid])
        b_vals = np.array([v[1]['b'] for v in valid])

        def linear(x, c, d):
            return c * x + d
        try:
            popt, _ = curve_fit(linear, 1.0/temps, a_vals)
            c_fit, d_fit = popt
            preds = linear(1.0/temps, c_fit, d_fit)
            r2 = 1 - np.sum((a_vals - preds)**2) / np.sum((a_vals - a_vals.mean())**2)
            b_mean = b_vals.mean()
            b_std = b_vals.std()
            b_cv = b_std / b_mean * 100
            corr, _ = pearsonr(1.0/temps, a_vals)

            print(f"\n  拟合结果:")
            print(f"    a = {c_fit:.2f}/T + {d_fit:.2f}  (R²={r2:.4f})")
            print(f"    b = {b_mean:.3f} ± {b_std:.3f}  (CV={b_cv:.1f}%)")
            print(f"    Pearson(a, 1/T) = {corr:.3f}")

            return {'c': c_fit, 'd': d_fit, 'r2': r2, 'b_mean': b_mean,
                    'b_std': b_std, 'b_cv': b_cv, 'corr': corr}
        except:
            pass
    return None


def main():
    print("=" * 70)
    print("PRISM v13c — 跨数据集验证普适规律")
    print("=" * 70)

    np.random.seed(42)
    torch.manual_seed(42)

    mn_X, mn_Y = load_mnist()
    fsdd_X, fsdd_Y = load_fsdd()
    cifar_X, cifar_Y = load_cifar10_subset(3000)

    temperatures = [0.05, 0.1, 0.2, 0.5, 1.0]

    # --- Condition 1: MNIST + FSDD (baseline, 重复确认) ---
    r1 = run_condition("MNIST + FSDD (baseline)", mn_X, mn_Y, fsdd_X, fsdd_Y,
                       temperatures, n_train=2000, n_test=500)

    # --- Condition 2: CIFAR-10 + FSDD ---
    r2 = run_condition("CIFAR-10 + FSDD", cifar_X, cifar_Y, fsdd_X, fsdd_Y,
                       temperatures, n_train=2000, n_test=500)

    # --- Condition 3: MNIST + FSDD, d_shared=32 ---
    r3 = run_condition("MNIST + FSDD, d=32", mn_X, mn_Y, fsdd_X, fsdd_Y,
                       temperatures, n_train=2000, n_test=500, d_shared=32)

    # --- Condition 4: MNIST + FSDD, d_shared=128 ---
    r4 = run_condition("MNIST + FSDD, d=128", mn_X, mn_Y, fsdd_X, fsdd_Y,
                       temperatures, n_train=2000, n_test=500, d_shared=128)

    # --- Condition 5: MNIST + FSDD, 5 classes only ---
    mask5v = mn_Y < 5
    mask5a = fsdd_Y < 5
    r5 = run_condition("MNIST + FSDD, 5 classes", mn_X[mask5v], mn_Y[mask5v],
                       fsdd_X[mask5a], fsdd_Y[mask5a],
                       temperatures, n_train=2000, n_test=500, n_classes=5)

    # --- Meta-analysis ---
    print("\n" + "=" * 70)
    print("跨条件对比")
    print("=" * 70)

    all_results = [
        ("MNIST+FSDD", r1),
        ("CIFAR-10+FSDD", r2),
        ("d=32", r3),
        ("d=128", r4),
        ("5 classes", r5),
    ]

    print(f"\n  {'条件':>20} {'c':>8} {'d':>8} {'R²':>8} {'b_mean':>8} {'b_CV%':>8} {'corr':>8}")
    print(f"  {'-'*70}")

    valid_c = []
    valid_d = []
    valid_b = []
    for name, r in all_results:
        if r:
            print(f"  {name:>20} {r['c']:>8.2f} {r['d']:>8.2f} {r['r2']:>8.4f} "
                  f"{r['b_mean']:>8.3f} {r['b_cv']:>8.1f} {r['corr']:>8.3f}")
            valid_c.append(r['c'])
            valid_d.append(r['d'])
            valid_b.append(r['b_mean'])
        else:
            print(f"  {name:>20}    (拟合失败)")

    if len(valid_c) >= 3:
        c_arr = np.array(valid_c)
        d_arr = np.array(valid_d)
        b_arr = np.array(valid_b)

        print(f"\n  --- 参数跨条件稳定性 ---")
        print(f"  c (1/T 系数): mean={c_arr.mean():.2f}, std={c_arr.std():.2f}, "
              f"CV={c_arr.std()/c_arr.mean()*100:.1f}%")
        print(f"  d (截距):     mean={d_arr.mean():.2f}, std={d_arr.std():.2f}, "
              f"CV={d_arr.std()/d_arr.mean()*100:.1f}%")
        print(f"  b (转折点):   mean={b_arr.mean():.3f}, std={b_arr.std():.3f}, "
              f"CV={b_arr.std()/b_arr.mean()*100:.1f}%")

        # 最终判断
        c_stable = c_arr.std()/c_arr.mean() < 0.3
        b_stable = b_arr.std()/b_arr.mean() < 0.1

        print(f"\n  === 最终结论 ===")
        if b_stable:
            print(f"  ✓ b ≈ {b_arr.mean():.2f} 在所有条件下稳定 (CV < 10%)")
            print(f"    这是数学必然: ratio=1 → 随机检索 → acc=50%")
        else:
            print(f"  ✗ b 不稳定 (CV > 10%)")

        if c_stable:
            print(f"  ✓ c ≈ {c_arr.mean():.1f} 在所有条件下稳定 (CV < 30%)")
            print(f"    普适公式: cos_acc = σ(({c_arr.mean():.1f}/T + {d_arr.mean():.1f}) · (ratio - {b_arr.mean():.2f}))")
        else:
            print(f"  ≈ c 不完全稳定，但函数形式普适")
            print(f"    a = f(T) 的线性形式成立，但斜率依赖数据")


if __name__ == '__main__':
    main()
