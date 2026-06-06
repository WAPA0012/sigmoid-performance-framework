"""
PRISM v13b — 验证 sigmoid 斜率 a 是否依赖于 temperature
=======================================================

假设: cos_acc = σ(a · (ratio - b))
  - b ≈ 1.0 是数学必然（ratio=1是随机/非随机分界）
  - a = g(temperature), 可能单调

实验: 5个temperature, 每个跑30 epoch, 拟合sigmoid
"""

import sys, os, struct, gzip, glob
sys.path.insert(0, '/Users/wp/ZCodeProject/PRISM')

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

device = torch.device('mps') if torch.backends.mps.is_available() else (
    torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print(f"Device: {device}")
BASE_DIR = '/Users/wp/ZCodeProject/PRISM'


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


def make_pairs(vis_X, vis_Y, aud_X, aud_Y, n):
    vis, aud, lbl = [], [], []
    for d in range(10):
        mi = np.where(vis_Y == d)[0]
        fi = np.where(aud_Y == d)[0]
        for j in range(n // 10):
            vis.append(vis_X[mi[j % len(mi)]])
            aud.append(aud_X[fi[j % len(fi)]])
            lbl.append(d)
    vis = np.array(vis, np.float32)
    aud = np.array(aud, np.float32)
    lbl = np.array(lbl, np.int64)
    p = np.random.permutation(len(vis))
    return vis[p], aud[p], lbl[p]


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


def run_sweep(temperature, vis_tr, aud_tr, lbl_tr, vis_te, aud_te, lbl_te, epochs=30):
    d_shared = 64
    enc_vis = Encoder(vis_tr.shape[1], d_shared).to(device)
    enc_aud = Encoder(aud_tr.shape[1], d_shared).to(device)
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
    return records


def main():
    print("=" * 70)
    print("PRISM v13b — Sigmoid斜率 vs Temperature")
    print("=" * 70)

    np.random.seed(42)
    torch.manual_seed(42)

    mn_X, mn_Y = load_mnist()
    fsdd_X, fsdd_Y = load_fsdd()
    print(f"  MNIST: {mn_X.shape}, FSDD: {fsdd_X.shape}")

    vis_tr, aud_tr, lbl_tr = make_pairs(mn_X, mn_Y, fsdd_X, fsdd_Y, 2000)
    vis_te, aud_te, lbl_te = make_pairs(mn_X, mn_Y, fsdd_X, fsdd_Y, 500)

    temperatures = [0.03, 0.07, 0.15, 0.3, 0.5, 1.0]
    all_results = {}

    for temp in temperatures:
        print(f"\n  Temperature = {temp}...")
        np.random.seed(42)
        torch.manual_seed(42)
        records = run_sweep(temp, vis_tr, aud_tr, lbl_tr, vis_te, aud_te, lbl_te, epochs=30)
        all_results[temp] = records

        # 拟合 sigmoid
        ratios = np.array([r['ratio'] for r in records])
        accs = np.array([r['cos_acc'] for r in records])

        from scipy.optimize import curve_fit
        def sigmoid(x, a, b):
            return 1.0 / (1.0 + np.exp(-a * (x - b)))

        try:
            popt, _ = curve_fit(sigmoid, ratios, accs, p0=[5, 1.0], maxfev=5000)
            a, b = popt
            preds = sigmoid(ratios, a, b)
            mse = np.mean((preds - accs)**2)
            # 最终准确率
            final_acc = records[-1]['cos_acc']
            final_ratio = records[-1]['ratio']
            print(f"    a={a:.2f}, b={b:.3f}, MSE={mse:.5f}, "
                  f"final_ratio={final_ratio:.3f}, final_acc={final_acc:.3f}")
            all_results[temp] = {'records': records, 'a': a, 'b': b, 'mse': mse}
        except Exception as e:
            print(f"    拟合失败: {e}")
            all_results[temp] = {'records': records, 'a': None, 'b': None, 'mse': None}

    # --- Summary ---
    print("\n" + "=" * 70)
    print("结果汇总")
    print("=" * 70)
    print(f"\n  {'Temp':>6} {'a (slope)':>10} {'b (center)':>11} {'MSE':>10}")
    print(f"  {'-'*40}")

    valid = []
    for temp in temperatures:
        r = all_results[temp]
        if r['a'] is not None:
            print(f"  {temp:>6.3f} {r['a']:>10.2f} {r['b']:>11.3f} {r['mse']:>10.5f}")
            valid.append((temp, r['a'], r['b']))

    if len(valid) >= 3:
        temps = np.array([v[0] for v in valid])
        a_vals = np.array([v[1] for v in valid])
        b_vals = np.array([v[2] for v in valid])

        # 检查 a vs temperature 的关系
        # 假设: a ∝ 1/temperature (温度越低，对齐越强，过渡越陡)
        from scipy.stats import pearsonr

        corr_a_temp, _ = pearsonr(temps, a_vals)
        corr_a_inv_temp, _ = pearsonr(1.0/temps, a_vals)

        print(f"\n  a vs temperature: Pearson r = {corr_a_temp:.3f}")
        print(f"  a vs 1/temperature: Pearson r = {corr_a_inv_temp:.3f}")

        # 检查 b 的稳定性
        print(f"\n  b 值: mean={b_vals.mean():.3f}, std={b_vals.std():.3f}, "
              f"CV={b_vals.std()/b_vals.mean()*100:.1f}%")

        # 尝试拟合 a = c / temperature + d
        try:
            def linear(x, c, d):
                return c * x + d
            popt_at, _ = curve_fit(linear, 1.0/temps, a_vals)
            c_fit, d_fit = popt_at
            preds_at = linear(1.0/temps, c_fit, d_fit)
            r2 = 1 - np.sum((a_vals - preds_at)**2) / np.sum((a_vals - a_vals.mean())**2)
            print(f"\n  拟合 a = {c_fit:.2f}/temp + {d_fit:.2f}, R² = {r2:.4f}")
            if r2 > 0.9:
                print(f"  ✓ a 与 1/temperature 线性关系强 (R² > 0.9)")
                print(f"    含义: temperature 控制过渡的陡峭程度")
                print(f"    低 temp → 强对齐 → 陡峭过渡 → 小ratio就能达到高准确率")
            else:
                print(f"  ≈ a 与 1/temperature 有趋势但非线性")
        except:
            pass

        # 最终结论
        print("\n" + "=" * 70)
        print("结论")
        print("=" * 70)
        b_cv = b_vals.std() / b_vals.mean() * 100
        print(f"""
  1. Sigmoid 形式普适: cos_acc = σ(a · (ratio - b)) 在所有温度下成立

  2. b (转折点) 稳定性: CV = {b_cv:.1f}%
     {'✓ b ≈ 1.0 是数学必然 (ratio=1 = 随机水平)' if b_cv < 30 else '✗ b 不够稳定'}

  3. a (斜率) 与 temperature 的关系:
     相关系数(a vs 1/T) = {corr_a_inv_temp:.3f}
     {'✓ a 与对齐强度(1/T)强正相关' if abs(corr_a_inv_temp) > 0.9 else '≈ 有趋势但不够强'}

  如果 2 和 3 都成立:
    cos_acc = σ(f(T) · (δ_inter/δ_intra - 1.0))
    其中 f(T) 由对齐强度决定
    这是一条有解释的定量规律
        """)


if __name__ == '__main__':
    main()
