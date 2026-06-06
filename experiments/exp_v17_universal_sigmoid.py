"""
PRISM v17 — MI→Performance Sigmoid定律的普适性测试
==================================================

核心假说：sigmoid MI→performance 不是跨模态检索独有的，
而是AI系统中信息→能力的通用映射。

具体测试：
  A. 单模态分类：I(X; Y) vs 分类准确率
     - 用信息瓶颈控制I(X;Y)，看acc是否是I的sigmoid函数
     - 如果成立，说明这不是跨模态的特殊性质

  B. 不同任务类型：分类、检索、生成
     - 分类：标准分类器，加噪声控制I
     - 检索：已有的sigmoid结果
     - 生成：I(X;Z) vs 重建质量（如果有时间）

  C. 涌现模拟：多个子任务，I超过不同阈值时子任务被"解锁"
     - 如果涌现也是sigmoid MI→acc，那这就是统一框架的起点

预测：
  所有任务的 performance = σ(α_task · I + β_task)
  不同任务有不同的 α, β，但sigmoid形式相同
  涌现 = 某个子任务的I*阈值被越过
"""

import sys, os
sys.path.insert(0, '/Users/wp/ZCodeProject/PRISM')

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import struct, gzip, glob, pickle, time

device = torch.device('mps') if torch.backends.mps.is_available() else (
    torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print(f"Device: {device}")
BASE_DIR = '/Users/wp/ZCodeProject/PRISM'


def sigmoid(x, a, b):
    return 1.0 / (1.0 + np.exp(-np.clip(a * (x - b), -500, 500)))


def sigmoid_lin(x, a, b):
    return 1.0 / (1.0 + np.exp(-np.clip(a * x + b, -500, 500)))


# ============================================================
# 数据加载
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


def load_cifar10_flat():
    CIFAR_DIR = os.path.join(BASE_DIR, 'data/cifar10/cifar-10-batches-py')
    train_X, train_Y = [], []
    for i in range(1, 6):
        with open(os.path.join(CIFAR_DIR, f'data_batch_{i}'), 'rb') as f:
            d = pickle.load(f, encoding='bytes')
        train_X.append(d[b'data']); train_Y.extend(d[b'labels'])
    X = np.concatenate(train_X).astype(np.float32) / 255.0
    Y = np.array(train_Y, np.int64)
    rp = np.random.RandomState(42).randn(X.shape[1], 256).astype(np.float32) / np.sqrt(X.shape[1])
    X_rp = X @ rp
    X_rp = (X_rp - X_rp.mean(0)) / (X_rp.std(0) + 1e-8)
    return X_rp, Y


# ============================================================
# Part A: 单模态分类 — I(X;Y) vs acc
# ============================================================
def partA_classification():
    """
    核心实验：在分类任务中，通过信息瓶颈控制I(X;Y)，测量acc。

    方法：训练一个autoencoder bottleneck，通过调节bottleneck宽度w来控制I(X;Z)。
    然后用Z做分类，测量acc。

    更简单的方法：直接在输入上加噪声控制I(X;Y)。
    """
    print("=" * 70)
    print("Part A: 单模态分类 — noise控制I(X;Y)，测acc")
    print("=" * 70)

    # MNIST
    X, Y = load_mnist()
    n = 5000
    np.random.seed(42)
    idx = np.random.choice(len(X), n, replace=False)
    X_sub, Y_sub = X[idx], Y[idx]

    # 训练一个简单的分类器
    d_in = X_sub.shape[1]
    classifier = nn.Sequential(
        nn.Linear(d_in, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 10),
    ).to(device)

    # 训练
    opt = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_sub), torch.tensor(Y_sub)),
        batch_size=128, shuffle=True)
    for ep in range(15):
        classifier.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.cross_entropy(classifier(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()

    # 测试：加不同程度噪声，测量I和acc
    n_test = 2000
    idx_te = np.random.choice(len(X), n_test, replace=False)
    X_te, Y_te = X[idx_te], Y[idx_te]

    # 计算I(X;Y)的方法：
    # I(X;Y) = H(Y) - H(Y|X)
    # 对于离散Y和连续X，用经验方法：I ≈ H(Y) - E_x[H(Y|X)]
    # H(Y|X) ≈ -E[log p(y|x)] 用分类器的softmax输出估计

    noise_levels = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
    results_mnist = []

    print("\n  [MNIST分类]")
    for sigma in noise_levels:
        X_noisy = X_te + np.random.randn(*X_te.shape).astype(np.float32) * sigma
        X_noisy = np.clip(X_noisy, 0, 1)

        classifier.eval()
        with torch.no_grad():
            logits = classifier(torch.tensor(X_noisy).to(device))
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()

        acc = (preds == Y_te).mean()

        # 估计I(X;Y) = H(Y) - H(Y|X)
        # H(Y|X) ≈ -mean(log p(y_i | x_i))
        log_py_x = np.log(probs[np.arange(len(Y_te)), Y_te] + 1e-10)
        H_Y_given_X = -np.mean(log_py_x)

        # H(Y) = -sum p(y) log p(y)
        class_counts = np.bincount(Y_te, minlength=10).astype(float)
        class_probs = class_counts / class_counts.sum()
        H_Y = -np.sum(class_probs * np.log(class_probs + 1e-10))

        I_XY = H_Y - H_Y_given_X
        I_XY = max(I_XY, 0)  # 估计误差可能给出微小负值

        results_mnist.append({'sigma': sigma, 'acc': acc, 'I': I_XY, 'H_Y|X': H_Y_given_X})
        print(f"    σ={sigma:>5.1f}: I(X;Y)={I_XY:>6.3f} bits, acc={acc:.3f}")

    # 拟合 acc = σ(a·(I - b))
    I_vals = np.array([r['I'] for r in results_mnist])
    acc_vals = np.array([r['acc'] for r in results_mnist])

    mask = (acc_vals > 0.02) & (acc_vals < 0.99)
    if mask.sum() >= 4:
        try:
            # 用bits
            popt, _ = curve_fit(sigmoid, I_vals[mask], acc_vals[mask], p0=[5, 0.5], maxfev=5000)
            preds = sigmoid(I_vals[mask], *popt)
            r2 = 1 - np.sum((acc_vals[mask] - preds)**2) / max(np.sum((acc_vals[mask] - acc_vals[mask].mean())**2), 1e-10)
            print(f"\n  MNIST: acc = σ({popt[0]:.2f}·(I - {popt[1]:.3f}))  R²={r2:.4f}")
            print(f"  转折点 I* = {popt[1]:.3f} bits")
        except Exception as e:
            print(f"  MNIST fit failed: {e}")

    # CIFAR-10 重复
    X_c, Y_c = load_cifar10_flat()
    idx_c = np.random.choice(len(X_c), min(5000, len(X_c)), replace=False)
    X_cs, Y_cs = X_c[idx_c], Y_c[idx_c]

    d_in_c = X_cs.shape[1]
    cls_cifar = nn.Sequential(
        nn.Linear(d_in_c, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 10),
    ).to(device)
    opt_c = torch.optim.Adam(cls_cifar.parameters(), lr=1e-3)
    loader_c = DataLoader(
        TensorDataset(torch.tensor(X_cs.astype(np.float32)), torch.tensor(Y_cs.astype(np.int64))),
        batch_size=128, shuffle=True)
    for ep in range(15):
        cls_cifar.train()
        for xb, yb in loader_c:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.cross_entropy(cls_cifar(xb), yb)
            opt_c.zero_grad(); loss.backward(); opt_c.step()

    idx_tc = np.random.choice(len(X_c), 2000, replace=False)
    X_tc, Y_tc = X_c[idx_tc], Y_c[idx_tc]

    results_cifar = []
    print("\n  [CIFAR-10分类]")
    for sigma in noise_levels:
        X_noisy = X_tc + np.random.randn(*X_tc.shape).astype(np.float32) * sigma
        X_noisy = X_noisy.astype(np.float32)
        cls_cifar.eval()
        with torch.no_grad():
            logits = cls_cifar(torch.tensor(X_noisy, dtype=torch.float32).to(device))
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()
        acc = (preds == Y_tc).mean()
        log_py_x = np.log(probs[np.arange(len(Y_tc)), Y_tc] + 1e-10)
        H_Y_given_X = -np.mean(log_py_x)
        class_counts = np.bincount(Y_tc, minlength=10).astype(float)
        class_probs = class_counts / class_counts.sum()
        H_Y = -np.sum(class_probs * np.log(class_probs + 1e-10))
        I_XY = max(H_Y - H_Y_given_X, 0)
        results_cifar.append({'sigma': sigma, 'acc': acc, 'I': I_XY})
        print(f"    σ={sigma:>5.1f}: I(X;Y)={I_XY:>6.3f} bits, acc={acc:.3f}")

    I_c = np.array([r['I'] for r in results_cifar])
    acc_c = np.array([r['acc'] for r in results_cifar])
    mask_c = (acc_c > 0.02) & (acc_c < 0.99)
    if mask_c.sum() >= 4:
        try:
            popt_c, _ = curve_fit(sigmoid, I_c[mask_c], acc_c[mask_c], p0=[5, 0.5], maxfev=5000)
            preds_c = sigmoid(I_c[mask_c], *popt_c)
            r2_c = 1 - np.sum((acc_c[mask_c] - preds_c)**2) / max(np.sum((acc_c[mask_c] - acc_c[mask_c].mean())**2), 1e-10)
            print(f"\n  CIFAR: acc = σ({popt_c[0]:.2f}·(I - {popt_c[1]:.3f}))  R²={r2_c:.4f}")
            print(f"  转折点 I* = {popt_c[1]:.3f} bits")
        except Exception as e:
            print(f"  CIFAR fit failed: {e}")

    return results_mnist, results_cifar


# ============================================================
# Part B: 信息瓶颈 — 显式控制I(X;Z)
# ============================================================
def partB_bottleneck():
    """
    更严格的信息控制：训练autoencoder，bottleneck维度从1到d，
    显式控制I(X;Z) ≈ dim(Z) · H(每维)
    """
    print("\n" + "=" * 70)
    print("Part B: 信息瓶颈 — bottleneck宽度控制I")
    print("=" * 70)

    X, Y = load_mnist()
    n = 5000
    np.random.seed(42)
    idx = np.random.choice(len(X), n, replace=False)
    X_sub, Y_sub = X[idx], Y[idx]

    d_in = X_sub.shape[1]
    bottleneck_widths = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    results = []
    print("\n  [MNIST bottleneck分类]")

    for w in bottleneck_widths:
        # Autoencoder + 分类头
        encoder = nn.Sequential(
            nn.Linear(d_in, 256), nn.ReLU(),
            nn.Linear(256, w),
        ).to(device)
        decoder = nn.Sequential(
            nn.Linear(w, 256), nn.ReLU(),
            nn.Linear(256, d_in),
        ).to(device)
        cls_head = nn.Linear(w, 10).to(device)

        params = list(encoder.parameters()) + list(decoder.parameters()) + list(cls_head.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)
        loader = DataLoader(
            TensorDataset(torch.tensor(X_sub), torch.tensor(Y_sub)),
            batch_size=128, shuffle=True)

        for ep in range(20):
            encoder.train(); decoder.train(); cls_head.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                z = encoder(xb)
                x_rec = decoder(z)
                l_rec = F.mse_loss(x_rec, xb)
                l_cls = F.cross_entropy(cls_head(z), yb)
                loss = l_rec + 0.5 * l_cls
                opt.zero_grad(); loss.backward(); opt.step()

        # 评估
        encoder.eval(); cls_head.eval()
        with torch.no_grad():
            Z = encoder(torch.tensor(X_sub).to(device)).cpu().numpy()
            logits = cls_head(torch.tensor(Z).to(device))
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()

        acc = (preds == Y_sub).mean()

        # 估计I(X;Z)
        # 简化：用Z的方差作为I的代理
        # I(X;Z) ≈ 0.5 * sum_dim log(1 + var_z_dim / sigma_noise²)
        # 对bottleneck，假设sigma_noise = 1
        z_var = np.var(Z, axis=0)
        I_est = 0.5 * np.sum(np.log(1 + z_var))

        # 也用分类器输出估计I(X;Y|Z→Y)
        log_py_x = np.log(probs[np.arange(len(Y_sub)), Y_sub] + 1e-10)
        H_Y_given_X = -np.mean(log_py_x)
        class_counts = np.bincount(Y_sub, minlength=10).astype(float)
        H_Y = -np.sum((class_counts / class_counts.sum()) * np.log(class_counts / class_counts.sum() + 1e-10))
        I_XY_through_Z = max(H_Y - H_Y_given_X, 0)

        results.append({
            'width': w, 'acc': acc, 'I_bottleneck': I_est,
            'I_XY_through_Z': I_XY_through_Z,
        })
        print(f"    w={w:>3d}: I(Z)={I_est:>6.2f} nats, I(X→Y)={I_XY_through_Z:.3f} bits, acc={acc:.3f}")

    # 拟合
    I_vals = np.array([r['I_XY_through_Z'] for r in results])
    acc_vals = np.array([r['acc'] for r in results])

    mask = (acc_vals > 0.02) & (acc_vals < 0.99)
    if mask.sum() >= 4:
        try:
            popt, _ = curve_fit(sigmoid, I_vals[mask], acc_vals[mask], p0=[5, 1.0], maxfev=5000)
            preds = sigmoid(I_vals[mask], *popt)
            r2 = 1 - np.sum((acc_vals[mask] - preds)**2) / max(np.sum((acc_vals[mask] - acc_vals[mask].mean())**2), 1e-10)
            print(f"\n  acc = σ({popt[0]:.2f}·(I - {popt[1]:.3f}))  R²={r2:.4f}")
            print(f"  转折点 I* = {popt[1]:.3f} bits")
        except Exception as e:
            print(f"  fit failed: {e}")

    return results


# ============================================================
# Part C: 涌现模拟 — 多阈值sigmoid
# ============================================================
def partC_emergence():
    """
    模拟涌现：定义多个子任务，每个需要不同的I阈值。
    当I超过某个阈值时，对应子任务的acc突然上升。
    总acc = 平均各子任务的sigmoid。
    """
    print("\n" + "=" * 70)
    print("Part C: 涌现模拟 — 多阈值sigmoid叠加")
    print("=" * 70)

    # 定义5个子任务，各有不同难度（不同I*阈值）
    tasks = [
        {"name": "简单分类(0-4)", "I_threshold": 0.3, "slope": 8.0},
        {"name": "中等分类(5-9)", "I_threshold": 0.8, "slope": 6.0},
        {"name": "细粒度区分", "I_threshold": 1.5, "slope": 5.0},
        {"name": "推理任务", "I_threshold": 2.5, "slope": 4.0},
        {"name": "复杂推理", "I_threshold": 3.5, "slope": 3.0},
    ]

    print("""
  模型：每个子任务的准确率 = σ(slope · (I - I_threshold))
  总能力 = 所有子任务的平均准确率

  当 I 从 0 增加：
    - I < 0.3: 所有任务随机
    - I ≈ 0.5: 简单任务涌现
    - I ≈ 1.0: 中等任务涌现
    - I ≈ 2.0: 细粒度涌现
    - I ≈ 3.0: 推理涌现
    - I > 4.0: 全部解决

  这就是涌现的信息论解释：涌现 = I超过特定任务的阈值
    """)

    I_range = np.linspace(0, 5, 200)
    total_acc = np.zeros_like(I_range)

    print(f"  {'任务':>15s} {'I*':>5s} {'slope':>6s}")
    for task in tasks:
        task_acc = sigmoid(I_range, task['slope'], task['I_threshold'])
        total_acc += task_acc
        print(f"  {task['name']:>15s} {task['I_threshold']:>5.1f} {task['slope']:>6.1f}")
    total_acc /= len(tasks)

    # 总能力也是sigmoid吗？
    try:
        popt, _ = curve_fit(sigmoid, I_range, total_acc, p0=[2, 2.0], maxfev=5000)
        preds = sigmoid(I_range, *popt)
        r2 = 1 - np.sum((total_acc - preds)**2) / np.sum((total_acc - total_acc.mean())**2)
        print(f"\n  总能力 = σ({popt[0]:.2f}·(I - {popt[1]:.3f}))  R²={r2:.4f}")
        if r2 > 0.99:
            print("  ✓ 多阈值sigmoid叠加后仍然是sigmoid — 这解释了为什么涌现看起来像相位跃迁")
        else:
            print(f"  R²={r2:.3f} — 不是完美的sigmoid，但接近")
    except:
        print("  fit failed")

    # 实验验证：用真实的bottleneck分类器
    print("\n  --- 实验验证：多子任务涌现 ---")
    X, Y = load_mnist()
    n = 5000
    np.random.seed(42)
    idx = np.random.choice(len(X), n, replace=False)
    X_sub, Y_sub = X[idx], Y[idx]
    d_in = X_sub.shape[1]

    # 子任务：数字0-4（简单），5-9（难）
    easy_mask = Y_sub < 5
    hard_mask = Y_sub >= 5

    noise_levels = [0, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]

    # 训练两个分类头：easy(5类)和hard(5类)
    backbone = nn.Sequential(
        nn.Linear(d_in, 256), nn.ReLU(),
        nn.Linear(256, 64), nn.ReLU(),
    ).to(device)
    cls_easy = nn.Linear(64, 5).to(device)
    cls_hard = nn.Linear(64, 5).to(device)

    params = list(backbone.parameters()) + list(cls_easy.parameters()) + list(cls_hard.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)

    # 训练
    loader = DataLoader(
        TensorDataset(torch.tensor(X_sub), torch.tensor(Y_sub)),
        batch_size=128, shuffle=True)
    for ep in range(15):
        backbone.train(); cls_easy.train(); cls_hard.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            feat = backbone(xb)
            # easy: 数字0-4
            easy_idx = yb < 5
            if easy_idx.sum() > 0:
                l_easy = F.cross_entropy(cls_easy(feat[easy_idx]), yb[easy_idx])
            else:
                l_easy = 0
            # hard: 数字5-9
            hard_idx = yb >= 5
            if hard_idx.sum() > 0:
                l_hard = F.cross_entropy(cls_hard(feat[hard_idx]), yb[hard_idx] - 5)
            else:
                l_hard = 0
            loss = l_easy + l_hard
            opt.zero_grad(); loss.backward(); opt.step()

    # 评估
    print(f"\n  {'σ':>5s} {'I(bits)':>7s} {'easy_acc':>8s} {'hard_acc':>8s} {'total':>7s}")
    easy_results = []
    hard_results = []
    for sigma in noise_levels:
        X_noisy = X_sub + np.random.randn(*X_sub.shape).astype(np.float32) * sigma
        X_noisy = np.clip(X_noisy, 0, 1)

        backbone.eval(); cls_easy.eval(); cls_hard.eval()
        with torch.no_grad():
            feat = backbone(torch.tensor(X_noisy).to(device))
            probs_easy = F.softmax(cls_easy(feat), dim=1).cpu().numpy()
            probs_hard = F.softmax(cls_hard(feat), dim=1).cpu().numpy()

        # Easy accuracy
        easy_idx = Y_sub < 5
        easy_preds = probs_easy[easy_idx].argmax(1)
        easy_acc = (easy_preds == Y_sub[easy_idx]).mean()

        # Hard accuracy
        hard_idx = Y_sub >= 5
        hard_preds = probs_hard[hard_idx].argmax(1)
        hard_acc = (hard_preds == (Y_sub[hard_idx] - 5)).mean()

        # I估计（用easy task的输出）
        log_py = np.log(probs_easy[easy_idx, easy_preds] + 1e-10)
        H_Y = np.log(5)  # 5类均匀
        H_Y_given_X = -np.mean(log_py)
        I_est = max(H_Y - H_Y_given_X, 0)

        total = (easy_acc * easy_mask.sum() + hard_acc * hard_mask.sum()) / len(Y_sub)

        easy_results.append({'sigma': sigma, 'I': I_est, 'acc': easy_acc})
        hard_results.append({'sigma': sigma, 'I': I_est, 'acc': hard_acc})

        print(f"  {sigma:>5.1f} {I_est:>7.3f} {easy_acc:>8.3f} {hard_acc:>8.3f} {total:>7.3f}")

    # 检验：easy和hard是否有不同的I*阈值？
    I_all = np.array([r['I'] for r in easy_results])
    acc_e = np.array([r['acc'] for r in easy_results])
    acc_h = np.array([r['acc'] for r in hard_results])

    mask_e = (acc_e > 0.05) & (acc_e < 0.98)
    mask_h = (acc_h > 0.05) & (acc_h < 0.98)

    if mask_e.sum() >= 3:
        try:
            popt_e, _ = curve_fit(sigmoid, I_all[mask_e], acc_e[mask_e], p0=[5, 0.5], maxfev=5000)
            print(f"\n  Easy: acc = σ({popt_e[0]:.2f}·(I - {popt_e[1]:.3f}))  I*={popt_e[1]:.3f} bits")
        except:
            pass
    if mask_h.sum() >= 3:
        try:
            popt_h, _ = curve_fit(sigmoid, I_all[mask_h], acc_h[mask_h], p0=[3, 1.0], maxfev=5000)
            print(f"  Hard: acc = σ({popt_h[0]:.2f}·(I - {popt_h[1]:.3f}))  I*={popt_h[1]:.3f} bits")

            if 'popt_e' in dir() and 'popt_h' in dir():
                print(f"\n  涌现阈值差: Hard比Easy需要多 {popt_h[1] - popt_e[1]:.3f} bits的信息")
                print("  这就是涌现的信息论解释：难任务需要更高的I阈值")
        except:
            pass

    return easy_results, hard_results


# ============================================================
# Main
# ============================================================
def main():
    t0 = time.time()

    print("=" * 70)
    print("PRISM v17 — MI→Performance Sigmoid的普适性测试")
    print("=" * 70)
    print("""
  核心问题：sigmoid MI→performance 是检索独有还是普适的？

  如果普适：
    performance = σ(α·I + β) 对所有AI任务成立
    涌现 = I超过某个任务的阈值
    缩放 = 更大模型→更大I→更多任务被解锁
    这就是统一框架
    """)

    partA_classification()
    partB_bottleneck()
    partC_emergence()

    print("\n" + "=" * 70)
    print(f"总时间: {(time.time()-t0)/60:.1f}分钟")
    print("=" * 70)


if __name__ == '__main__':
    main()
