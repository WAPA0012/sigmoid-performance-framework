"""
PRISM v18 — 涌现的Sigmoid验证
================================

用公开的LLM benchmark数据验证：
  涌现 = performance(N) 的sigmoid形状

数据来源：
  1. OPT模型套件 (125M-66B) 在标准benchmark上的表现
  2. GPT-3 (125M-175B) 在标准benchmark上的表现
  3. 已知涌现任务的跨规模数据

拟合对比：
  A. 幂律: acc = a · N^(-α)  (Kaplan et al.)
  B. Sigmoid: acc = σ(α·(log(N) - β))  (我们的预测)
  C. 阶跃: acc = H(log(N) - N*)  (简单涌现)
  D. Broken幂律: 分段拟合

如果sigmoid拟合最好 → 涌现是信息阈值的 sigmoid 越过
"""

import numpy as np
from scipy.optimize import curve_fit
import json, time

def sigmoid(x, a, b):
    return 1.0 / (1.0 + np.exp(-np.clip(a * (x - b), -500, 500)))

def power_law(x, a, alpha):
    return np.clip(a * x**(-alpha), 0, 1)

def step_fn(x, x0):
    """阶跃函数（宽松版，用陡峭sigmoid）"""
    return sigmoid(x, 100, x0)

def broken_power(x, a1, a2, alpha1, alpha2, x_break):
    """分段幂律"""
    y1 = a1 * x**(-alpha1)
    y2 = a2 * x**(-alpha2)
    return np.where(x < x_break, y1, y2)

def fit_and_compare(x_vals, y_vals, task_name, random_baseline=0.25):
    """对一组数据拟合四种模型，比较R²"""
    # 归一化y到[0,1]范围（减去随机基线）
    y_norm = np.clip((y_vals - random_baseline) / (1.0 - random_baseline + 1e-10), 0, 1)

    # 只对有变化的数据拟合
    if y_norm.max() - y_norm.min() < 0.05:
        print(f"  [{task_name}] 数据范围太小，跳过")
        return None

    results = {}

    # A. Sigmoid
    try:
        popt_s, _ = curve_fit(sigmoid, x_vals, y_norm, p0=[5, np.median(x_vals)], maxfev=5000)
        pred_s = sigmoid(x_vals, *popt_s)
        r2_s = 1 - np.sum((y_norm - pred_s)**2) / max(np.sum((y_norm - y_norm.mean())**2), 1e-10)
        results['sigmoid'] = {'r2': r2_s, 'params': popt_s, 'pred': pred_s}
    except:
        results['sigmoid'] = {'r2': -999}

    # B. Power law
    try:
        popt_p, _ = curve_fit(power_law, x_vals, y_norm, p0=[1.0, 0.1], maxfev=5000)
        pred_p = power_law(x_vals, *popt_p)
        r2_p = 1 - np.sum((y_norm - pred_p)**2) / max(np.sum((y_norm - y_norm.mean())**2), 1e-10)
        results['power'] = {'r2': r2_p, 'params': popt_p}
    except:
        results['power'] = {'r2': -999}

    # C. Step function
    try:
        popt_step, _ = curve_fit(step_fn, x_vals, y_norm, p0=[np.median(x_vals)], maxfev=5000)
        pred_step = step_fn(x_vals, *popt_step)
        r2_step = 1 - np.sum((y_norm - pred_step)**2) / max(np.sum((y_norm - y_norm.mean())**2), 1e-10)
        results['step'] = {'r2': r2_step, 'params': popt_step}
    except:
        results['step'] = {'r2': -999}

    # D. 线性（baseline）
    try:
        from numpy.polynomial import polynomial as P
        coef = P.polyfit(x_vals, y_norm, 1)
        pred_l = P.polyval(x_vals, coef)
        r2_l = 1 - np.sum((y_norm - pred_l)**2) / max(np.sum((y_norm - y_norm.mean())**2), 1e-10)
        results['linear'] = {'r2': r2_l}
    except:
        results['linear'] = {'r2': -999}

    # 报告
    print(f"\n  [{task_name}] (baseline={random_baseline})")
    print(f"    数据点: {len(x_vals)}")
    print(f"    x range: [{x_vals.min():.1f}, {x_vals.max():.1f}]")
    print(f"    y range: [{y_vals.min():.3f}, {y_vals.max():.3f}]")

    best = max(results.keys(), key=lambda k: results[k]['r2'])
    for name, r in sorted(results.items(), key=lambda x: -x[1]['r2']):
        marker = " ★" if name == best else ""
        print(f"    {name:>10s}: R² = {r['r2']:.4f}{marker}")

    if 'sigmoid' in results and results['sigmoid']['r2'] > -900:
        p = results['sigmoid']['params']
        print(f"    Sigmoid: σ({p[0]:.2f}·(log(N) - {p[1]:.2f}))")
        print(f"    涌现阈值: N* ≈ 10^{p[1]:.1f} = {10**p[1]:.0e} params")

    results['task'] = task_name
    return results


def main():
    print("=" * 70)
    print("PRISM v18 — 涌现的Sigmoid验证")
    print("=" * 70)

    all_results = []

    # ============================================================
    # 数据集1: GPT-3 标准benchmark (Brown et al. 2020, Table 2.1)
    # 模型规模 (参数数) 和准确率
    # ============================================================
    print("\n" + "=" * 70)
    print("数据集1: GPT-3 Benchmark (Brown et al. 2020)")
    print("=" * 70)

    # 模型规模 (log10 params)
    gpt3_sizes = np.array([125e6, 350e6, 1.3e9, 2.7e9, 6.7e9, 13e9, 175e9])
    log_sizes = np.log10(gpt3_sizes)

    # SuperGLUE tasks (Table 3.1-3.2)
    # BoolQ
    gpt3_boolq = np.array([0.585, 0.615, 0.651, 0.663, 0.689, 0.710, 0.765])
    # RTE
    gpt3_rte = np.array([0.527, 0.545, 0.560, 0.567, 0.588, 0.603, 0.657])
    # WSC
    gpt3_wsc = np.array([0.404, 0.413, 0.462, 0.512, 0.558, 0.615, 0.785])
    # MultiRC
    gpt3_multirc = np.array([0.240, 0.260, 0.308, 0.340, 0.460, 0.556, 0.752])

    r = fit_and_compare(log_sizes, gpt3_boolq, "GPT-3 BoolQ", 0.5)
    if r: all_results.append(r)
    r = fit_and_compare(log_sizes, gpt3_rte, "GPT-3 RTE", 0.5)
    if r: all_results.append(r)
    r = fit_and_compare(log_sizes, gpt3_wsc, "GPT-3 WSC", 0.5)
    if r: all_results.append(r)
    r = fit_and_compare(log_sizes, gpt3_multirc, "GPT-3 MultiRC", 0.25)
    if r: all_results.append(r)

    # TriviaQA (emergence example)
    gpt3_triviaqa = np.array([0.005, 0.012, 0.035, 0.062, 0.118, 0.185, 0.642])
    r = fit_and_compare(log_sizes, gpt3_triviaqa, "GPT-3 TriviaQA", 0.0)
    if r: all_results.append(r)

    # LAMBADA (language modeling)
    gpt3_lambada = np.array([0.403, 0.512, 0.624, 0.682, 0.732, 0.758, 0.762])
    r = fit_and_compare(log_sizes, gpt3_lambada, "GPT-3 LAMBADA", 0.0)
    if r: all_results.append(r)

    # ============================================================
    # 数据集2: OPT benchmark (Zhang et al. 2022, Table 1)
    # ============================================================
    print("\n" + "=" * 70)
    print("数据集2: OPT Benchmark (Zhang et al. 2022)")
    print("=" * 70)

    opt_sizes = np.array([125e6, 350e6, 1.3e9, 2.7e9, 6.7e9, 13e9, 30e9, 66e9])
    log_opt = np.log10(opt_sizes)

    # HellaSwag
    opt_hellaswag = np.array([0.257, 0.318, 0.430, 0.501, 0.598, 0.647, 0.697, 0.732])
    r = fit_and_compare(log_opt, opt_hellaswag, "OPT HellaSwag", 0.25)
    if r: all_results.append(r)

    # PIQA
    opt_piqa = np.array([0.604, 0.637, 0.687, 0.712, 0.749, 0.769, 0.787, 0.800])
    r = fit_and_compare(log_opt, opt_piqa, "OPT PIQA", 0.5)
    if r: all_results.append(r)

    # WinoGrande
    opt_winogrande = np.array([0.515, 0.530, 0.568, 0.594, 0.637, 0.663, 0.690, 0.714])
    r = fit_and_compare(log_opt, opt_winogrande, "OPT WinoGrande", 0.5)
    if r: all_results.append(r)

    # ARC-Easy
    opt_arc_e = np.array([0.348, 0.402, 0.486, 0.530, 0.589, 0.620, 0.653, 0.674])
    r = fit_and_compare(log_opt, opt_arc_e, "OPT ARC-Easy", 0.25)
    if r: all_results.append(r)

    # ARC-Challenge (known emergence task)
    opt_arc_c = np.array([0.224, 0.250, 0.285, 0.309, 0.354, 0.380, 0.418, 0.441])
    r = fit_and_compare(log_opt, opt_arc_c, "OPT ARC-Challenge", 0.25)
    if r: all_results.append(r)

    # LAMBADA
    opt_lambada = np.array([0.392, 0.491, 0.608, 0.662, 0.720, 0.744, 0.763, 0.775])
    r = fit_and_compare(log_opt, opt_lambada, "OPT LAMBADA", 0.0)
    if r: all_results.append(r)

    # ============================================================
    # 数据集3: 经典涌现任务 (Wei et al. 2022 style)
    # 多数表决数学 (BIG-Bench)
    # 数据从论文图表估算
    # ============================================================
    print("\n" + "=" * 70)
    print("数据集3: 经典涌现任务 (估算自 Wei et al. 2022)")
    print("=" * 70)

    # Log FLOPs 作为x轴 (Wei et al. 用的是FLOPs不是params)
    # 模型: GPT-2 small(124M), medium(355M), large(774M), xl(1.5B),
    #        GPT-3 ada(350M), babbage(1.3B), curie(6.7B), davinci(175B),
    #        PaLM 8B, 62B, 540B

    # 多数表决数学 (Wei et al. Fig 1A)
    # x = log10(FLOPs), y = accuracy
    # 小模型: ~0%, 大模型: ~80%
    math_x = np.array([19, 20, 21, 22, 22.5, 23, 23.5, 24, 24.5, 25])
    math_y = np.array([0.0, 0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80])
    r = fit_and_compare(math_x, math_y, "多数表决数学 (Wei)", 0.0)
    if r: all_results.append(r)

    # 思维链推理 (Wei et al. Fig 2)
    # GSM8K或类似
    cot_x = np.array([19, 20, 21, 22, 22.5, 23, 23.5, 24, 24.5, 25])
    cot_y = np.array([0.0, 0.0, 0.0, 0.01, 0.02, 0.05, 0.10, 0.25, 0.45, 0.55])
    r = fit_and_compare(cot_x, cot_y, "思维链推理 (Wei)", 0.0)
    if r: all_results.append(r)

    # ============================================================
    # 综合分析
    # ============================================================
    print("\n" + "=" * 70)
    print("综合分析")
    print("=" * 70)

    if all_results:
        sig_r2 = [r['sigmoid']['r2'] for r in all_results if 'sigmoid' in r and r['sigmoid']['r2'] > -900]
        pow_r2 = [r['power']['r2'] for r in all_results if 'power' in r and r['power']['r2'] > -900]
        step_r2 = [r['step']['r2'] for r in all_results if 'step' in r and r['step']['r2'] > -900]
        lin_r2 = [r['linear']['r2'] for r in all_results if 'linear' in r and r['linear']['r2'] > -900]

        print(f"\n  模型对比 ({len(all_results)} 个任务):")
        if sig_r2:
            print(f"    Sigmoid:  mean R² = {np.mean(sig_r2):.3f} ± {np.std(sig_r2):.3f}")
        if pow_r2:
            print(f"    Power:    mean R² = {np.mean(pow_r2):.3f} ± {np.std(pow_r2):.3f}")
        if step_r2:
            print(f"    Step:     mean R² = {np.mean(step_r2):.3f} ± {np.std(step_r2):.3f}")
        if lin_r2:
            print(f"    Linear:   mean R² = {np.mean(lin_r2):.3f} ± {np.std(lin_r2):.3f}")

        # Sigmoid赢了多少次？
        sig_wins = 0
        for r in all_results:
            best = max(['sigmoid', 'power', 'step', 'linear'],
                       key=lambda k: r.get(k, {}).get('r2', -999))
            if best == 'sigmoid':
                sig_wins += 1

        print(f"\n    Sigmoid最佳次数: {sig_wins}/{len(all_results)}")

        # 按涌现强度分组
        print("\n  按任务类型:")
        for r in all_results:
            name = r['task']
            sig = r.get('sigmoid', {}).get('r2', -999)
            pw = r.get('power', {}).get('r2', -999)
            winner = 'sigmoid' if sig >= pw else 'power'
            print(f"    {name:30s}: sigmoid R²={sig:.3f}, power R²={pw:.3f} → {winner}")

    print("""
  解读：
    如果 sigmoid R² > power R² → 涌现更像是信息阈值的越过（支持我们的理论）
    如果 power R² > sigmoid R² → 涌现更像是连续缩放（传统解释）
    如果 step R² ≈ sigmoid R² → 涌现是近似的阶跃函数
    """)


if __name__ == '__main__':
    main()
