"""
PRISM v18b — Pythia本地验证
============================

用本地下载的Pythia模型验证 sigmoid emergence。
从本地目录加载模型，不需要网络。
"""

import sys, os, glob
sys.path.insert(0, '/Users/wp/ZCodeProject/PRISM')

import torch, torch.nn.functional as F
import numpy as np
from scipy.optimize import curve_fit
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device('cpu')  # CPU模式更稳定
print(f"Device: {device}")
BASE_DIR = '/Users/wp/ZCodeProject/PRISM'
MODEL_DIR = os.path.join(BASE_DIR, 'data/pythia_models')


def sigmoid(x, a, b):
    return 1.0 / (1.0 + np.exp(-np.clip(a * (x - b), -500, 500)))


def evaluate_next_token(tokenizer, model, prompts_answers):
    """测量正确答案token的概率"""
    correct_top1 = 0
    total_log_prob = 0
    n = len(prompts_answers)

    for prompt, answer in prompts_answers:
        inputs = tokenizer(prompt, return_tensors="pt")
        answer_ids = tokenizer.encode(" " + answer, add_special_tokens=False)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1]
            probs = F.softmax(logits, dim=0)

        if len(answer_ids) > 0:
            prob = probs[answer_ids[0]].item()
            total_log_prob += np.log(prob + 1e-10)
            if probs.argmax().item() == answer_ids[0]:
                correct_top1 += 1

    return correct_top1 / n, total_log_prob / n


def evaluate_perplexity(tokenizer, model, texts):
    """测量perplexity"""
    total_loss = 0
    total_tokens = 0
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])
            total_loss += outputs.loss.item() * inputs['input_ids'].shape[1]
            total_tokens += inputs['input_ids'].shape[1]
    avg_loss = total_loss / total_tokens
    return np.exp(avg_loss), avg_loss


def main():
    print("=" * 70)
    print("PRISM v18b — Pythia本地验证")
    print("=" * 70)

    # 测试任务
    knowledge_prompts = [
        ("The capital of France is", "Paris"),
        ("The color of grass is", "green"),
        ("2 + 2 equals", "4"),
        ("The largest planet is", "Jupiter"),
        ("Birds can", "fly"),
        ("The sky is", "blue"),
        ("Fish live in", "water"),
        ("Ice is", "cold"),
        ("Fire is", "hot"),
        ("The sun rises in the", "east"),
        ("Cats say", "meow"),
        ("Apples are", "red"),
        ("Snow is", "white"),
        ("Sugar is", "sweet"),
        ("Night is", "dark"),
    ]

    test_texts = [
        "The history of artificial intelligence began in antiquity, with myths and stories of artificial beings endowed with intelligence.",
        "Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience.",
        "Natural language processing is a subfield of linguistics and artificial intelligence concerned with the interactions between computers and human language.",
    ]

    # 扫描可用的模型（直接检查每个目录）
    model_sizes = {
        'pythia-70m-deduped': 70e6,
        'pythia-160m-deduped': 160e6,
        'pythia-410m-deduped': 410e6,
        'pythia-1b-deduped': 1e9,
        'pythia-1.4b-deduped': 1.4e9,
        'pythia-2.8b-deduped': 2.8e9,
    }
    model_dirs = [os.path.join(MODEL_DIR, name) for name in model_sizes.keys()
                  if os.path.isdir(os.path.join(MODEL_DIR, name))]

    results = []

    for md in model_dirs:
        name = os.path.basename(md)
        if name not in model_sizes:
            continue

        # 检查是否有完整模型文件
        safetensors = os.path.join(md, 'model.safetensors')
        pytorch_bin = os.path.join(md, 'pytorch_model.bin')
        has_model = False
        model_file_size = 0

        if os.path.exists(safetensors):
            model_file_size = os.path.getsize(safetensors)
            if model_file_size > 50_000_000:
                has_model = True

        if not has_model and os.path.exists(pytorch_bin):
            model_file_size = os.path.getsize(pytorch_bin)
            if model_file_size > 50_000_000:
                has_model = True

        if not has_model:
            print(f"  {name}: no valid model file found, skipping")
            continue

        n_params = model_sizes[name]
        print(f"\n  Loading {name} ({n_params/1e6:.0f}M, {model_file_size/1e6:.0f}MB)...")

        try:
            tokenizer = AutoTokenizer.from_pretrained(md, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(md, local_files_only=True)
            model.eval()

            # 评估
            acc, log_prob = evaluate_next_token(tokenizer, model, knowledge_prompts)
            ppl, loss = evaluate_perplexity(tokenizer, model, test_texts)

            results.append({
                'name': name, 'n_params': n_params,
                'acc': acc, 'log_prob': log_prob,
                'ppl': ppl, 'loss': loss,
                'I_approx': -loss,
            })

            print(f"  acc={acc:.3f}, ppl={ppl:.1f}, loss={loss:.3f}, I≈{-loss:.3f}")

            del model
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

    # 分析
    if len(results) < 3:
        print(f"\n  只有 {len(results)} 个模型，需要至少3个")
        print("  等待更多模型下载完成...")
        return

    print("\n" + "=" * 70)
    print("分析")
    print("=" * 70)

    log_N = np.log10(np.array([r['n_params'] for r in results]))
    accs = np.array([r['acc'] for r in results])
    I_vals = np.array([r['I_approx'] for r in results])

    # 详细表
    print(f"\n  {'Model':>20s} {'N':>8s} {'acc':>6s} {'ppl':>7s} {'loss':>6s} {'I':>6s}")
    for r in results:
        print(f"  {r['name']:>20s} {r['n_params']/1e6:>7.0f}M {r['acc']:>6.3f} "
              f"{r['ppl']:>7.1f} {r['loss']:>6.3f} {r['I_approx']:>6.3f}")

    # 拟合 acc = σ(α·log(N) + β)
    mask = (accs > 0.01) & (accs < 0.99)
    if mask.sum() >= 3:
        try:
            popt, _ = curve_fit(sigmoid, log_N[mask], accs[mask], p0=[5, 8], maxfev=5000)
            preds = sigmoid(log_N[mask], *popt)
            r2 = 1 - np.sum((accs[mask] - preds)**2) / max(np.sum((accs[mask] - accs[mask].mean())**2), 1e-10)
            print(f"\n  acc = σ({popt[0]:.2f}·(log(N) - {popt[1]:.2f}))  R²={r2:.4f}")
            print(f"  涌现阈值 N* ≈ 10^{popt[1]:.1f}")
        except Exception as e:
            print(f"  sigmoid fit failed: {e}")

    # I vs acc
    print(f"\n  I(loss) vs acc (信息论验证):")
    for r in results:
        print(f"    I={r['I_approx']:.3f} → acc={r['acc']:.3f}")


if __name__ == '__main__':
    main()
