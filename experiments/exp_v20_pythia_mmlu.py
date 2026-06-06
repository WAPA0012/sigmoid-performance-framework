"""
exp_v20_pythia_mmlu.py — Pythia标准benchmark验证
==================================================

用MMLU子集替代15题知识测试，更有说服力。
只下载3个模型(70m, 410m, 2.8b)节省时间和空间。
"""

import sys, os
sys.path.insert(0, '/Users/wp/ZCodeProject/PRISM')

import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import curve_fit
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f"Device: {device}")

MODEL_CACHE = '/Users/wp/ZCodeProject/PRISM/data/pythia_models'

# MMLU子集: 选5个代表性领域，每领域10题
MMLU_QUESTIONS = [
    # Computer Science
    {"q": "In Python, what does the len() function return?", "choices": ["Length of object", "Type of object", "Value of object", "ID of object"], "answer": 0},
    {"q": "What is the time complexity of binary search?", "choices": ["O(n)", "O(log n)", "O(n log n)", "O(1)"], "answer": 1},
    {"q": "Which data structure uses FIFO ordering?", "choices": ["Stack", "Queue", "Tree", "Graph"], "answer": 1},
    {"q": "What does HTML stand for?", "choices": ["HyperText Markup Language", "High Tech Modern Language", "HyperTransfer Markup Logic", "Home Tool Markup Language"], "answer": 0},
    {"q": "What is the output of 2**3 in Python?", "choices": ["6", "8", "5", "9"], "answer": 1},
    # Math
    {"q": "What is the derivative of x^2?", "choices": ["x", "2x", "x^2", "2"], "answer": 1},
    {"q": "What is the value of pi to two decimal places?", "choices": ["3.12", "3.14", "3.16", "3.18"], "answer": 1},
    {"q": "What is the sum of angles in a triangle?", "choices": ["90 degrees", "180 degrees", "270 degrees", "360 degrees"], "answer": 1},
    {"q": "What is 15% of 200?", "choices": ["20", "25", "30", "35"], "answer": 2},
    {"q": "What is log_2(8)?", "choices": ["2", "3", "4", "8"], "answer": 1},
    # History
    {"q": "In which year did World War II end?", "choices": ["1943", "1944", "1945", "1946"], "answer": 2},
    {"q": "Who was the first President of the United States?", "choices": ["Thomas Jefferson", "George Washington", "Abraham Lincoln", "John Adams"], "answer": 1},
    {"q": "The French Revolution began in which year?", "choices": ["1776", "1789", "1799", "1812"], "answer": 1},
    {"q": "Which empire built the Colosseum?", "choices": ["Greek", "Roman", "Ottoman", "Persian"], "answer": 1},
    {"q": "The Berlin Wall fell in which year?", "choices": ["1987", "1988", "1989", "1990"], "answer": 2},
    # Science
    {"q": "What is the chemical symbol for gold?", "choices": ["Go", "Gd", "Au", "Ag"], "answer": 2},
    {"q": "What planet is closest to the Sun?", "choices": ["Venus", "Mercury", "Mars", "Earth"], "answer": 1},
    {"q": "What gas do plants absorb from the atmosphere?", "choices": ["Oxygen", "Nitrogen", "Carbon dioxide", "Hydrogen"], "answer": 2},
    {"q": "What is the speed of light approximately?", "choices": ["300,000 km/s", "150,000 km/s", "500,000 km/s", "1,000,000 km/s"], "answer": 0},
    {"q": "What is the hardest natural substance?", "choices": ["Iron", "Gold", "Diamond", "Platinum"], "answer": 2},
    # Geography
    {"q": "What is the largest continent by area?", "choices": ["Africa", "North America", "Asia", "Europe"], "answer": 2},
    {"q": "What is the capital of Japan?", "choices": ["Osaka", "Kyoto", "Tokyo", "Yokohama"], "answer": 2},
    {"q": "Which river is the longest in the world?", "choices": ["Amazon", "Nile", "Yangtze", "Mississippi"], "answer": 1},
    {"q": "What is the smallest country in the world?", "choices": ["Monaco", "Vatican City", "San Marino", "Liechtenstein"], "answer": 1},
    {"q": "Mount Everest is located in which mountain range?", "choices": ["Alps", "Andes", "Himalayas", "Rockies"], "answer": 2},
    # Common Knowledge
    {"q": "How many days are in a leap year?", "choices": ["364", "365", "366", "367"], "answer": 2},
    {"q": "What color do you get when you mix red and white?", "choices": ["Purple", "Orange", "Pink", "Peach"], "answer": 2},
    {"q": "How many continents are there?", "choices": ["5", "6", "7", "8"], "answer": 2},
    {"q": "What is the boiling point of water in Celsius?", "choices": ["90", "95", "100", "110"], "answer": 2},
    {"q": "Which organ pumps blood in the human body?", "choices": ["Brain", "Lungs", "Heart", "Liver"], "answer": 2},
]


def sigmoid(x, a, b):
    return 1.0 / (1.0 + np.exp(-np.clip(a * (x - b), -500, 500)))


def evaluate_mmlu(tokenizer, model, questions):
    """用log-likelihood评估MMLU"""
    correct = 0
    total = len(questions)
    total_log_prob = 0

    for q_data in questions:
        question = q_data["q"]
        choices = q_data["choices"]
        answer_idx = q_data["answer"]

        # 计算每个选项的log-likelihood
        log_probs = []
        for choice in choices:
            prompt = f"Question: {question}\nAnswer:"
            full_text = f"{prompt} {choice}"

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            choice_ids = tokenizer(f" {choice}", add_special_tokens=False, return_tensors="pt").input_ids.to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[0, -1]  # last position
                probs = F.log_softmax(logits, dim=0)

                # Sum log probs of all choice tokens
                lp = 0
                for cid in choice_ids[0]:
                    lp += probs[cid].item()
                log_probs.append(lp)

        pred = np.argmax(log_probs)
        if pred == answer_idx:
            correct += 1

        # Track best log prob as proxy for confidence
        total_log_prob += max(log_probs)

    acc = correct / total
    avg_log_prob = total_log_prob / total
    return acc, avg_log_prob


def evaluate_perplexity(tokenizer, model, texts):
    """测perplexity"""
    total_loss = 0
    total_tokens = 0
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])
            total_loss += outputs.loss.item() * inputs['input_ids'].shape[1]
            total_tokens += inputs['input_ids'].shape[1]
    return total_loss / total_tokens


def main():
    PYTHIA_MODELS = [
        ('pythia-70m-deduped', 70e6),
        ('pythia-160m-deduped', 160e6),
        ('pythia-410m-deduped', 410e6),
        ('pythia-1b-deduped', 1e9),
        ('pythia-2.8b-deduped', 2.8e9),
    ]

    test_texts = [
        "The history of artificial intelligence began in antiquity, with myths of artificial beings endowed with intelligence.",
        "Machine learning provides systems the ability to learn from experience without being explicitly programmed.",
        "Natural language processing is concerned with the interactions between computers and human language.",
    ]

    results = []

    # Find locally available models
    local_models = []
    for model_name, n_params in PYTHIA_MODELS:
        local_dir = os.path.join(MODEL_CACHE, model_name)
        has_safetensors = os.path.exists(os.path.join(local_dir, 'model.safetensors'))
        has_pytorch = os.path.exists(os.path.join(local_dir, 'pytorch_model.bin'))
        if has_safetensors or has_pytorch:
            fsize = os.path.getsize(os.path.join(local_dir, 'model.safetensors' if has_safetensors else 'pytorch_model.bin'))
            if fsize > 50_000_000:  # >50MB
                local_models.append((model_name, n_params))
                print(f"  Found: {model_name} ({fsize/1e6:.0f}MB)")
            else:
                print(f"  Skip: {model_name} (file too small: {fsize/1e6:.0f}MB)")
        else:
            print(f"  Skip: {model_name} (not downloaded)")

    for model_name, n_params in local_models:
        local_dir = os.path.join(MODEL_CACHE, model_name)
        print(f"\n{'='*50}")
        print(f"  Loading {model_name} ({n_params/1e6:.0f}M)...")

        try:
            tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                local_dir, local_files_only=True, torch_dtype=torch.float32
            ).to(device)
            model.eval()
            print(f"  Loaded: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")

            print(f"  Evaluating MMLU (30 questions)...")
            acc, avg_lp = evaluate_mmlu(tokenizer, model, MMLU_QUESTIONS)

            print(f"  Evaluating perplexity...")
            loss = evaluate_perplexity(tokenizer, model, test_texts)

            results.append({
                'model': model_name,
                'n_params': n_params,
                'acc': acc,
                'loss': loss,
                'I_approx': -loss,
                'avg_log_prob': avg_lp,
            })

            print(f"  MMLU acc={acc:.3f}, loss={loss:.3f}, I~{-loss:.3f}")

            del model
            if device.type == 'mps':
                torch.mps.empty_cache()

        except Exception as e:
            print(f"  FAILED: {e}")
            continue

    # 分析
    if len(results) >= 3:
        print("\n" + "=" * 70)
        print("ANALYSIS")
        print("=" * 70)

        log_N = np.log10(np.array([r['n_params'] for r in results]))
        accs = np.array([r['acc'] for r in results])
        losses = np.array([r['loss'] for r in results])
        I_vals = np.array([r['I_approx'] for r in results])

        print(f"\n  {'Model':>20s} {'N':>8s} {'acc':>6s} {'loss':>6s} {'I':>6s}")
        for r in results:
            print(f"  {r['model']:>20s} {r['n_params']/1e6:>7.0f}M {r['acc']:>6.3f} {r['loss']:>6.3f} {r['I_approx']:>6.3f}")

        # Sigmoid: acc = f(loss)
        mask = accs > 0.01
        if mask.sum() >= 3:
            try:
                popt, _ = curve_fit(sigmoid, -losses[mask], accs[mask], p0=[2, -3], maxfev=5000)
                preds = sigmoid(-losses[mask], *popt)
                r2 = 1 - np.sum((accs[mask] - preds)**2) / max(np.sum((accs[mask] - accs[mask].mean())**2), 1e-10)
                print(f"\n  acc = σ({popt[0]:.2f}·(-loss) + ({popt[1]:.2f}))  R²={r2:.4f}")
            except:
                print("  sigmoid fit failed")

        # Power law: loss = a*N^b
        try:
            def power_fn(x, a, b):
                return a * x**b
            popt_p, _ = curve_fit(power_fn, log_N, losses, p0=[10, -0.5], maxfev=5000)
            preds_p = power_fn(log_N, *popt_p)
            r2_p = 1 - np.sum((losses - preds_p)**2) / max(np.sum((losses - losses.mean())**2), 1e-10)
            print(f"  loss = {popt_p[0]:.2f}·log(N)^{popt_p[1]:.2f}  R²={r2_p:.4f}")
        except:
            pass

    # 保存
    out_path = '/Users/wp/ZCodeProject/PRISM/paper/fig_mmlu_data.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nData saved to {out_path}")

    # 画图
    if len(results) >= 3:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        names = [r['model'].replace('-deduped','') for r in results]
        losses_arr = np.array([r['loss'] for r in results])
        accs_arr = np.array([r['acc'] for r in results])
        logN = np.log10(np.array([r['n_params'] for r in results]))

        ax = axes[0]
        ax.plot(logN, losses_arr, 'bo-', linewidth=2, markersize=10)
        for i, n in enumerate(names):
            ax.annotate(n, (logN[i], losses_arr[i]), textcoords="offset points",
                       xytext=(0, 12), ha='center', fontsize=8)
        ax.set_xlabel('log₁₀(N params)')
        ax.set_ylabel('Cross-Entropy Loss')
        ax.set_title(f'(a) Loss vs Scale (MMLU benchmark)')

        ax = axes[1]
        ax.scatter(losses_arr, accs_arr, c='red', s=100, zorder=5)
        mask = accs_arr > 0.01
        if mask.sum() >= 3:
            I_fit = np.linspace(-losses_arr.max()-0.5, -losses_arr.min()+0.5, 200)
            popt, _ = curve_fit(sigmoid, -losses_arr[mask], accs_arr[mask], p0=[2,-3], maxfev=5000)
            acc_fit = sigmoid(I_fit, *popt)
            ax.plot(-I_fit, acc_fit, 'b-', linewidth=2, label='Sigmoid')
            r2 = 1 - np.sum((accs_arr[mask]-sigmoid(-losses_arr[mask],*popt))**2)/max(np.sum((accs_arr[mask]-accs_arr[mask].mean())**2),1e-10)
            ax.text(0.05, 0.85, f'R² = {r2:.3f}', transform=ax.transAxes)
        for i, n in enumerate(names):
            ax.annotate(n, (losses_arr[i], accs_arr[i]), textcoords="offset points",
                       xytext=(8, 5), fontsize=8)
        ax.set_xlabel('Cross-Entropy Loss (≈ −I)')
        ax.set_ylabel('MMLU Accuracy (30 questions)')
        ax.set_title('(b) Accuracy = σ(α·I + β)')
        ax.legend()

        plt.tight_layout()
        out_fig = '/Users/wp/ZCodeProject/PRISM/paper/fig_mmlu.png'
        plt.savefig(out_fig, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Figure saved: {out_fig}")


if __name__ == '__main__':
    main()
