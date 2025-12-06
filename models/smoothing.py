import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter, defaultdict
calib_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")

model_path = "./gpt2-wikitext2-finetuned"



tokenizer = AutoTokenizer.from_pretrained(model_path)
def tokenize(example):
    return tokenizer(example["text"], add_special_tokens=False)["input_ids"]

calib_tokens = []
for ex in calib_data:
    calib_tokens.extend(tokenize(ex))
counts = Counter(calib_tokens)

Phi = defaultdict(int)
for tok, c in counts.items():
    Phi[c] += 1
# Plot the raw Φ values
ts = sorted([t for t in Phi.keys() if t > 0])
phi_vals = [Phi[t] for t in ts]

plt.figure(figsize=(12, 5))

# Plot 1: Raw Φ(t) - Linear scale
plt.subplot(1, 2, 1)
plt.scatter(ts, phi_vals, alpha=0.6, s=50)
plt.xlabel('Frequency t (how many times a token appears)')
plt.ylabel('Φ(t) (number of tokens with frequency t)')
plt.title('Raw Frequency-of-Frequencies')
plt.grid(True, alpha=0.3)

# Plot 2: Log-log scale (this is where we fit the line)
plt.subplot(1, 2, 2)
plt.scatter(ts, phi_vals, alpha=0.6, s=50, label='Observed')
plt.xlabel('log(Frequency t)')
plt.ylabel('log(Φ(t))')
plt.xscale('log')
plt.yscale('log')
plt.title('Log-Log Scale (for fitting)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('phi_distribution.png', dpi=150)
print("Saved plot to phi_distribution.png")

# Print some statistics
print(f"\nTotal buckets: {len(Phi)}")
print(f"Max frequency: {max(ts)}")
print(f"Buckets with gaps (Φ(t+1) = 0):")
gaps = []
for t in range(1, max(ts)):
    if Phi.get(t, 0) > 0 and Phi.get(t+1, 0) == 0:
        gaps.append(t)
print(f"  {len(gaps)} gaps found at frequencies: {gaps[:20]}...")  # Show first 20