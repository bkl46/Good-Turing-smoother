import math
import torch
from collections import Counter, defaultdict
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np
from scipy import stats

device = "cuda" if torch.cuda.is_available() else "cpu"


#######################################################################
# 0. Load model and tokenizer
#######################################################################

model_path = "./gpt2-wikitext2-finetuned"

model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
vocab_size = tokenizer.vocab_size


#######################################################################
# 1. Load datasets: calibration + test
#######################################################################


test_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
calib_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")


def tokenize(example):
    return tokenizer(example["text"], add_special_tokens=False)["input_ids"]

calib_tokens = []
for ex in calib_data:
    calib_tokens.extend(tokenize(ex))


#######################################################################
# 2. Compute multiplicities N_x and Φ_t
#######################################################################


def smooth_phi(Phi, max_t):
    """
    Smooth the frequency-of-frequencies using log-linear regression.
    Returns smoothed Φ values for all t from 1 to max_t.
    """
    # Get observed (t, Φ(t)) pairs
    ts = []
    phis = []
    for t, phi_t in Phi.items():
        if t > 0 and phi_t > 0:  # Only use positive values for log
            ts.append(t)
            phis.append(phi_t)
    
    if len(ts) < 2:
        # Not enough data to fit
        return Phi
    
    # Fit log-log linear model
    log_ts = np.log(ts)
    log_phis = np.log(phis)
    slope, intercept, _, _, _ = stats.linregress(log_ts, log_phis)
    
    # Generate smoothed values for all t
    smoothed_Phi = {}
    for t in range(1, max_t + 2):  # +2 because we need t+1
        if t in Phi and Phi[t] > 0:
           
            smoothed_Phi[t] = Phi[t]
        else:
           
            log_t = np.log(t)
            log_phi_smooth = intercept + slope * log_t
            smoothed_Phi[t] = max(np.exp(log_phi_smooth), 0.1)  # Floor at 0.1
    
    return smoothed_Phi
counts = Counter(calib_tokens)

Phi = defaultdict(int)
for tok, c in counts.items():
    Phi[c] += 1


n = len(calib_tokens)

seen_tokens = set(counts.keys())
Phi_0 = vocab_size - len(seen_tokens)


#######################################################################
# 3. Compute Simple Good-Turing bucket mass estimates
#######################################################################

gt_mass = {}  


Phi1 = Phi.get(1, 0)
gt_mass[0] = Phi1 / n


max_t = max(Phi.keys()) if len(Phi) > 0 else 1

Phi = smooth_phi(Phi, max_t)

for t in range(1, max_t + 1):
    next_count = Phi.get(t + 1, 0)
    gt_mass[t] = (t + 1) * next_count / n
total_mass = sum(gt_mass.values())
for t in gt_mass:
    gt_mass[t] /= total_mass

print("GT mass values:")
for t in sorted(gt_mass.keys())[:20]:
    print(f"  Bucket {t}: {gt_mass[t]:.6f}")
print(f"  Sum of all gt_mass: {sum(gt_mass.values()):.6f}")


#######################################################################
# 4. Build token -> bucket map
#######################################################################

def get_bucket(token_id):
    """Bucket = multiplicity (0,1,2,...)"""
    return counts.get(token_id, 0)

token_bucket = {tok_id: get_bucket(tok_id) for tok_id in range(vocab_size)}


#######################################################################
# 5. The SGT smoothing operator
#######################################################################

def apply_sgt_smoothing(logits, token_bucket, gt_mass, vocab_size):
    """
    logits: [batch, vocab]
    Returns:
        new_probs: [batch, vocab]
    """
    # Base model distribution
    q = torch.softmax(logits, dim=-1)

    # Collect tokens per bucket
    bucket_to_tokens = defaultdict(list)
    for tok in range(vocab_size):
        t = token_bucket.get(tok, 0)
        bucket_to_tokens[t].append(tok)

    new_q = torch.zeros_like(q)

    eps = 1e-12

    # DEBUG: Track mass before/after
    original_mass = {}
    new_mass = {}

    # For each bucket t:
    for t, tok_list in bucket_to_tokens.items():
        tok_tensor = torch.tensor(tok_list, device=logits.device)

        model_mass = q[:, tok_tensor].sum(dim=-1, keepdim=True)  # [batch,1]
        original_mass[t] = model_mass[0].item()

        if t not in gt_mass:
            # No GT estimate → keep model probabilities
            new_q[:, tok_tensor] = q[:, tok_tensor]
            new_mass[t] = model_mass[0].item()
            continue

        gt_mass_t = gt_mass[t]

        # Expand GT mass
        target_mass = torch.full_like(model_mass, gt_mass_t)

        # Scale factor
        scale = target_mass / (model_mass + eps)

        new_q[:, tok_tensor] = q[:, tok_tensor] * scale
        new_mass[t] = new_q[0, tok_tensor].sum().item()

    # DEBUG: Print mass distribution for first batch
    if torch.rand(1).item() < 0.001:  # Print occasionally
        print("\n=== Mass Distribution ===")
        for t in sorted(set(list(original_mass.keys()) + list(new_mass.keys()))):
            print(f"Bucket {t}: Original={original_mass.get(t, 0):.6f}, GT={gt_mass.get(t, 0):.6f}, New={new_mass.get(t, 0):.6f}")
        print(f"Total new mass before renorm: {sum(new_mass.values()):.6f}")
    min_prob = 1e-10
    new_q = torch.clamp(new_q, min=min_prob)

    # Renormalize to a normalized distribution
    new_q = new_q / new_q.sum(dim=-1, keepdim=True)
    return new_q
#######################################################################
# 6. Evaluate function (baseline or smoothed)
#######################################################################

def evaluate_model(use_sgt=False):
    total_nll = 0.0
    total_tokens = 0

    for ex in tqdm(test_data):
        text = ex["text"]
        tokens = tokenizer(text, truncation=True, max_length=512)["input_ids"]

        if len(tokens) < 2:
            continue

        input_ids = torch.tensor(tokens[:-1]).unsqueeze(0).to(device)
        target_ids = torch.tensor(tokens[1:]).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(input_ids).logits  # [1, seq-1, vocab]

            # Evaluate token by token
            for t in range(logits.size(1)):
                step_logits = logits[:, t, :]

                if use_sgt:
                    probs = apply_sgt_smoothing(step_logits, token_bucket, gt_mass, vocab_size)
                else:
                    probs = torch.softmax(step_logits, dim=-1)

                true_tok = target_ids[0, t]
                log_p = torch.log(probs[0, true_tok] + 1e-12)

                total_nll += -log_p.item()
                total_tokens += 1

    ce = total_nll / total_tokens
    ppl = math.exp(ce)
    return ce, ppl

def evaluate_model_vectorized(use_sgt=False):
    total_nll = 0.0
    total_tokens = 0

    for ex in tqdm(test_data):
        text = ex["text"]
        tokens = tokenizer(text, truncation=True, max_length=512)["input_ids"]

        if len(tokens) < 2:
            continue

        input_ids = torch.tensor(tokens[:-1]).unsqueeze(0).to(device)
        target_ids = torch.tensor(tokens[1:]).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(input_ids).logits  # [1, seq-1, vocab]

            if use_sgt:
                # apply SGT smoothing to the entire logits tensor at once
                seq_len = logits.size(1)
                probs = apply_sgt_smoothing(
                    logits.view(-1, vocab_size),
                    token_bucket,
                    gt_mass,
                    vocab_size
                )
                # reshape back to [1, seq_len, vocab]
                probs = probs.view(1, seq_len, vocab_size)
                # DEBUG
                print(f"probs shape: {probs.shape}")
                print(f"target_ids shape: {target_ids.shape}")
                print(f"probs sum per position: {probs.sum(dim=-1)}")
                print(f"probs min: {probs.min()}, max: {probs.max()}")
            else:
                probs = torch.softmax(logits, dim=-1)

          
            log_probs = torch.log(
                probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1) + 1e-12
            )

            total_nll += -log_probs.sum().item()
            total_tokens += target_ids.numel()

    ce = total_nll / total_tokens
    ppl = math.exp(ce)
    return ce, ppl
#######################################################################
# 7. Evaluate baseline (no SGT) + SGT-smoothed model
#######################################################################

print("\nEvaluating BASELINE model...")
base_ce, base_ppl = evaluate_model_vectorized(use_sgt=False)
print("Baseline cross-entropy:", base_ce)
print("Baseline perplexity:", base_ppl)

print("\nEvaluating SGT-SMOOTHED model...")
sgt_ce, sgt_ppl = evaluate_model_vectorized(use_sgt=True)
print("SGT-smoothed cross-entropy:", sgt_ce)
print("SGT-smoothed perplexity:", sgt_ppl)


#######################################################################
# 8. Summary
#######################################################################

print("\n===== RESULTS =====")
print(f"Baseline CE: {base_ce:.4f}  | Baseline PPL: {base_ppl:.4f}")
print(f"SGT CE:      {sgt_ce:.4f}  | SGT PPL:      {sgt_ppl:.4f}")
print("Improvement (CE):", base_ce - sgt_ce)
