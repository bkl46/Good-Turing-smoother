import torch
import math
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from collections import Counter

def compute_cross_entropy_and_approx_kl(model, tokenizer, test_texts, batch_size=8, max_length=128):
    model.eval()
    device = next(model.parameters()).device
    
    inputs = tokenizer(test_texts, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    input_ids = inputs["input_ids"]
    attn_mask = inputs["attention_mask"]
    ds = torch.utils.data.TensorDataset(input_ids, attn_mask)
    loader = DataLoader(ds, batch_size=batch_size)

    total_logprob = 0.0
    total_tokens = 0
    all_tokens = []

    with torch.no_grad():
        for batch in loader:
            ids, mask = [b.to(device) for b in batch]
            outputs = model(ids, attention_mask=mask)
            # outputs.logits: (B, L, V)
            logits = outputs.logits
            # shift tokens: predict token t given context up to t-1
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = ids[:, 1:].contiguous()
            log_probs = torch.log_softmax(shift_logits, dim=-1)
            # gather log-prob of true next tokens
            token_logprobs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)  # (B, L-1)
            # mask padding tokens
            token_mask = mask[:, 1:].contiguous()
            total_logprob += (token_logprobs * token_mask).sum().item()
            num_tokens = token_mask.sum().item()
            total_tokens += num_tokens
            all_tokens.extend(shift_labels[token_mask.bool()].cpu().tolist())

    cross_entropy = - total_logprob / total_tokens  
 
    
    ctr = Counter(all_tokens)
    total = sum(ctr.values())

    H_p = - sum((c/total) * math.log(c/total) for c in ctr.values())
    approx_KL = cross_entropy - H_p
    return cross_entropy, H_p, approx_KL


model = AutoModelForCausalLM.from_pretrained("./gpt2-wikitext2-finetuned")
tokenizer = AutoTokenizer.from_pretrained("./gpt2-wikitext2-finetuned")
model = model.cuda()


ds = load_dataset("wikitext", "wikitext-2-raw-v1")
test_texts = list(ds['test']['text']) 


ce, h_p, kl = compute_cross_entropy_and_approx_kl(
    model, 
    tokenizer, 
    test_texts, 
    batch_size=8, 
    max_length=128
)

print(f"Cross-Entropy: {ce:.4f} nats")
print(f"Perplexity: {math.exp(ce):.2f}")
print(f"Empirical Entropy (H_p): {h_p:.4f} nats")
print(f"Approx KL Divergence: {kl:.4f} nats")