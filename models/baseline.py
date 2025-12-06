from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
import numpy as np

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

#load model
model_name = "gpt2"   
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).cuda()

# Set pad token for gp2 model
tokenizer.pad_token = tokenizer.eos_token

ds = load_dataset("wikitext", "wikitext-2-raw-v1")

# tokenize dataset
tokenized_ds = ds.map(tokenize_function, batched=True, remove_columns=["text"])


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

args = TrainingArguments(
    output_dir="./tmp",
    per_device_train_batch_size=8,
    num_train_epochs=1,
    logging_steps=100,
    save_strategy="no",
    fp16=True,
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    data_collator=data_collator,
)


trainer.train()

#save
model.save_pretrained("./gpt2-wikitext2-finetuned")
tokenizer.save_pretrained("./gpt2-wikitext2-finetuned")

#eval
eval_results = trainer.evaluate(eval_dataset=tokenized_ds["test"])
print(f"Perplexity: {np.exp(eval_results['eval_loss']):.2f}")