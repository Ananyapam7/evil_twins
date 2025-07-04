from evil_twins import load_model_tokenizer, DocDataset, optim_gcg, plot_training_curves


model_name = "gpt2"  # Small and fast model
model, tokenizer = load_model_tokenizer(model_name, use_flash_attn_2=False)

optim_prompt = "! " * 5  # Shorter prompt
dataset = DocDataset(
  model=model,
  tokenizer=tokenizer,
  orig_prompt="Tell me a recipe.",  # Shorter prompt
  optim_prompt=optim_prompt,
  n_docs=2,  # Much fewer documents
  doc_len=8,  # Much shorter documents
  gen_batch_size=1,  # Smaller batch size
)

results, ids = optim_gcg(
  model=model,
  tokenizer=tokenizer,
  dataset=dataset,
  n_epochs=2,  # Much fewer epochs
  kl_every=1,
  log_fpath="twin_log.json",
  batch_size=1,  # Smaller batch size
  top_k=10,  # Smaller top_k
  gamma=0.0, 
)

# Automatically plot and save training curves after optimization
plot_training_curves("twin_log.json", "training_curves.png")