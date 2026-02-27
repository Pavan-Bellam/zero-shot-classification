import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import wandb
from model import BiEncoderModel, PolyEncoderModel, ColBERTModel
from dataset import CustomDataset, BiEncoderCollator
from utils import load_samples, split_samples, samples_to_pairs
from pathlib import Path
import yaml



def train(model, train_data, val_data, train_cfg, device, save_dir):
    model.train()
    opt = AdamW(
        model.parameters(),
        lr=float(train_cfg['lr'])
    )
    max_steps = train_cfg['max_steps']
    warmup_ratio = train_cfg.get('warmup_ratio', 0.1)
    scheduler = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=int(max_steps * warmup_ratio),
        num_training_steps=max_steps,
    )
    wandb.define_metric("step")
    wandb.define_metric("train/*", step_metric="step")
    wandb.define_metric("val/*", step_metric="step")
    global_step = 0
    best_val_loss = float('inf')
    pbar = tqdm(total=max_steps, desc="Training")
    while global_step < max_steps:
        for batch in train_data:
            if global_step >= max_steps:
                break
            loss, scores = model(batch)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            scheduler.step()
            global_step += 1

            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            if global_step % 10 == 0:
                wandb.log({"train/loss": loss.item(), "step": global_step})

            if global_step % 50 == 0:
                model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    for val_batch in val_data:
                        v_loss, _ = model(val_batch)
                        val_loss += v_loss.item()

                avg_val_loss = val_loss / len(val_data)
                wandb.log({"val/loss": avg_val_loss, "step": global_step})
                tqdm.write(f"Step {global_step} | val_loss: {avg_val_loss:.4f}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    model.save_pretrained(save_dir / "best")
                    tqdm.write(f"  -> New best model saved (val_loss: {avg_val_loss:.4f})")

                model.train()
    pbar.close()


def main(config_path: str):

    with open(config_path) as f:
        config = yaml.safe_load(f)

    seed = config.get('seed', 422)
    torch.manual_seed(seed)

    data_path = config['data']['path']
    train_cfg = config['train']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_type = train_cfg.get('model_type', 'bi_encoder')
    if model_type == 'poly_encoder':
        model = PolyEncoderModel(
            train_cfg['model_name'],
            train_cfg.get('max_num_labels', 5),
            train_cfg.get('pos_weight_scale', 0.0),
            train_cfg.get('n_code_vectors', 2),
        ).to(device)
    elif model_type == 'colbert':
        model = ColBERTModel(
            train_cfg['model_name'],
            train_cfg.get('max_num_labels', 5),
            train_cfg.get('pos_weight_scale', 0.0),
            train_cfg.get('colbert_dim', 128),
        ).to(device)
    else:
        model = BiEncoderModel(
            train_cfg['model_name'],
            train_cfg.get('max_num_labels', 5),
            train_cfg.get('pos_weight_scale', 0.0),
        ).to(device)

    samples = load_samples(data_path)
    train_samples, val_samples, test_samples = split_samples(samples, seed)
    train_pairs = samples_to_pairs(train_samples)
    val_pairs = samples_to_pairs(val_samples)

    train_data = CustomDataset(train_pairs, tokenizer=model.tokenizer)
    val_data = CustomDataset(val_pairs, tokenizer=model.tokenizer)
    collator = BiEncoderCollator(tokenizer=model.tokenizer)
    train_dl = DataLoader(train_data, batch_size=train_cfg['batch_size'], shuffle=True, collate_fn=collator)
    val_dl = DataLoader(val_data, batch_size=train_cfg['batch_size'], shuffle=False, collate_fn=collator)

    print(f"Samples=> train: {len(train_samples)} | val: {len(val_samples)} | test: {len(test_samples)}")
    print(f"Pairs=>   train: {len(train_data)} | val: {len(val_data)}")

    wandb.init(project="fastino-2", config=config)
    save_dir = Path("checkpoints") / wandb.run.name
    train(model, train_dl, val_dl, train_cfg, device, save_dir)

    model.save_pretrained(save_dir / "last")
    print(f"Last model saved to {save_dir / 'last'}")
    print(f"Best model saved to {save_dir / 'best'}")

    wandb.finish()




if __name__ == "__main__":
    main('./config.yml')
