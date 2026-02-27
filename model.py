import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download
from pathlib import Path
import json


def _resolve_file(path, filename):
    """Load a file from a local directory or HuggingFace repo. Returns (filepath, exists)."""
    local = Path(path) / filename
    if local.exists():
        return str(local), True
    if not Path(path).is_dir():
        try:
            return hf_hub_download(repo_id=path, filename=filename), True
        except Exception:
            return None, False
    return None, False


class BiEncoderModel(nn.Module):
    def __init__(self, model_name, max_num_labels=5, pos_weight_scale=0.0):
        super(BiEncoderModel, self).__init__()
        self.shared_encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_num_labels = max_num_labels
        self.pos_weight_scale = pos_weight_scale

    def encode(self, input_ids, attention_mask):
        """
        Encodes pre-tokenized inputs using the shared encoder.
        Returns mask-aware mean-pooled embeddings.
        """
        output = self.shared_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        return (output * mask).sum(1) / mask.sum(1).clamp(min=1)

    def encode_texts(self, texts):
        """
        Tokenizes raw strings and encodes them.
        """
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        return self.encode(inputs['input_ids'], inputs['attention_mask'])

    def forward(self, batch):
        """
        In-batch contrastive training forward pass.

        Each batch item is a (text, label) pair. We compute a B*B similarity matrix
        between all text embeddings and all label embeddings. The target matrix marks
        which pairs are positive (diagonal + duplicate texts/labels).

        batch: dict with:
            inputs: {input_ids, attention_mask} — tokenized texts [B, seq_len]
            labels: {input_ids, attention_mask} — tokenized labels [B, seq_len]
            targets: [B, B] binary target matrix

        Returns: (loss, scores)
        """
        device = next(self.parameters()).device
        text_emb = self.encode(
            batch["inputs"]["input_ids"].to(device),
            batch["inputs"]["attention_mask"].to(device),
        )
        label_emb = self.encode(
            batch["labels"]["input_ids"].to(device),
            batch["labels"]["attention_mask"].to(device),
        )
        logits = text_emb @ label_emb.T
        targets = batch["targets"].to(device)
        if self.pos_weight_scale > 0:
            pos_weight = (targets.numel() / targets.sum().clamp(min=1)) ** self.pos_weight_scale
            weight = torch.where(targets == 1, pos_weight, torch.ones_like(targets))
            loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, weight=weight)
        else:
            loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
        return loss, torch.sigmoid(logits)

    def save_pretrained(self, path):
        """Save encoder weights, tokenizer, and model config to a directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.shared_encoder.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        with open(path / "bi_encoder_config.json", "w") as f:
            json.dump({"model_type": "bi_encoder", "max_num_labels": self.max_num_labels, "pos_weight_scale": self.pos_weight_scale}, f)

    @classmethod
    def from_pretrained(cls, path):
        """Load a BiEncoderModel from a directory or HuggingFace repo."""
        config_file, found = _resolve_file(path, "bi_encoder_config.json")
        max_num_labels = 5
        pos_weight_scale = 0.0
        if found:
            with open(config_file) as f:
                cfg = json.load(f)
            max_num_labels = cfg.get("max_num_labels", 5)
            pos_weight_scale = cfg.get("pos_weight_scale", 0.0)
        return cls(path, max_num_labels, pos_weight_scale)



def load_model(path):
    """Load the correct model type from a saved checkpoint or HuggingFace repo."""
    config_file, found = _resolve_file(path, "bi_encoder_config.json")
    if found:
        with open(config_file) as f:
            cfg = json.load(f)
        if cfg.get("model_type") == "poly_encoder":
            return PolyEncoderModel.from_pretrained(path)
        if cfg.get("model_type") == "colbert":
            return ColBERTModel.from_pretrained(path)
    return BiEncoderModel.from_pretrained(path)


class PolyEncoderModel(nn.Module):
    def __init__(self, model_name, max_num_labels=5, pos_weight_scale=0.0, n_code_vectors=2):
        super(PolyEncoderModel, self).__init__()
        self.shared_encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_num_labels = max_num_labels
        self.pos_weight_scale = pos_weight_scale
        self.n_code_vectors = n_code_vectors
        hidden_size = self.shared_encoder.config.hidden_size
        self.codes = nn.Parameter(torch.randn(n_code_vectors, hidden_size) * 0.02)

    def encode_text(self, input_ids, attention_mask):
        """
        Round 1: m code vectors attend over token outputs → m context vectors.
        Returns: [B, m, D]
        """
        output = self.shared_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1)  # [B, seq_len, 1]
        output = output * mask  # zero out padding

        scores = torch.matmul(output, self.codes.T)  # [B, seq_len, m]
        # mask padding positions before softmax so they get zero weight
        padding_mask = (attention_mask == 0).unsqueeze(-1)  # [B, seq_len, 1]
        scores = scores.masked_fill(padding_mask, float('-inf'))
        weights = torch.softmax(scores, dim=1).nan_to_num(0.0)  # [B, seq_len, m]
        context_vecs = torch.matmul(weights.transpose(1, 2), output)  # [B, m, D]

        return context_vecs

    def encode_label(self, input_ids, attention_mask):
        """
        Mean-pooled label embedding. Returns: [B, D]
        """
        output = self.shared_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        return (output * mask).sum(1) / mask.sum(1).clamp(min=1)

    def score(self, context_vecs, label_embed):
        """
        Round 2: each label attends over each text's m context vectors.
        context_vecs: [B_text, m, D]
        label_embed: [B_label, D]
        Returns: [B_text, B_label]
        """
        # attention scores: each label dot each context vector
        attn_scores = torch.einsum('tmd,ld->tlm', context_vecs, label_embed)  # [B_t, B_l, m]
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B_t, B_l, m]

        # weighted sum of context vectors per (text, label) pair
        text_embed = torch.einsum('tlm,tmd->tld', attn_weights, context_vecs)  # [B_t, B_l, D]

        # dot product with label embedding for final score
        scores = torch.einsum('tld,ld->tl', text_embed, label_embed)  # [B_t, B_l]

        return scores

    def forward(self, batch):
        """
        In-batch contrastive training forward pass.
        Returns: (loss, scores [B, B])
        """
        device = next(self.parameters()).device
        context_vecs = self.encode_text(
            batch["inputs"]["input_ids"].to(device),
            batch["inputs"]["attention_mask"].to(device),
        )
        label_embed = self.encode_label(
            batch["labels"]["input_ids"].to(device),
            batch["labels"]["attention_mask"].to(device),
        )
        logits = self.score(context_vecs, label_embed)
        targets = batch["targets"].to(device)
        if self.pos_weight_scale > 0:
            pos_weight = (targets.numel() / targets.sum().clamp(min=1)) ** self.pos_weight_scale
            weight = torch.where(targets == 1, pos_weight, torch.ones_like(targets))
            loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, weight=weight)
        else:
            loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
        return loss, torch.sigmoid(logits)

    def _encode_raw_texts(self, texts):
        """Tokenize and encode texts → context vectors [B, m, D]"""
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        return self.encode_text(inputs['input_ids'], inputs['attention_mask'])

    def _encode_raw_labels(self, labels):
        """Tokenize and encode labels → mean-pooled [B, D]"""
        inputs = self.tokenizer(labels, return_tensors='pt', padding=True, truncation=True)
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        return self.encode_label(inputs['input_ids'], inputs['attention_mask'])

    def save_pretrained(self, path):
        """Save encoder weights, tokenizer, code vectors, and config."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.shared_encoder.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        torch.save(self.codes.data, path / "codes.pt")
        with open(path / "bi_encoder_config.json", "w") as f:
            json.dump({
                "model_type": "poly_encoder",
                "max_num_labels": self.max_num_labels,
                "pos_weight_scale": self.pos_weight_scale,
                "n_code_vectors": self.n_code_vectors,
            }, f)

    @classmethod
    def from_pretrained(cls, path):
        """Load a PolyEncoderModel from a directory or HuggingFace repo."""
        config_file, found = _resolve_file(path, "bi_encoder_config.json")
        max_num_labels = 5
        pos_weight_scale = 0.0
        n_code_vectors = 2
        if found:
            with open(config_file) as f:
                cfg = json.load(f)
            max_num_labels = cfg.get("max_num_labels", 5)
            pos_weight_scale = cfg.get("pos_weight_scale", 0.0)
            n_code_vectors = cfg.get("n_code_vectors", 2)
        model = cls(path, max_num_labels, pos_weight_scale, n_code_vectors)
        codes_file, found = _resolve_file(path, "codes.pt")
        if found:
            model.codes.data = torch.load(codes_file, weights_only=True, map_location='cpu')
        return model



class ColBERTModel(nn.Module):
    """
    Late interaction model inspired by ColBERT.
    Encodes text and labels into per-token embeddings projected to a lower dimension,
    then scores via MaxSim: for each label token, take the max similarity with any
    text token, then average over label tokens.
    """

    def __init__(self, model_name, max_num_labels=5, pos_weight_scale=0.0, dim=128):
        super().__init__()
        self.shared_encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_num_labels = max_num_labels
        self.pos_weight_scale = pos_weight_scale
        self.dim = dim
        hidden_size = self.shared_encoder.config.hidden_size
        self.projection = nn.Linear(hidden_size, dim)
        # learnable scale: L2-normalized embeddings produce cosine similarities in [-1,1],
        # which is too narrow for sigmoid+BCE. scale amplifies logits to a useful range.
        self.score_scale = nn.Parameter(torch.tensor(10.0))

    def encode(self, input_ids, attention_mask):
        """
        Per-token encoding with linear projection and L2 normalization.
        Returns: (embeddings [B, seq_len, dim], attention_mask [B, seq_len])
        """
        output = self.shared_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        projected = self.projection(output)
        projected = nn.functional.normalize(projected, dim=-1)
        return projected, attention_mask

    def maxsim(self, text_emb, text_mask, label_emb, label_mask):
        """
        Late interaction scoring via MaxSim.
        For each label token, take max similarity over text tokens, then average.

        text_emb:   [B_text, seq_text, dim]
        text_mask:  [B_text, seq_text]
        label_emb:  [B_label, seq_label, dim]
        label_mask: [B_label, seq_label]
        Returns:    [B_text, B_label]
        """
        sim = torch.einsum('tsd,lrd->tlsr', text_emb, label_emb)

        text_pad = (text_mask == 0).unsqueeze(1).unsqueeze(-1)  
        sim = sim.masked_fill(text_pad, float('-inf'))

        max_sim = sim.max(dim=2).values

        label_pad = (label_mask == 0).unsqueeze(0)  
        max_sim = max_sim.masked_fill(label_pad, 0.0)

        label_lengths = label_mask.sum(dim=-1).unsqueeze(0).clamp(min=1) 
        scores = max_sim.sum(dim=-1) / label_lengths

        return scores

    def forward(self, batch):
        """
        In-batch contrastive training forward pass using MaxSim scoring.
        Returns: (loss, scores [B, B])
        """
        device = next(self.parameters()).device
        text_emb, text_mask = self.encode(
            batch["inputs"]["input_ids"].to(device),
            batch["inputs"]["attention_mask"].to(device),
        )
        label_emb, label_mask = self.encode(
            batch["labels"]["input_ids"].to(device),
            batch["labels"]["attention_mask"].to(device),
        )
        logits = self.score_scale * self.maxsim(text_emb, text_mask, label_emb, label_mask)
        targets = batch["targets"].to(device)
        if self.pos_weight_scale > 0:
            pos_weight = (targets.numel() / targets.sum().clamp(min=1)) ** self.pos_weight_scale
            weight = torch.where(targets == 1, pos_weight, torch.ones_like(targets))
            loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, weight=weight)
        else:
            loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
        return loss, torch.sigmoid(logits)

    def encode_raw(self, strings, batch_size=32):
        """
        Tokenize and encode a list of strings in batches.
        Returns per-token embeddings and masks with consistent padding.
        """
        all_embs = []
        all_masks = []
        max_seq = 0
        device = next(self.parameters()).device

        for i in range(0, len(strings), batch_size):
            batch = strings[i:i + batch_size]
            inputs = self.tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            emb, mask = self.encode(inputs['input_ids'], inputs['attention_mask'])
            all_embs.append(emb)
            all_masks.append(mask)
            max_seq = max(max_seq, emb.size(1))

        # pad all batches to the same seq_len for concatenation
        padded_embs = []
        padded_masks = []
        for emb, mask in zip(all_embs, all_masks):
            pad_len = max_seq - emb.size(1)
            if pad_len > 0:
                emb = nn.functional.pad(emb, (0, 0, 0, pad_len))
                mask = nn.functional.pad(mask, (0, pad_len))
            padded_embs.append(emb)
            padded_masks.append(mask)

        return torch.cat(padded_embs), torch.cat(padded_masks)

    def save_pretrained(self, path):
        """Save encoder weights, tokenizer, projection layer, and config."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.shared_encoder.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        torch.save(self.projection.state_dict(), path / "projection.pt")
        torch.save(self.score_scale.data, path / "score_scale.pt")
        with open(path / "bi_encoder_config.json", "w") as f:
            json.dump({
                "model_type": "colbert",
                "max_num_labels": self.max_num_labels,
                "pos_weight_scale": self.pos_weight_scale,
                "dim": self.dim,
            }, f)

    @classmethod
    def from_pretrained(cls, path):
        """Load a ColBERTModel from a directory or HuggingFace repo."""
        config_file, found = _resolve_file(path, "bi_encoder_config.json")
        max_num_labels = 5
        pos_weight_scale = 0.0
        dim = 128
        if found:
            with open(config_file) as f:
                cfg = json.load(f)
            max_num_labels = cfg.get("max_num_labels", 5)
            pos_weight_scale = cfg.get("pos_weight_scale", 0.0)
            dim = cfg.get("dim", 128)
        model = cls(path, max_num_labels, pos_weight_scale, dim)
        proj_file, found = _resolve_file(path, "projection.pt")
        if found:
            model.projection.load_state_dict(torch.load(proj_file, weights_only=True, map_location='cpu'))
        scale_file, found = _resolve_file(path, "score_scale.pt")
        if found:
            model.score_scale.data = torch.load(scale_file, weights_only=True, map_location='cpu')
        return model


