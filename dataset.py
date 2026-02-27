from typing import Any

from torch.utils.data import Dataset
import torch
from transformers import PreTrainedTokenizerBase
from transformers import DataCollatorWithPadding


class CustomDataset(Dataset):
    
    def __init__(self, data: list[tuple[str,str]], tokenizer: PreTrainedTokenizerBase) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.data = self.load_data(data)

    def load_data(self, data: list[tuple[str,str]]) -> list[tuple[list[int], list[int]]]:
        texts = []
        labels = []
        for sample in data:
            texts.append(sample[0])
            labels.append(sample[1])

        text_input_ids = self.tokenizer(texts, padding=False, truncation=True)['input_ids']
        label_input_ids = self.tokenizer(labels, padding=False, truncation=True)['input_ids']

        data = [(text_ids, label_ids) for text_ids,label_ids in zip(text_input_ids, label_input_ids)]


        return data

    def __len__(self) -> int:
        return len(self.data)
            
    def __getitem__(self, idx: int) -> dict[str,dict[str,list[int]]]:
        return {
            "inputs": {"input_ids": self.data[idx][0]},
            "labels": {"input_ids": self.data[idx][1]},
        }


class BiEncoderCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.collator = DataCollatorWithPadding(tokenizer)
    
    def __call__(self, batch: list[dict[str,list[int]]]) -> Any:
        inputs = [sample["inputs"] for sample in batch]                                                                                                                            
        labels = [sample["labels"] for sample in batch]  
        padded_inputs = self.collator(inputs) #dict[str,tenosr(batch*max_size)]
        padded_labels = self.collator(labels)
        targets = torch.eye(len(batch))

        for i in range(len(inputs)):
            for j in range(i + 1, len(inputs)):
                if inputs[i]['input_ids'] == inputs[j]['input_ids']:
                    targets[i][j] = 1
                    targets[j][i] = 1
                if labels[i]['input_ids'] == labels[j]['input_ids']:
                    targets[i][j] = 1
                    targets[j][i] = 1

        return {
            "inputs": padded_inputs,
            "labels": padded_labels,
            "targets": targets
        }
    

