from torch.utils.data import Dataset
import torch
import json
import os
import random
from datasets import load_dataset, Features, Sequence, Value
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def pre_processing_chat(conversations, add_system_ratio=0.2, rng=random):
    # Keep tool-use conversations intact.
    if any(conv.get('tools') for conv in conversations): return conversations

    SYSTEM_PROMPTS = [
        "You are a knowledgeable AI. Give the user accurate information.",
        "You are MiniMind, a small and helpful language model.",
        "You are a professional AI assistant. Give useful answers.",
        "You are MiniMind. Help the user solve their problem.",
        "You are a reliable AI. Answer accurately.",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    # Add a system message with the configured probability.
    if conversations and conversations[0].get('role') != 'system':
        if rng.random() < add_system_ratio:
            return [{'role': 'system', 'content': rng.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.2, remove_empty_think=None, rng=random):
    # Remove empty thinking tags with 80% probability.
    if '<think>\n\n</think>\n\n' in prompt_content:
        if remove_empty_think is None:
            remove_empty_think = rng.random() > empty_think_ratio
        if remove_empty_think:
            prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        tokens = self.tokenizer(str(sample['text']), add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels


class ChatDataset(Dataset):
    epoch_seed = None

    def sample_rng(self, index):
        # Each record gets the same augmentation after a restart or worker-count change.
        return random if self.epoch_seed is None else random.Random(f'{self.epoch_seed}:{index}')


class SFTDataset(ChatDataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        if max_length < 2:
            raise ValueError("SFT max_length must allow a context token and a target token")
        features = Features({'conversations': [{'role': Value('string'), 'content': Value('string'), 'reasoning_content': Value('string'), 'tools': Value('string'), 'tool_calls': Value('string')}]})
        self.samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), len(input_ids))):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        rng = self.sample_rng(index)
        conversations = pre_processing_chat(sample['conversations'] or [], rng=rng)
        prompt = self.create_chat_prompt(conversations) if conversations else ''
        ids = self.tokenizer(post_processing_chat(prompt, rng=rng), add_special_tokens=False).input_ids
        input_ids = ids[:self.max_length]
        labels = self.generate_labels(input_ids)
        if len(ids) > self.max_length and all(label == -100 for label in labels[1:]):
            # Recover the first answer and as much preceding context as fits.
            all_labels = self.generate_labels(ids)
            first = next((i for i, label in enumerate(all_labels) if label != -100), None)
            if first is not None:
                stop = first
                limit = min(len(ids), first + max(1, self.max_length - len(self.bos_id)))
                while stop < limit and all_labels[stop] != -100:
                    stop += 1
                start = max(0, stop - self.max_length)
                input_ids, labels = ids[start:stop], all_labels[start:stop]
        padding = self.max_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding
        labels += [-100] * padding
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class DPODataset(ChatDataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        self.samples = load_dataset('json', data_files=file_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        chosen = sample['chosen']  # A list of messages with role and content fields.
        rejected = sample['rejected']  # Same format as the chosen conversation.
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        remove_empty_think = self.sample_rng(index).random() > 0.2
        chosen_prompt = post_processing_chat(chosen_prompt, remove_empty_think=remove_empty_think)
        rejected_prompt = post_processing_chat(rejected_prompt, remove_empty_think=remove_empty_think)
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


class RLAIFDataset(ChatDataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024, thinking_ratio=0.5):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.thinking_ratio = thinking_ratio  # Enable thinking with the configured probability.
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations, rng=random):
        conversations = pre_processing_chat(conversations, rng=rng)
        use_thinking = rng.random() < self.thinking_ratio
        return self.tokenizer.apply_chat_template(
            conversations[:-1],
            tokenize=False,
            open_thinking=use_thinking,
            add_generation_prompt=True
        )
    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample['conversations'], self.sample_rng(index))

        return {
            'prompt': prompt,
            'answer': ""
        }

class AgentRLDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def parse_conversations(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            messages.append(message)
        return messages[:-1], tools

    def __getitem__(self, index):
        sample = self.samples[index]
        messages, tools = self.parse_conversations(sample['conversations'])
        return {'messages': messages, 'tools': tools, 'gt': sample['gt']}


if __name__ == "__main__":
    pass
