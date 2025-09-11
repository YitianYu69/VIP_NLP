import os
# os.environ['HF_HOME'] = "/storage/coda1/p-cwiese7/0/yyu496/.cache/huggingface" # Use your path
# os.environ["NCCL_BLOCKING_WAIT"] = "1" # Use your path
# os.environ["NCCL_DEBUG"] = "INFO" Use your path

import pandas as pd
import os

import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig, get_cosine_schedule_with_warmup
from evaluate import load
from peft import LoraConfig, get_peft_model, TaskType

from tqdm import tqdm

from sklearn.model_selection import train_test_split

def create_df(path):
    # Expand the user path if needed
    path = os.path.expanduser(path)
    df = pd.read_csv(path)
    df = df.where(df['translation'] != 'No translation').dropna()
    df['sentence'] = df['sentence'].apply(lambda x: x.strip('""'))
    df.rename(columns={'sentence': 'Sentence', 'translation': 'Translation'}, inplace=True)
    return df


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer):
        super().__init__()
        self.dataframe = dataframe
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        source_text = self.dataframe.Sentence.iloc[idx]
        input_prompt = f'translate english to english: {source_text}'
        target_text = self.dataframe.Translation.iloc[idx]
        
        tokenized_original_sentence = self.tokenizer(
            input_prompt,
            padding='max_length',
            max_length=128,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt')
        
        tokenized_translated_sentence = self.tokenizer(
            target_text,
            padding='max_length',
            max_length=128,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt')
            
        target_ids = tokenized_translated_sentence["input_ids"].squeeze(0)
        labels = target_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return (tokenized_original_sentence['input_ids'].squeeze(0), 
                tokenized_original_sentence['attention_mask'].squeeze(0), 
                labels)



class TorchDatasetAndDataLoader():
    def __init__(self, dataframe, tokenizer, torch_dataset, train_test_split, data_type='binary'):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.torch_dataset = torch_dataset
        self.train_test_split = train_test_split
        
        if data_type == 'binary':
            self.dataframe['Class'] = self.dataframe.discipline.apply(
                lambda x: 'Contains Jargons' if x != 'No Discipline' else 'No Jargons')
        elif data_type == 'multiclass':
            self.dataframe.rename(columns={'discipline': 'Class'}, inplace=True)
        else:
            raise ValueError('Please input either binary or multiclass for data_type')
        
    
    def get_torch_dataset(self, split=None):
        if split:
            df_train, temp = self.train_test_split(self.dataframe, test_size=0.2, random_state=42)
            df_valid, df_test = self.train_test_split(temp, test_size=0.5, random_state=42)
            
            return {
                'train': self.torch_dataset(df_train, self.tokenizer),
                'valid': self.torch_dataset(df_valid, self.tokenizer),
                'test': self.torch_dataset(df_test, self.tokenizer)
            }
        else:
            df = self.dataframe
            return self.torch_dataset(df, self.tokenizer)
        
        
    def get_torch_dataloader(
        self, rank, world_size, shuffle, batch_size, num_workers, pin_memory, persistent_workers, split=None):
        
        if split:
            custom_torch_dataset = self.get_torch_dataset(split=split)
            
            train_sampler = DistributedSampler(custom_torch_dataset['train'], num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=True)
            valid_sampler = DistributedSampler(custom_torch_dataset['valid'], num_replicas=world_size, rank=rank, drop_last=True)
            test_sampler = DistributedSampler(custom_torch_dataset['test'], num_replicas=world_size, rank=rank, drop_last=True)
            
            return {
                'train': torch.utils.data.DataLoader(
                    custom_torch_dataset['train'],
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=persistent_workers,
                    sampler=train_sampler,
                    prefetch_factor=4),
                'valid': torch.utils.data.DataLoader(
                    custom_torch_dataset['valid'],
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=persistent_workers,
                    sampler=valid_sampler,
                    prefetch_factor=4),
                'test': torch.utils.data.DataLoader(
                    custom_torch_dataset['test'],
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=persistent_workers,
                    sampler=test_sampler,
                    prefetch_factor=4)
            }
        else:
            custom_torch_dataset = self.get_torch_dataset()
            return torch.utils.data.DataLoader(
                custom_torch_dataset, 
                shuffle=shuffle, 
                batch_size=batch_size, 
                num_workers=num_workers, 
                pin_memory=pin_memory, 
                persistent_workers=persistent_workers)
        


class EarlyStop():
    def __init__(self, patience, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.early_stopping = False

        # best_loss_score = negative of val_loss (so higher = better)
        self.best_loss_score = float('-inf')
        # best_sacrebleu_score = direct BLEU (so higher = better)
        self.best_sacrebleu_score = float('-inf')

    def __call__(self, val_loss, sacrebleu_score, model):
        loss_score = -val_loss

        if self.best_loss_score == float('-inf'):
            # First time: always record as best
            self.best_loss_score = loss_score
            self.best_sacrebleu_score = sacrebleu_score
            if self.verbose:
                print(f"Initial loss = {val_loss:.4f}, BLEU = {sacrebleu_score:.2f}")
        elif loss_score < self.best_loss_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'Patience Count: {self.counter}/{self.patience}')
            if sacrebleu_score > self.best_sacrebleu_score:
                self.best_sacrebleu_score = sacrebleu_score
                self.save_checkpoint(model)
                if self.verbose:
                    print(f"New best BLEU = {sacrebleu_score:.2f}")
            if self.counter >= self.patience:
                self.early_stopping = True
        else:
            if self.verbose:
                print(f'Validation loss improved: {val_loss:.4f}')
            self.counter = 0
            self.best_loss_score = loss_score

            if sacrebleu_score > self.best_sacrebleu_score:
                self.best_sacrebleu_score = sacrebleu_score
                self.save_checkpoint(model)
                if self.verbose:
                    print(f"New best BLEU = {sacrebleu_score:.2f}")

    def save_checkpoint(self, model):
        torch.save(model.module.state_dict(), '/storage/coda1/p-cwiese7/0/yyu496/Model/T5_small_checkpoint.pt')
        if self.verbose:
            print("Checkpoint saved.")


def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, gradient_accumulation_steps, metrics, rank, epoch, num_epochs):
    model.train()
    train_loss = 0
    
    device = torch.device(f"cuda:{rank}")
    progress_bar = tqdm(dataloader, total=len(dataloader), desc=f'Epoch: {epoch}/{num_epochs}')
    step = 0
    for batch in progress_bar:
        feature_input_ids, feature_attention_mask, target_input_ids = batch
        feature_input_ids = feature_input_ids.to(device)
        feature_attention_mask = feature_attention_mask.to(device)
        target_input_ids = target_input_ids.to(device)
        
        with torch.amp.autocast(enabled=False, device_type='cuda'):
            outputs = model(
                input_ids=feature_input_ids,
                attention_mask=feature_attention_mask,
                labels=target_input_ids)
            loss = outputs.loss / gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            train_loss += loss.item()
            step += 1
            
            if step % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

    return train_loss / len(dataloader)
            

def validation(model, dataloader, tokenizer, metrics, rank):
    model.eval()
    device = torch.device(f"cuda:{rank}")
    valid_loss = 0
    predictions = []
    references = []
    
    progress_bar = tqdm(dataloader, total=len(dataloader), desc='Validation: ')
    with torch.no_grad():
        with torch.amp.autocast(enabled=False, device_type='cuda'):
            for batch in progress_bar:
                feature_input_ids, feature_attention_mask, target_input_ids = batch
                feature_input_ids = feature_input_ids.to(device)
                feature_attention_mask = feature_attention_mask.to(device)
                target_input_ids = target_input_ids.to(device)
                
                outputs = model(
                    input_ids=feature_input_ids,
                    attention_mask=feature_attention_mask,
                    labels=target_input_ids
                )
                loss = outputs.loss
                valid_loss += loss.item()
                
                generated_ids = model.module.generate(input_ids=feature_input_ids,
                                               attention_mask=feature_attention_mask)
            
            
                labels = target_input_ids.clone()
                labels[labels == -100] = tokenizer.pad_token_id
                decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                decoded_ref = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
                predictions.extend(decoded_preds)
                references.extend(decoded_ref)
    
    valid_loss = valid_loss / len(dataloader)
    sacrebleu_score = metrics['sacrebleu'].compute(predictions=predictions, references=[[ref] for ref in references])['score']
    meteor_score = metrics['meteor'].compute(predictions=predictions, references=references)['meteor']
    rouge_score = metrics['rouge'].compute(predictions=predictions, references=references)
    
    return {
        'valid_loss' : valid_loss,
        'sacrebleu': sacrebleu_score,
        'meteor': meteor_score,
        'rouge-1': rouge_score['rouge1'],
        'rouge-2': rouge_score['rouge2'],
        'rouge-L': rouge_score['rougeL']
    }


def testing(model, dataloader, tokenizer, metrics, rank):
    model.eval()
    device = torch.device(f"cuda:{rank}")
    predictions = []
    references = []
    
    progress_bar = tqdm(dataloader, total=len(dataloader), desc='Testing: ')
    with torch.no_grad():
        with torch.amp.autocast(enabled=False, device_type='cuda'):
            for batch in progress_bar:
                feature_input_ids, feature_attention_mask, target_input_ids = batch
                feature_input_ids = feature_input_ids.to(device)
                feature_attention_mask = feature_attention_mask.to(device)
            
                generated_ids = model.module.generate(input_ids=feature_input_ids,
                                               attention_mask=feature_attention_mask)
            
                
                labels = target_input_ids.clone()
                labels[labels == -100] = tokenizer.pad_token_id
                
                decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                decoded_ref = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
                predictions.extend(decoded_preds)
                references.extend(decoded_ref)
            
    sacrebleu_score = metrics['sacrebleu'].compute(predictions=predictions, references=[[ref] for ref in references])['score']
    meteor_score = metrics['meteor'].compute(predictions=predictions, references=references)['meteor']
    rouge_score = metrics['rouge'].compute(predictions=predictions, references=references)
    
    return {
        'sacrebleu': sacrebleu_score,
        'meteor': meteor_score,
        'rouge-1': rouge_score['rouge1'],
        'rouge-2': rouge_score['rouge2'],
        'rouge-L': rouge_score['rougeL']
    }


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    
def cleanup():
    dist.destroy_process_group()


def main(rank, model_type, model, tokenizer, world_size, num_epochs, batch_size, patience):
    
    # --------------
    # Initialize DDP
    # --------------
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    
    # -----
    # Model
    # -----
    
    #peft_config = LoraConfig(
        #task_type=TaskType.SEQ_2_SEQ_LM,  # For sequence-to-sequence models.
        #r=8,                             # Low-rank dimension.
        #lora_alpha=32,                   # Scaling factor for LoRA updates.
        #lora_dropout=0.1,                # Dropout rate applied to LoRA layers.
        #target_modules=["q", "v"],       # Names of target modules in T5's attention.
    #)

    device = torch.device(f"cuda:{rank}")
    model.to(device)
    #model = get_peft_model(model, peft_config)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    # -------
    # Dataset
    # -------
    df = create_df('~/p-cwiese7-0/data/cleaned_final_jargon_dataset.csv')
    torchDataLoader = TorchDatasetAndDataLoader(df, tokenizer, TextDataset, train_test_split).get_torch_dataloader(
        rank=rank, world_size=world_size, shuffle=True, batch_size=batch_size, 
        num_workers=4, pin_memory=True, persistent_workers=True, split=True)
    training_data = torchDataLoader['train']
    validation_data = torchDataLoader['valid']
    testing_data = torchDataLoader['test']
    
    # -------
    # Metrics
    # -------
    metrics = {
        'sacrebleu': load('sacrebleu'),
        'meteor': load('meteor'),
        'rouge': load('rouge')
    }
    
    # ----------------------------------
    # Scheduler, optimizer, and scaler
    # ----------------------------------
    num_training_steps = len(training_data) * num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)
    
    scaler = torch.amp.GradScaler()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)
    
    # ---------
    # EarlyStop (Optional: call it after validation if needed)
    # ---------
    early_stop = EarlyStop(patience=patience, verbose=True)
    
    # -------------
    # Training loop
    # -------------
    try:
        if model_type == 'train':
            for epoch in range(1, num_epochs + 1):
                training_data.sampler.set_epoch(epoch)
                training_loss = train_one_epoch(model, training_data, optimizer, scheduler, scaler, 8, metrics, rank, epoch, num_epochs)
                
                validation_data.sampler.set_epoch(epoch)
                validation_metrics = validation(model, validation_data, tokenizer, metrics, rank)
                
                if rank == 0:
                    print(f"Epoch {epoch}/{num_epochs}")
                    print(f"Training Loss: {training_loss:.7f}")
                    print(f"Validation:\n  SacreBLEU Score: {validation_metrics['sacrebleu']} || Meteor Score: {validation_metrics['meteor']}\n"
                        f"  Rouge-1: {validation_metrics['rouge-1']} || Rouge-2: {validation_metrics['rouge-2']}\n"
                        f"  Rouge-L: {validation_metrics['rouge-L']}")
                
                # DDP Early Stop
                valid_loss_local = validation_metrics['valid_loss']
                valid_loss_tensor = torch.tensor([valid_loss_local], device=device)
                
                dist.all_reduce(valid_loss_tensor, op=dist.ReduceOp.SUM)
                valid_loss_global = valid_loss_tensor.item() / world_size
                
                if rank == 0:
                    early_stop(valid_loss_global, validation_metrics['sacrebleu'], model)
                    stop_now = early_stop.early_stopping
                else:
                    stop_now = False
                    
                stop_now_tensor = torch.tensor([1 if stop_now else 0], device=device)
                dist.broadcast(stop_now_tensor, src=0)
                stop_now = (stop_now_tensor.item() == 1)
                
                if stop_now:
                    break

        else:
            state_dict = torch.load('/storage/coda1/p-cwiese7/0/yyu496/Model/T5_small_checkpoint.pt', weights_only=True) # Use your path
            new_state_dict = {"module." + k: v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
            testing_metrics = testing(model, testing_data, tokenizer, metrics, rank)
            print(f"Testing:\n  SacreBLEU Score: {testing_metrics['sacrebleu']} || Meteor Score: {testing_metrics['meteor']}\n"
                  f"  Rouge-1: {testing_metrics['rouge-1']} || Rouge-2: {testing_metrics['rouge-2']}\n"
                  f"  Rouge-L: {testing_metrics['rouge-L']}")
    
    finally:
        cleanup()


if __name__ == '__main__':
    token = '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' # Use your key
    model = AutoModelForSeq2SeqLM.from_pretrained('google-t5/t5-small',
                                                  use_auth_token=token, 
                                                  trust_remote_code=True)
    #model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained('google-t5/t5-small', token=token)
    #if tokenizer.pad_token is None:
        #tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    world_size = torch.cuda.device_count()

    model_type = 'test'
    num_epochs = 999
    batch_size = 64
    patience = 50
    
    mp.spawn(main,
             args=(model_type, model, tokenizer, world_size, num_epochs, batch_size, patience),
             nprocs=world_size)
