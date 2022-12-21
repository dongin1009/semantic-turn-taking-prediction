import datasets
import transformers
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import SegmentationModel
import argparse
from dataset import DialogDataset
from utils import EarlyStopping, test_prediction, set_seed

device = torch.device("cuda")

def train_model(args, model, optimizer, train_loader, valid_loader):
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=args.path)
    
    for each_epoch in range(1, args.epochs+1):
        train_loss = 0.0
        model.train()
        for each_batch in tqdm(train_loader):
            optimizer.zero_grad()
            input_ids = each_batch['input_ids'].to(device)
            attention_mask = each_batch['attention_mask'].to(device)
            labels = each_batch[args.label_name].to(device)
            out = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = out['loss']#out.loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        with torch.no_grad():
            model.eval()
            valid_loss = 0.0
            for each_batch in tqdm(valid_loader):
                input_ids = each_batch['input_ids'].to(device)
                attention_mask = each_batch['attention_mask'].to(device)
                labels = each_batch[args.label_name].to(device)
                out = model(input_ids, attention_mask=attention_mask, labels=labels)
                valid_loss += out['loss'].item()#out.loss.item()
            print(f'Epoch {each_epoch}: Train Loss: {train_loss / len(train_loader):.4f}, Valid Loss: {valid_loss / len(valid_loader):.4f}')

            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print(f"Early stopping at {each_epoch-early_stopping.patience} !")
                break

def evaluate(args, model, test_loader):
    model.load_state_dict(torch.load(args.path))
    _, _, _, test_recall, test_precision, test_f1 = test_prediction(model, test_loader, args.label_name, device) # test_logit, test_pred, test_real, test_recall, test_precision, test_f1
    print(f"recall: {test_recall}, precision: {test_precision}, f1-score: {test_f1}")

def main(args):
    tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>', return_value='pt')
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset =  DialogDataset(datasets.load_dataset(path=args.data_name, split='train'), tokenizer)
    valid_dataset = DialogDataset(datasets.load_dataset(path=args.data_name, split='validation'), tokenizer)
    test_dataset = DialogDataset(datasets.load_dataset(path=args.data_name, split='test'), tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    args.path = f'saved/{args.data_name}_{args.model_type}_{args.task_type}_model{args.lr}.pt'
    model = SegmentationModel(model_name='gpt2', model_type=args.model_type, task_type=args.task_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    args.label_name = 'label_cls' if args.task_type == 'classification' else 'label_reg'
    
    train_model(args, model, optimizer, train_loader, valid_loader)
    evaluate(args, model, test_loader)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="gpt2", choices=['gpt2', 'lstm', 'gru'], help="model name")
    parser.add_argument("--data_name", default="daily_dialog", choices=['multi_woz_v22', 'daily_dialog'], help="dataset name")
    parser.add_argument("--epochs", type=int, default=500, help="epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--patience", type=int, default=5, help="patience")

    parser.add_argument("--task_type", default="classification", choices=['classification', 'regression'], help="task type")
    parser.add_argument("--gpu", default="0", help="gpu id")
    parser.add_argument("--seed", default=333, type=int, help="random seed")
    
    args = parser.parse_args()
    if args.gpu != '0':
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    os.environ['PYTHONHASHSEED'] = str(args.seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    set_seed(args.seed)

    main(args)