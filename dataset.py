import torch
import datasets
import re

class DialogDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length=None):
        
        #self.data = datasets.load_dataset(path=data_name) # 'multi_woz_v22', 'daily_dialog'
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = self.tokenizer.model_max_length if max_length is None else max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        output = {}
        label_reg = torch.zeros(0)
        if self.data.info.builder_name == 'multi_woz_v22':
            utterances = self.data[index]['turns']['utterance']
        elif self.data.info.builder_name == 'daily_dialog':
            utterances = self.data[index]['dialog']
        
        each_utt = [re.sub("[^ +a-zA-Z0-9]+", "", x) for x in utterances]
        each_utt = [x.lower().strip() for x in each_utt if x.strip()]
        label = self.tokenizer(each_utt)['attention_mask']
        for i, each_label in enumerate(label):
            # classification label
            label[i][-1] = 0
            # regression label
            label_reg = torch.cat([label_reg, torch.arange(0, 0.9, 1/len(each_label))], dim=-1)
            label_reg[-1] = 1.0 # regression
        label_cls = torch.tensor(sum(label, []), dtype=torch.long)[:self.max_length]
        label_cls = (~label_cls.bool()).float()
        label_reg = label_reg[:self.max_length]
        if len(label_reg) < self.max_length: # assign padding token label
            label_reg = torch.cat([label_reg, torch.zeros(self.max_length - len(label_reg))])
            label_cls = torch.cat([label_cls, torch.zeros(self.max_length - len(label_cls))])
        token_dict = self.tokenizer(' '.join(each_utt), truncation=True, max_length=self.max_length, padding="max_length", return_tensors='pt')
        output['input_ids'], output['attention_mask'] = token_dict['input_ids'].squeeze(), token_dict['attention_mask'].squeeze()
        output['label_reg'] = label_reg
        output['label_cls'] = label_cls.type(torch.LongTensor)
        return output
