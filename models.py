import torch
import transformers


class SegmentationModel(torch.nn.Module):
    def __init__(self, model_name='gpt2', model_type='gpt2', task_type='classification'):
        super().__init__()
        self.embedding_model = transformers.AutoModel.from_pretrained(model_name, cache_dir='/data/.cache/huggingface/transformers')
        self.model_type = model_type
        self.task_type = task_type
        if task_type == 'classification':
           self.num_label = 2
           self.loss_fn = torch.nn.CrossEntropyLoss()#reduction='none')
        elif task_type == 'regression':
           self.num_label = 1
           self.loss_fn = torch.nn.MSELoss()
        
        if model_type=='gpt2':
            self.linear0 = torch.nn.Linear(768, 64)
        else:
            for param in self.embedding_model.parameters():
                param.requires_grad = False
            if model_type=='lstm':
                self.rnn = torch.nn.LSTM(input_size=768, hidden_size=64, num_layers=1, batch_first=True, bidirectional=False)
            elif model_type=='gru':
                self.rnn = torch.nn.GRU(input_size=768, hidden_size=64, batch_first=True, bidirectional=False)
        self.linear1 = torch.nn.Linear(64, self.num_label, bias=False)
        self.dropout = torch.nn.Dropout(0.2)
        
    def forward(self, input_ids, attention_mask, labels=None):
        output = self.embedding_model(input_ids, attention_mask=attention_mask).last_hidden_state
        if self.model_type=='gpt2':
            logits = torch.nn.functional.relu(self.linear0(output))
            logits = self.linear1(self.dropout(logits)).squeeze()
        else:
            logits = self.rnn(output)[0].squeeze()
            logits = self.linear1(self.dropout(torch.nn.functional.relu(logits)))

        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.num_label), labels.view(-1))
            #print(f'loss_shape: {loss.shape}, attention_mask_shape: {attention_mask.shape}, loss_sq: {loss.squeeze().shape}, attention_mask_sq: {attention_mask.squeeze().shape}')
            #loss = (loss * attention_mask.view(-1)).sum()
            #loss = loss.sum() / attention_mask.view(-1).sum()
            return {'loss': loss, 'logits': logits}
        else:
            return {'logits': logits}
        
        # logits = torch.nn.functional.relu(self.linear1(output.last_hidden_state))
        # logits = self.linear2(self.dropout(logits)).squeeze()
        # if labels is not None:
        #     loss = self.loss_fn(logits.view(-1, self.num_label), labels.view(-1))
        #     return {'loss': loss, 'logits':logits}
        # else:
        #     return {'logits': logits}