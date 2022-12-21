import torch
import numpy as np
import re
#from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score, average_precision_score

class EarlyStopping:
	def __init__(self, patience=7, verbose=True, delta=0, path='./checkpoint.pt'):
		self.patience = patience
		self.verbose = verbose
		self.counter = 0
		self.best_score = None
		self.early_stop = False
		self.val_loss_min = float("inf")
		self.delta = delta
		self.path = path

	def __call__(self, val_loss, model):
		if self.patience > 0:
			score = -val_loss

			if self.best_score is None:
				self.best_score = score
				self.save_checkpoint(val_loss, model)
			elif score < self.best_score + self.delta:
				self.counter += 1
				print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
				if self.counter >= self.patience:
					self.early_stop = True
			else:
				self.best_score = score
				self.save_checkpoint(val_loss, model)
				self.counter = 0

	def save_checkpoint(self, val_loss, model):
		if self.verbose:
			print("")
			print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
		torch.save(model.state_dict(), self.path)
		self.val_loss_min = val_loss
  
def set_seed(SEED):
	torch.manual_seed(SEED)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(SEED)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
	np.random.seed(SEED)
    
def check_token_stats(data, tokenizer):
	turn_list, tok_list, tok_avg_list = [], [], []
	for each_split in ['train', 'validation', 'test']:
		for each_data in data[each_split]:
			if data[each_split].info.builder_name == 'multi_woz_v22':
				utt = each_data['turns']['utterance']
			elif data[each_split].info.builder_name == 'daily_dialog':
				utt = each_data['dialog']
			each_utt = [re.sub("[^ +a-zA-Z0-9]+", "", x) for x in utt]
			each_utt = [x.lower().strip() for x in each_utt]
			turn_len = len(each_utt)
			tok = np.array(sum(tokenizer(each_utt)['attention_mask'], []))
			turn_list.append(turn_len)
			tok_list.append(tok.sum())
			tok_avg_list.append(tok.sum()/turn_len)
	print(f'tokens num -- min: {np.array(tok_list).min()}, max: {np.array(tok_list).max()}, mean: {np.array(tok_list).mean()}, std: {np.array(tok_list).std()}')

def test_prediction(model, test_loader, label_name, device):
	model.eval()
	logit_test = [] # torch.zeros(0)
	pred_test = [] # torch.zeros(0)
	real_test = [] # torch.zeros(0)
	with torch.no_grad():	
		for each_batch in test_loader:
			input_ids = each_batch['input_ids'].to(device)
			attention_mask = each_batch['attention_mask'].to(device)
			out = model(input_ids, attention_mask=attention_mask)

			test_pred = torch.argmax(out['logits'], dim=-1).cpu().squeeze()
			test_real = each_batch[label_name].cpu().squeeze()
			ind = attention_mask.cpu().sum().item()
			logit_test.append(out['logits'].cpu())
			pred_test.append(test_pred[:ind])
			real_test.append(test_real[:ind])
		logit_test, pred_test, real_test = torch.stack(logit_test).numpy(), torch.cat(pred_test).numpy(), torch.cat(real_test).numpy()
		test_recall = ((pred_test==1) & (real_test==1)).sum() / (real_test==1).sum()
		test_precision = ((pred_test==1) & (real_test==1)).sum() / (pred_test==1).sum()
		test_f1 = 2 * test_precision * test_recall / (test_precision + test_recall)
	print(f"recall: {test_recall}, precision: {test_precision}, f1-score: {test_f1}")
	return logit_test, pred_test, real_test, test_recall, test_precision, test_f1
 
 
def extract_sample(NUM, test_logit, test_dataset, tokenizer, TASK_TYPE='classification'):
	end_count = 0
	sample, sample_logits = test_dataset[NUM], test_logit[NUM]
	print("[Predicted Sample] ...")
	for each_sample_id, each_sample_logit in zip(torch.tensor(sample['input_ids']), torch.tensor(sample_logits.squeeze())):
        
		print(tokenizer.decode(each_sample_id), end= " ")
		if TASK_TYPE=='classification':
			if each_sample_logit.argmax() ==1:   # term with argmax==1, without threshold
				print()
				print(f'        %Prob% : {torch.sigmoid(each_sample_logit[1]):.4f}')
				print(f'        %Logit% : {each_sample_logit[1].item()}')
		elif TASK_TYPE=='regression':
			if torch.sigmoid(each_sample_logit)>=0.6:   # term with threshold equal to 0.6
				print()
				print(f'        %Prob% : {torch.sigmoid(each_sample_logit):.4f}')
				print(f'        %Logit% : {each_sample_logit.item()}')
		if each_sample_id==50256: # eos
			end_count+=1
		else:
			end_count=0
		if end_count>5:
			break

    