import os
import random
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertForPreTraining, BertModel
from Lookahead.optimizer import Lookahead
from attacks import FGM, PGD


from NeZha.model.modeling_nezha import NeZhaPreTrainedModel, NeZhaModel
from NeZha.model.configuration_nezha import NeZhaConfig
from transformers.models.bert.modeling_bert import (
    BertOutput,
    BertPooler,
    BertSelfOutput,
    BertIntermediate,
    BertOnlyMLMHead,
    BertOnlyNSPHead,
    BertPreTrainingHeads,
    BERT_START_DOCSTRING,
    BERT_INPUTS_DOCSTRING,
)


SEEDS = [0, 68, 82, 1989, 75, 2021, 3033, 192, 385, 2022] 

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class NewNeZha(NeZhaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = NeZhaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        # Pretrain
        '''
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        return prediction_scores
        '''

        # Finetune
        # pooled_output = outputs[1]
        # pooled_output = self.dropout(pooled_output)
        # print(outputs[0].shape, outputs[0][:,0,:].shape)
        # cls = self.dropout(outputs[0][:,0,:])
        logits = self.classifier(outputs[0][:,0,:])
        return logits


# Load dataset
pretrain = pd.read_csv('./track1_data/pretrain.tsv', sep=',', names=['id', 'text', 'label'], dtype=str)
test = pd.read_csv('./track1_data/track1_round1_testA_20210222.csv', sep=',', names=['id', 'text', 'label'], dtype=str)

# Construct word dict
def get_dict(data):
    words_dict = defaultdict(int)
    for i in tqdm(range(data.shape[0])):
        text = data.text.iloc[i]
        if text[-1] == '|':
            text = text[1:-1].split()
        else:
            text = text[1:].split()
        for c in text:
            words_dict[c] += 1
    return words_dict
word_dict = get_dict(pretrain)
print('original vocab size:', len(word_dict))

# Save
special_tokens = ["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]"]
vocab = special_tokens + list(word_dict.keys())
vocab_dict = {}
for k,v in enumerate(vocab):
    vocab_dict[v] = k

# Define custom dataset
class BERTDataset(Dataset):
    def __init__(self, corpus, vocab, test_flag, seq_len=128): 
        self.vocab = vocab
        self.seq_len = seq_len
        self.texts = corpus
        self.test_flag = test_flag
        self.ratio = self.pn(3)

    def __len__(self):
        return len(self.texts)

    def get_sentence(self, idx):
        #print(self.texts.iloc[idx])
        _, t, label = self.texts.iloc[idx]

        if t[-1] == '|':
            t = t[1:-1]
        else:
            t = t[1:]
        return t

    # 每个片段长度n（最大为3），根据概率公式计算得到。比如1-gram、2-gram、3-gram的的概率分别为6/11、3/11、2/11
    # reference: albert https://blog.csdn.net/weixin_37947156/article/details/101529943
    def pn(self, n):
        denominator = 0
        for i in range(1, n+1):
             denominator += 1.0 / i
        ratio = []
        for i in range(1, n+1):
            ratio.append(1.0 / i / denominator)

        return ratio

    def random_word(self, sentence):
        tokens = sentence.split()
        k = len(tokens)

        i = 0
        flag = False
        output_label = []
        while i < k:
            prob = random.random()
            if self.test_flag == False and flag == False and prob < 0.15:
                prob /= 0.15
                if prob < 0.8:
                    prob /= 0.8
                    if prob <= self.ratio[0]:
                        n_gram = 1
                    elif prob <= self.ratio[0] + self.ratio[1]:
                        n_gram = 2
                    else:
                        n_gram = 3

                    j = i + n_gram
                    while i < j and i < k:
                        token = tokens[i]
                        tokens[i] = self.vocab['[MASK]']
                        output_label.append(self.vocab.get(token, self.vocab['[UNK]']))
                        i += 1

                    flag = True
                elif prob < 0.9:
                    token = str(random.randrange(5, len(self.vocab))) 
                    tokens[i] = self.vocab.get(token, self.vocab['[UNK]'])
                    output_label.append(self.vocab.get(token, self.vocab['[UNK]']))
                    
                    flag = False
                    i += 1
                else:
                    # Tokens with indices set to -100 are ignored (masked)
                    token = tokens[i]
                    tokens[i] = self.vocab.get(token, self.vocab['[UNK]'])
                    output_label.append(-100)

                    flag = False
                    i += 1
            else:
                # Tokens with indices set to -100 are ignored (masked)
                token = tokens[i]
                tokens[i] = self.vocab.get(token, self.vocab['[UNK]'])
                output_label.append(-100)

                flag = False
                i += 1
        if len(tokens) != len(output_label):
            print('error!', k, len(tokens), len(output_label))

        return tokens, output_label

    def __getitem__(self, idx):
        # Get original input
        t = self.get_sentence(idx)

        # 1-gram to 3-gram
        t_random, t_label = self.random_word(t)

        # targets(test has no targets)
        try:
            label_list = self.texts.label.iloc[idx][1:]
            if label_list != '':
                label_list = label_list.split()
            else:
                label_list = []
        except:
            label_list = []
        
        # Construct bert input
        t = [self.vocab['[CLS]']] + t_random + [self.vocab['[SEP]']]

        #t_label = [self.vocab['[CLS]']] + t_label + [self.vocab['[SEP]']]
        t_label = [-100] + t_label + [self.vocab['[SEP]']]

        bert_input = t[:self.seq_len]

        # Pretrain
        #bert_label = t_label[:self.seq_len]
        #bert_label.extend(padding_label)
        #print(t, t_label)

        # Finetune
        bert_label = [0] * 17
        for i in label_list:
            bert_label[int(i)] = 1

        # Padding
        padding = [self.vocab['[PAD]'] for _ in range(self.seq_len - len(bert_input))]
        padding_label = [-100 for _ in range(self.seq_len - len(bert_input))]
        attention_mask = len(bert_input) * [1] + len(padding) * [0]

        bert_input.extend(padding)

        output = {"input_ids": np.array(bert_input),
                  'attention_mask': np.array(attention_mask),
                  "bert_label": np.array(bert_label)}

        return output


BATCH_SIZE = 16


def evaluate(model, data_loader, device='cuda'):
    model.eval()

    losses = []
    criterion = nn.BCEWithLogitsLoss(reduction='sum')
    for data in tqdm(data_loader):
        with torch.no_grad():
            labels = data['bert_label'].to(device).float()
            outputs = model(input_ids=data['input_ids'].to(device).long(), 
                            attention_mask=data['attention_mask'].to(device).long())

            preds = outputs
            loss = criterion(preds, labels)
            losses.append(loss)

    auc_score = np.sum(losses) / (len(valid_dataset) * 17)
    print(f'valid score:{1-auc_score:.4f}')

    return 1 - auc_score

# Predict test
def predict(model, data_loader, device='cuda'):
    model.eval()

    preds_list = []
    for data in tqdm(data_loader):
        with torch.no_grad():
            outputs = model(input_ids=data['input_ids'].to(device).long(), 
                            attention_mask=data['attention_mask'].to(device).long())

            preds = outputs
            preds = F.sigmoid(preds)
            #print(preds)
            preds_list.append(preds.cpu().detach())

    preds_list = np.concatenate(preds_list)
    print('preds_list:', preds_list.shape)

    return preds_list


# Construct model, loss function, optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Start training
NUM_EPOCHS = 5
best_auc = -100
criterion = nn.BCEWithLogitsLoss()

config = NeZhaConfig.from_pretrained('pretrain/nezha_config.json', num_labels=17)

for i in range(10):
    set_seed(SEEDS[i])

    train = pd.read_csv(f'./track1_data/train_fold_{i}.tsv', sep=',', names=['id', 'text', 'label'], dtype=str)
    valid = pd.read_csv(f'./track1_data/valid_fold_{i}.tsv', sep=',', names=['id', 'text', 'label'], dtype=str)

    train_dataset = BERTDataset(train, vocab_dict, True, 128)
    valid_dataset = BERTDataset(valid, vocab_dict, True, 128)
    test_dataset = BERTDataset(test, vocab_dict, True, 128)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = NewNeZha(config)
    model.load_state_dict(torch.load('model/model_nezha_245_0.350.pth'))
    #model.load_state_dict(torch.load('model_nezha_299_0.241.pth'))
    model = model.to(device)
    #print(model)

    # LookAhead
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-7)
    optimizer = Lookahead(optimizer=optimizer,k=5,alpha=0.5)

    # fgm
    fgm = FGM(model)
    K = 3

    for epoch  in range(NUM_EPOCHS):
        losses = []
        model.train()
        for data in tqdm(train_loader):
            optimizer.zero_grad()

            labels = data['bert_label'].to(device).float()

            outputs = model(input_ids=data['input_ids'].to(device).long(), 
                            attention_mask=data['attention_mask'].to(device).long())

            #print(outputs.shape, labels.shape)
            loss = criterion(outputs, labels).mean()
            #print(loss)
            losses.append(loss.cpu().detach().numpy())

            loss.backward()

            fgm.attack(epsilon=0.1) # 在embedding上添加对抗扰动
            adv_outputs = model(input_ids=data['input_ids'].to(device).long(), 
                                attention_mask=data['attention_mask'].to(device).long())
            loss_adv = criterion(adv_outputs, labels).mean()
            loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            fgm.restore() # 恢复embedding参数

            optimizer.step()
            model.zero_grad()

            tqdm.write(f'350 fold:{i} epoch:{epoch} train loss:{np.mean(losses):.3f}')

        valid_auc = evaluate(model, valid_loader)
        if valid_auc > best_auc:
            best_auc = valid_auc
        torch.save(model.state_dict(), 'model_nezha_final_0.350_fold_{}_{}_{:.4f}_{:.4f}.pth'.format(i, epoch, valid_auc, np.mean(losses)))

        result = predict(model, test_loader)
        w = open('submit_nezha_final_0.350_fold_{}_{}_{:.4f}_{:.4f}.csv'.format(i, epoch, valid_auc, np.mean(losses)), 'w')
        cnt = 0
        with open('./data/track1_round1_testA_20210222.csv') as f:
            for line in f:
                l = line[:line.index('|')]
                l += '|,|'
                for j in range(17):
                    l += str(result[cnt,j])
                    if j != 16:
                        l += ' '
                l += '\n'
                w.write(l)
                cnt += 1
        w.close()

