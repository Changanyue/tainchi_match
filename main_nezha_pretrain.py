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

#wordpiece bpe  21128  20000

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
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        return prediction_scores

        # Finetune
        # pooled_output = outputs[1]
        # pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)
        # return logits



# Load dataset
train = pd.read_csv('./track1_data/pretrain.tsv', sep=',', names=['id', 'text', 'label'], dtype=str)

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
word_dict = get_dict(train)
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
        # print(self.texts.iloc[idx])
        _, t, label = self.texts.iloc[idx]  # t就是得到的句子

        if t[-1] == '|':
            t = t[1:-1]
        else:
            t = t[1:]
        return t

    # 每个片段长度n（最大为3），根据概率公式计算得到。比如1-gram、2-gram、3-gram的的概率分别为6/11、3/11、2/11
    # Masked-ngram-LM reference: albert https://blog.csdn.net/weixin_37947156/article/details/101529943
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
            if self.test_flag == False and flag == False and prob < 0.15:  # 15%的词被mask
                prob /= 0.15
                if prob < 0.8:   # 被mask的词中，80%用[MASK]token代替
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
                        output_label.append(self.vocab.get(token, self.vocab['[UNK]']))  #得到token对应的id
                        # get()函数的第二个参数：字典中不包含该键时，get函数函数返回第二个参数
                        i += 1

                    flag = True   # 防止连续mask,不然可能整个句子都被mask了
                elif prob < 0.9:   # 10%的词用其他随机一个词代替
                    token = str(random.randrange(5, len(self.vocab))) 
                    tokens[i] = self.vocab.get(token, self.vocab['[UNK]'])
                    #print(token, tokens[i])
                    output_label.append(self.vocab.get(token, self.vocab['[UNK]']))
                    
                    flag = False
                    i += 1
                else:   # 10%的词不变
                    # -100就表示不对该词进行mask
                    token = tokens[i]
                    tokens[i] = self.vocab.get(token, self.vocab['[UNK]'])
                    output_label.append(-100)

                    flag = False
                    i += 1
            else:
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
        # label_list = self.texts.label.iloc[idx][1:].split()

        # Construct bert input
        t = [self.vocab['[CLS]']] + t_random + [self.vocab['[SEP]']]

        #t_label = [self.vocab['[CLS]']] + t_label + [self.vocab['[SEP]']]
        t_label = [-100] + t_label + [self.vocab['[SEP]']]

        bert_input = t[:self.seq_len]
        bert_label = t_label[:self.seq_len]
        #print(t, t_label)

        # Padding
        padding = [self.vocab['[PAD]'] for _ in range(self.seq_len - len(bert_input))]
        padding_label = [-100 for _ in range(self.seq_len - len(bert_input))]
        attention_mask = len(bert_input) * [1] + len(padding) * [0]

        bert_input.extend(padding)
        bert_label.extend(padding_label)

        output = {"input_ids": np.array(bert_input),
                  'attention_mask': np.array(attention_mask),
                  "bert_label": np.array(bert_label)}

        return output


pretrain_dataset = BERTDataset(train, vocab_dict, False, 128)

BATCH_SIZE = 16
train_loader = DataLoader(pretrain_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Construct model, loss function, optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'

criterion  = nn.CrossEntropyLoss()

config = NeZhaConfig.from_pretrained('pretrain/nezha_config.json', num_labels=17)
model = NewNeZha(config)
#model.load_state_dict(torch.load('pretrain/model_nezha_large_pre_270_0.591.pth'))
model = model.to(device)
#print(model)

# LookAhead优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-7, weight_decay=1e-8)
optimizer = Lookahead(optimizer=optimizer,k=5,alpha=0.5)

# fgm
fgm = FGM(model)
K = 3

# Start training
NUM_EPOCHS = 300
best_auc = 0.
for epoch in range(NUM_EPOCHS):
    losses = []
    model.train()
    for data in tqdm(train_loader):
        optimizer.zero_grad()

        labels = data['bert_label'].to(device).long()

        outputs = model(input_ids=data['input_ids'].to(device).long(), 
                        attention_mask=data['attention_mask'].to(device).long())
        #print(outputs.shape, labels.shape)

        mask = (labels != -100)
        loss = criterion(outputs[mask].view(-1, len(vocab_dict)), labels[mask].view(-1))
        losses.append(loss.cpu().detach().numpy())

        loss.backward()

        optimizer.step()
        model.zero_grad()

        tqdm.write(f'epoch:{epoch} train loss:{np.mean(losses):.3f}')

    if np.mean(losses) < 0.35 or epoch % 10 == 0:
        torch.save(model.state_dict(), 'model_nezha_{}_{:.3f}.pth'.format(epoch, np.mean(losses)))

