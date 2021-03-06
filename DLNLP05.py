# coding:utf-8
import os
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import jieba
from tqdm import tqdm

class Dictionary(object):

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def __len__(self):
        return len(self.word2idx)

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1


class Corpus(object):

    def __init__(self):
        self.dictionary = Dictionary()

    def get_data(self, path, batch_size=20):
        # step 1
        with open(path, 'r', encoding="utf-8") as f:
            tokens = 0
            for line in f.readlines():
                words = jieba.lcut(line) + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # step 2
        ids = torch.LongTensor(tokens)
        token = 0
        with open(path, 'r', encoding="utf-8") as f:
            for line in f.readlines():
                words = jieba.lcut(line) + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        # step 3
        num_batches = ids.size(0) // batch_size
        ids = ids[:num_batches * batch_size]
        ids = ids.view(batch_size, -1)
        return ids

    def get_data_txt(self,input_txt,batch_size = 20):
        # step 1
        words = jieba.lcut(input_txt) + ['<eos>']
        tokens = len(words)
        for word in words:
            self.dictionary.add_word(word)

        # step 2
        ids = torch.LongTensor(tokens)
        token = 0
        words = jieba.lcut(input_txt) + ['<eos>']
        for word in words:
            ids[token] = self.dictionary.word2idx[word]
            token += 1

        # step 3
        num_batches = ids.size(0) // batch_size
        ids = ids[:num_batches * batch_size]
        ids = ids.view(batch_size, -1)
        return ids




class LSTMmodel(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(LSTMmodel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h):
        x = self.embed(x)
        out, (h, c) = self.lstm(x, h)
        out = out.reshape(out.size(0) * out.size(1), out.size(2))
        out = self.linear(out)

        return out, (h, c)




def mytrain():

    for epoch in range(num_epochs):

        states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
                  torch.zeros(num_layers, batch_size, hidden_size).to(device))

        for i in tqdm(range(0, ids.size(1) - seq_length, seq_length)):
            inputs = ids[:, i:i+seq_length].to(device)
            targets = ids[:, (i+1):(i+1)+seq_length].to(device)
            # inputs???targets???????????????x???y???

            states = [state.detach() for state in states]
            # ??????detach????????????????????????????????????

            outputs, states = model(inputs, states)
            # ???inputs???states??????model????????????????????????????????????outputs???????????????states

            loss = cost(outputs, targets.reshape(-1))
            # ????????????outputs????????????targets??????cost???????????????????????????

            model.zero_grad()
            # ???????????????????????????????????????????????????????????????????????????????????????zero_grad??????????????????????????????

            loss.backward()

            clip_grad_norm_(model.parameters(), 0.5)
            # ???????????????????????????????????????clip_grad_norm_????????????????????? 0.5

            optimizer.step()
            # ????????????optimizer????????????


    torch.save(model,txt_path + 'mymodel.pkl')


def txt_generate(exp=1):

    if exp == 1:

        article = str()

        state = (torch.zeros(num_layers, 1, hidden_size).to(device),
                torch.zeros(num_layers, 1, hidden_size).to(device))
        # state = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
        #         torch.zeros(num_layers, batch_size, hidden_size).to(device))
        # state???????????????????????????????????????????????????(h, c)


        # prob = torch.ones(vocab_size)
        # # prob??????????????????outputs???????????????????????????????????????????????????????????????????????????????????????????????????

        # _input = torch.multinomial(prob, num_samples=1).unsqueeze(1).to(device)
        # # print('_input.shape',_input.shape)
        # # # _input????????????Python????????????input????????????????????????????????????_???????????????????????????????????????????????????????????????
        # print('_input:', _input)
        # print(int(_input))
        # print('_input.shape',_input.shape)
        # print('word:', corpus.dictionary.idx2word[int(_input

        txt_idx = corpus.dictionary.word2idx['??????']
        # print(_input) # ??????
        # print('_input.shape',_input.shape) # error
        _input = torch.tensor([[txt_idx]])

        print('_input:', _input)
        print(int(_input))
        print('_input.shape',_input.shape)
        print('word:', corpus.dictionary.idx2word[int(_input)])

        num_samples = 100
        # ?????????num_samples????????????????????????


        for i in range(num_samples):
            output, state = model(_input, state)
            # output???state???LSTMmodel??????????????????_input???state???????????????

            prob = output.exp()
            # print('prob.shape',prob.shape)
            # prob????????????????????????output????????????????????????????????????????????????
            word_id = torch.multinomial(prob, num_samples=1)

            # print('shape',word_id.shape)
            
            word_id = word_id.item()

            # print('word_id:',word_id)
            # print('type:',type(word_id))
            # word_id?????????torch_multinomial??????prob??????????????????????????????????????????????????????1(???num_samples)

            _input.fill_(word_id)
            # ????????????????????????????????????fill_???????????????????????????(word_id)??????_input??????

            word = corpus.dictionary.idx2word[word_id]
            # ??????????????????Dictionary????????????????????????(???word_id)???????????????

            word = '\n' if word == '<eos>' else word
            # ???????????????????????????????????????(???<eos>???????????????EndOfSentence)?????????????????????
            article = article + word

        print(article)
        print('len(article):',len(article))

        seg_txt = ' '.join(jieba.lcut(article))
        print(seg_txt)

    elif exp == 2:


        article = str()
        state = (torch.zeros(num_layers, 1, hidden_size).to(device),
                torch.zeros(num_layers, 1, hidden_size).to(device))

        input_txt = u'?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????'
        word_list = jieba.lcut(input_txt)
        idx_list = []
        for i in word_list:
            txt_idx = corpus.dictionary.word2idx[i]
            idx_list.append(txt_idx)

        length = len(idx_list)
        print('length',length)
        input_tensor = torch.tensor([idx_list])
        print('input_tensor.shape',input_tensor.shape)

        _input = input_tensor
        num_samples = 1

        for i in range(num_samples):
            output, state = model(_input, state)
            # output???state???LSTMmodel??????????????????_input???state???????????????

            prob = output.exp()
            # print('prob.shape',prob.shape)
            # print('prob.shape',prob.shape)
            # prob????????????????????????output????????????????????????????????????????????????
            word_id = torch.multinomial(prob, num_samples=1)

            # print('type',type(word_id))
            # print('word_id.shape',word_id.shape)
            # shape torch.Size([613, 613])

            # print(word_id)

        eos_num = 0
        for i in word_id:
            word = corpus.dictionary.idx2word[int(i)]

            if word == '<eos>' :
                word = '\n'
                eos_num += 1
            else :
                word = word
            article = article + word

        print(article)
        print('len(article):',len(article))
        print('eos_num',eos_num)

        seg_txt = ' '.join(jieba.lcut(article))
        print(seg_txt)

    else:
        print('sth wrong !')






if __name__ == '__main__':

    # seed = 0
    # torch.manual_seed(seed)
    # https://blog.csdn.net/weixin_35097346/article/details/112018664

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    txt_path = '/Users/huzikang/Desktop/jyxsqj_after_process/'

    embed_size = 128
    hidden_size = 1024
    num_layers = 1
    num_epochs = 1
    batch_size = 50
    seq_length = 30
    learning_rate = 0.001

    corpus = Corpus()
    ids = corpus.get_data(txt_path + '????????????.txt', batch_size)
    # ??????????????????
    vocab_size = len(corpus.dictionary)
    print('vocab_size',vocab_size)


    model = LSTMmodel(vocab_size, embed_size, hidden_size, num_layers).to(device)
    cost = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    if os.path.exists(txt_path+'mymodel.pkl'):
        print('------------loading model------------')
        model = torch.load(txt_path+'mymodel.pkl')
        txt_generate(2)
    else: 
        mytrain()
        txt_generate()





