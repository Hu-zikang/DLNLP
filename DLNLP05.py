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
            # inputs和targets是训练集的x和y值

            states = [state.detach() for state in states]
            # 通过detach方法，定义参数的终点位置

            outputs, states = model(inputs, states)
            # 把inputs和states传入model，得到通过模型计算出来的outputs和更新后的states

            loss = cost(outputs, targets.reshape(-1))
            # 把预测值outputs和实际值targets传入cost损失函数，计算差值

            model.zero_grad()
            # 由于参数在反馈时，梯度默认是不断积累的，所以在这里需要通过zero_grad方法，把梯度清零以下

            loss.backward()

            clip_grad_norm_(model.parameters(), 0.5)
            # 为了避免梯度爆炸的问题，用clip_grad_norm_设定参数阈值为 0.5

            optimizer.step()
            # 用优化器optimizer进行优化


    torch.save(model,txt_path + 'mymodel.pkl')


def txt_generate(exp=1):

    if exp == 1:

        article = str()

        state = (torch.zeros(num_layers, 1, hidden_size).to(device),
                torch.zeros(num_layers, 1, hidden_size).to(device))
        # state = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
        #         torch.zeros(num_layers, batch_size, hidden_size).to(device))
        # state是初始化的模型参数，相当于模型中的(h, c)


        # prob = torch.ones(vocab_size)
        # # prob对应模型中的outputs，是输入变量经过语言模型得到的输出值，相当于此时每个单词的概率分布

        # _input = torch.multinomial(prob, num_samples=1).unsqueeze(1).to(device)
        # # print('_input.shape',_input.shape)
        # # # _input，出于和Python自带函数input冲突，在变量明前加下划线_，是从字典里随机抽样一个单词，作为文章开头
        # print('_input:', _input)
        # print(int(_input))
        # print('_input.shape',_input.shape)
        # print('word:', corpus.dictionary.idx2word[int(_input

        txt_idx = corpus.dictionary.word2idx['杨过']
        # print(_input) # 整数
        # print('_input.shape',_input.shape) # error
        _input = torch.tensor([[txt_idx]])

        print('_input:', _input)
        print(int(_input))
        print('_input.shape',_input.shape)
        print('word:', corpus.dictionary.idx2word[int(_input)])

        num_samples = 100
        # 生成由num_samples个单词组成的文章


        for i in range(num_samples):
            output, state = model(_input, state)
            # output、state是LSTMmodel在接收到变量_input和state后的输出值

            prob = output.exp()
            # print('prob.shape',prob.shape)
            # prob是对上一步得到的output进行指数化，加强高概率结果的权重
            word_id = torch.multinomial(prob, num_samples=1)

            # print('shape',word_id.shape)
            
            word_id = word_id.item()

            # print('word_id:',word_id)
            # print('type:',type(word_id))
            # word_id，通过torch_multinomial，以prob为权重，对结果进行加权抽样，样本数为1(即num_samples)

            _input.fill_(word_id)
            # 为下一次运算作准备，通过fill_方法，把最新的结果(word_id)作为_input的值

            word = corpus.dictionary.idx2word[word_id]
            # 从字典映射表Dictionary里，找到当前索引(即word_id)对应的单词

            word = '\n' if word == '<eos>' else word
            # 如果获得到的单词是特殊符号(如<eos>，句尾符号EndOfSentence)，替换成换行符
            article = article + word

        print(article)
        print('len(article):',len(article))

        seg_txt = ' '.join(jieba.lcut(article))
        print(seg_txt)

    elif exp == 2:


        article = str()
        state = (torch.zeros(num_layers, 1, hidden_size).to(device),
                torch.zeros(num_layers, 1, hidden_size).to(device))

        input_txt = u'箫声中忽听得远处脚步声响，有人疾奔而来。那少女放下玉箫，走到门口，叫道：表妹！一人奔向屋前，气喘吁吁的道：表姊，那女魔头查到了我的踪迹，正一路寻来，咱们快走！杨过听话声正是陆无双，心下一喜，但随即听她说那女魔头即将追到，指的自是李莫愁，不由得暗暗吃惊，随即又想：原来这位姑娘是媳妇儿的表姊。只听那少女道：有人受了伤，在这里养伤。陆无双道：是谁？那少女道：你的救命恩人。陆无双叫道：傻蛋！他……他在这里！说著冲进门来。月光下只见她喜容满脸，叫道：傻蛋，傻蛋！你怎麽寻到了这里？这次可轮到你受伤啦。'
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
            # output、state是LSTMmodel在接收到变量_input和state后的输出值

            prob = output.exp()
            # print('prob.shape',prob.shape)
            # print('prob.shape',prob.shape)
            # prob是对上一步得到的output进行指数化，加强高概率结果的权重
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
    ids = corpus.get_data(txt_path + '神雕侠侣.txt', batch_size)
    # 获取数据位置
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





