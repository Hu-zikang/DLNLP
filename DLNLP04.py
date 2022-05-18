
import jieba

import gensim
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


def is_chinese(uchar):
    if uchar >= '\u4e00' and uchar <= '\u9fa5':
        return True
    else:
        return False

def reserve_chinese(content):
    content_str = ''
    for i in content:
        if is_chinese(i):
            content_str += i
    return content_str
'''
Ref.
https://blog.csdn.net/lxx199603/article/details/98774653
'''

def reserve_chinese_list(input_list):
	out_list = []
	for i in range(len(input_list)):
		if is_chinese(input_list[i]):
			out_list.append(input_list[i])
	return out_list


def preprocessing():

	stop_words_file = open(path + "stop_words_2.txt", 'r',encoding='utf-8')
	stop_words = list()
	for line in stop_words_file.readlines():
		line = line.strip()   # 去掉每行末尾的换行符
		stop_words.append(line)
	stop_words_file.close()

	people_names_file = open(path + "金庸小说全人物2.txt", 'r',encoding='utf-8')
	people_names = list()
	for line in people_names_file.readlines():
		line = line.strip()   # 去掉每行末尾的换行符
		jieba.add_word(line)
		people_names.append(line)
	people_names_file.close()

	kongfu_file = open(path + "金庸小说全武功.txt", 'r',encoding='gb18030')
	for line in kongfu_file.readlines():
		line = line.strip()   # 去掉每行末尾的换行符
		jieba.add_word(line)
	kongfu_file.close()

	menpai_file = open(path + "金庸小说全门派.txt", 'r',encoding='gb18030')
	for line in menpai_file.readlines():
		line = line.strip()   # 去掉每行末尾的换行符
		jieba.add_word(line)
	menpai_file.close()


	reocrd_inf = []
	with open(path+'inf.txt','r',encoding='utf-8') as f:
		while True:
			lines = f.readline()
			if not lines:
				break
			else:
				if len(lines) != 0:
					reocrd_inf.append(lines.strip('\n'))
				else:
					pass
	# print(reocrd_inf)
	# 上述部分读取inf.txt记录的小说标题信息

	txt_title = reocrd_inf[0].split(',')
	# print(txt_title)


	new_file = open(path+'all.txt','w',encoding='utf-8')

	for i in range(len(txt_title)):
		print(txt_title[i])
		with open(path+'{}.txt'.format(txt_title[i]),'r',encoding='utf-8') as f:

			while True:
				lines = f.readline()
				if not lines:
					break
				else:
					sents = lines.strip('\n').split('。')
					for j in range(len(sents)):
						if len(sents[j]) != 0:
							sent = jieba.lcut(sents[j])

							word_list = []

							for word in sent:
								if word not in stop_words:
									if word != '\t':
										if word[:2] in people_names:
											word_list.append(word[:2])
										word_list.append(word)
							sent_pro = ' '.join(word_list)
							new_file.write(sent_pro + '\n')
						else:
							pass
	new_file.close()
	pass




def main():

	model = Word2Vec(
	    LineSentence(open(path + 'all.txt', 'r', encoding='utf8')),
	    sg = 0,
	    size = 100,
	    window = 5,
	    min_count = 5,
	    workers=8 )

	# exp1(model)
	exp2(model)


	# dic = model.wv.index2word
	# print(dic)

	# print(len(dic))
	# 159397
	 
	# print(model.wv['杨过'])
	# # 获取词向量
	# print('*'*20)

	# # print(model.most_similar('杨过', topn=1))
	# print(model.wv.most_similar('杨过', topn=1))

	# print('*'*20)



	# #输入与“杨过”相近的10个词
	# for key in model.wv.similar_by_word('杨过', topn =10):
	# 	print(key)


	# uv = np.linalg.norm(model.wv.vectors, axis=1).reshape(-1, 1)  # Unit Vector
	# wv_vectors = model.wv.vectors / uv  # Vector or matrix norm
	# # 聚类
	# n_clusters = 10

	# labels = KMeans(n_clusters).fit(wv_vectors).labels_
	# # 输出excel
	# df = pd.DataFrame([(w, labels[e]) for e, w in enumerate(model.wv.index2word)], columns=['word', 'label'])
	# df.sort_values(by='label', inplace=True)
	# df.to_excel('/Users/huzikang/Desktop/word_cluster.xlsx', index=False)

	# Ref.
	# https://www.zhihu.com/question/283833918/answer/1464235195
	# https://yellow520.blog.csdn.net/article/details/108508540




def find_relation(model,a, b, c):
    d, _ = model.wv.most_similar(positive=[c, b], negative=[a])[0]
    print (c,d)


def exp1(model):
	print(model.wv.similarity('张无忌', '周芷若'))
	print(model.wv.similarity('张无忌', '赵敏'))
	print('*'*20)
	for i in model.wv.most_similar("张无忌", topn=5):
		print(i)
	print('*'*20)
	for i in model.wv.most_similar("峨嵋派", topn=5):
		print(i)
	print('*'*20)
	for i in model.wv.most_similar("韦小宝", topn=10):
		print(i)
	print('*'*20)
	for i in model.wv.most_similar("王重阳", topn=10):
		print(i)
	print('*'*20)
	print(find_relation(model,"杨过","小龙女","张无忌"))
	print('*'*20)
	print(find_relation(model,"杨过","小龙女","郭靖"))
	print('*'*20)
	print(find_relation(model,"杨过","小龙女","黄蓉"))
	print('*'*20)
	print(find_relation(model,"武当派","张三丰","峨嵋派"))
	print('*'*20)
	print(find_relation(model,"武当派","张三丰","天地会"))


def my_cluster(model,input_list):

	dic = model.wv.index2word
	tag_vec = []
	new_list = []
	for tag in input_list:
		if tag in dic:
			vec_cameroon = model.wv[tag]
			tag_vec.append(vec_cameroon)
			new_list.append(tag)

	tag_vec = np.array(tag_vec)
	# print(tag_vec)

	n_clusters=10
	# 建立模型。n_clusters参数用来设置分类个数，即K值，这里表示将样本分为10类。
	cluster = KMeans(n_clusters=n_clusters,random_state=0).fit(tag_vec)

	y_pred = cluster.labels_

	if len(y_pred) == len(new_list):
		for i in range(10):
			n_category = i
			count = 0
			print('\n'+f"类别为{n_category}的数据:"+'\n')
			for j in range(len(new_list)):
				if y_pred[j] == n_category:
					print(f"{new_list[j]}\t",end="")
					count = count + 1
					if(count % 10 == 0):
						print(f"\n")
	else:
		print('something wrong')

	# Ref.
	# https://zhuanlan.zhihu.com/p/437545109



def exp2(model):
	people_names_file = open(path + "金庸小说全人物2.txt", 'r',encoding='utf-8')
	people_names = []
	for line in people_names_file.readlines():
		line = line.strip()   # 去掉每行末尾的换行符
		people_names.append(line)
	people_names_file.close()

	kongfu_file = open(path + "金庸小说全武功.txt", 'r',encoding='gb18030')
	kongfu = []
	for line in kongfu_file.readlines():
		line = line.strip()   # 去掉每行末尾的换行符
		kongfu.append(line)
	kongfu_file.close()

	menpai_file = open(path + "金庸小说全门派.txt", 'r',encoding='gb18030')
	menpai = []
	for line in menpai_file.readlines():
		line = line.strip()   # 去掉每行末尾的换行符
		menpai.append(line)
	menpai_file.close()


	my_cluster(model,people_names)
	print('\n'+'*'*20+'\n')
	my_cluster(model,kongfu)
	print('\n'+'*'*20+'\n')
	my_cluster(model,menpai)
	print('\n'+'*'*20+'\n')





def test():

	pass



if __name__ == '__main__':
	path = '/Users/huzikang/Desktop/jyxsqj_after_process/'
	# preprocessing()
	main()
	# test()






