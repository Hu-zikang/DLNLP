import time
import jieba

import matplotlib
matplotlib.use('TkAgg')  #必须要写在这两个import中间
import matplotlib.pyplot as plt


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.svm import SVC



def plot_top_words(model, feature_names, n_top_words, title):
    # fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
    fig, axes = plt.subplots(2, 5, sharex=True)
    axes = axes.flatten()

    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        # ax.barh(top_features, weights, height=0.7)
        # ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 30})
        # ax.invert_yaxis()
        # ax.tick_params(axis="both", which="major", labelsize=20)
        # for i in "top right left".split():
        #     ax.spines[i].set_visible(False)
        # fig.suptitle(title, fontsize=40)

        ax.barh(top_features, weights)
        ax.set_title(f"Topic {topic_idx +1}")
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major")
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()
'''
Ref.
https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py
'''


def is_chinese(uchar):
    if uchar >= '\u4e00' and uchar <= '\u9fa5':
        return True
    else:
        return False
'''
Ref.
https://blog.csdn.net/lxx199603/article/details/98774653
'''
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

def reserve_special(content):
	special_character = "：；、，。！？‘’“”（）《》…:;\,.!?''""()"
	# 不保留空格
	# 包含中英文两种形式的标点符号
	# special_english_character = ":;\,.!?''""()"

	special_chinese_character = "：；、，。！？‘’“”（）《》…"
	
	content_str = ''
	for i in content:
		if is_chinese(i):
			content_str += i
		else:
			if i in special_character:
				# content_str += i

				if i in special_chinese_character:
					content_str += i
				else:		
					if i == ':':
						content_str += '：'
					if i == ';':
						content_str += '；'
					if i == ',':
						content_str += '，'
					if i == '.':
						content_str += '。'
					if i == '!':
						content_str += '！'
					if i == '?':
						content_str += '?'
					# if i == '\'':
					# 	content_str += '‘'
					# 不分左右不考虑
					# if i == '\"':
					# 	content_str += '”'
					# 此类情况不存在
					if i == '(':
						content_str += '（'
					if i == ')':
						content_str += '）'
					else:
						pass
			else:
				pass
	return content_str


def preprocess():

	path = '/Users/huzikang/Desktop/jyxstxtqj_downcc.com/'
	reocrd_inf = []
	with open(path+'inf.txt','r',encoding='gbk') as f:
		while True:
			lines = f.readline()
			reocrd_inf.append(lines)
			if not lines:
				break
	print(reocrd_inf)
	# 上述部分读取inf.txt记录的小说标题信息


	record_title = reocrd_inf[0].split(',')
	print(len(record_title))
	print(record_title)
	# 将标题记录为列表

	new_path = '/Users/huzikang/Desktop/jyxsqj_after_process/'

	new_file = open(new_path+'inf.txt','w',encoding='utf-8')
	for i in range(len(reocrd_inf)):
		if len(reocrd_inf[i]) != 0:
			new_file.write(reocrd_inf[i])
			new_file.write('\n')
	new_file.close()


	for i in range(len(record_title)):
		record_content = []
		new_file = open(new_path+'{}.txt'.format(record_title[i]),'w',encoding='utf-8')
		if True:
			print(record_title[i])
			with open(path+record_title[i]+'.txt','r',encoding='gb18030') as f:
				while True:
					
					lines = f.readline()

					lines_1 = lines.strip('\n').strip('\u3000')

					if lines_1 == '本书来自www.cr173.com免费txt小说下载站' or lines_1 =='更多更新免费电子书请关注www.cr173.com':
						pass
					else:
						lines_2 = reserve_special(lines_1)
						# 此步骤保留必要的标点符号
						if len(lines_2) != 0:
							new_file.write(lines_2)
							new_file.write('\n')
					if not lines:
						break
			new_file.close()
	pass

	
def tj_500():
	large_500 = {}

	new_path = '/Users/huzikang/Desktop/jyxsqj_after_process/'

	reocrd_inf = []
	with open(new_path+'inf.txt','r',encoding='utf-8') as f:
		while True:
			lines = f.readline()
			reocrd_inf.append(lines)
			if not lines:
				break
	print(reocrd_inf)
	# 上述部分读取inf.txt记录的小说标题信息

	record_title = reocrd_inf[0].strip('\n').split(',')
	print(len(record_title))
	print(record_title)
	# 将标题记录为列表


	for i in range(len(record_title)):
		num = 0

		if True:
			print(record_title[i])
			with open(new_path+record_title[i]+'.txt','r',encoding='utf-8') as f:
				while True:
					
					lines = f.readline().strip('\n')
					if len(jieba.lcut(lines)) >= 500:
						# print(jieba.lcut(lines))
						num += 1

					if not lines:
						break
		large_500[record_title[i]] = num
		print(num)

	return large_500

	# {'白马啸西风': 0, '碧血剑': 111, '飞狐外传': 104, '连城诀': 0, '鹿鼎记': 119, 
	# '三十三剑客图': 29, '射雕英雄传': 201, '神雕侠侣': 66, '书剑恩仇录': 182, '天龙八部': 64, 
	# '侠客行': 0, '笑傲江湖': 189, '雪山飞狐': 0, '倚天屠龙记': 175, '鸳鸯刀': 0, '越女剑': 0}

def len_txt():

	sample_path = '/Users/huzikang/Desktop/jyxsqj_sample/'

	new_path = '/Users/huzikang/Desktop/jyxsqj_after_process/'

	reocrd_inf = []
	with open(new_path+'inf.txt','r',encoding='utf-8') as f:
		while True:
			lines = f.readline()
			reocrd_inf.append(lines)
			if not lines:
				break
	print(reocrd_inf)
	# 上述部分读取inf.txt记录的小说标题信息

	record_title = reocrd_inf[0].strip('\n').split(',')

	print('new_path')
	for i in range(len(record_title)):
		print(record_title[i])
		len_text = 0

		with open(new_path+record_title[i]+'.txt','r',encoding='utf-8') as f:
			while True:
				lines = f.readline().strip('\n')
				len_text += len(lines)
				if not lines:
					break
		print(len_text)
	pass

def random_output():
	large_500 = {'白马啸西风': 0, '碧血剑': 111, '飞狐外传': 104, '连城诀': 0, '鹿鼎记': 119, '三十三剑客图': 29, '射雕英雄传': 201, '神雕侠侣': 66, '书剑恩仇录': 182, '天龙八部': 64, '侠客行': 0, '笑傲江湖': 189, '雪山飞狐': 0, '倚天屠龙记': 175, '鸳鸯刀': 0, '越女剑': 0}

	large_keys = list(large_500.keys())
	large_values = list(large_500.values())

	# ['白马啸西风', '碧血剑', '飞狐外传', '连城诀', '鹿鼎记', '三十三剑客图', '射雕英雄传', '神雕侠侣', '书剑恩仇录', '天龙八部', '侠客行', '笑傲江湖', '雪山飞狐', '倚天屠龙记', '鸳鸯刀', '越女剑']
	# [0, 111, 104, 0, 119, 29, 201, 66, 182, 64, 0, 189, 0, 175, 0, 0]

	record = []

	for i in range(len(large_values)):
		if large_values[i] != 0:
			record.append(i)

	# [1, 2, 4, 5, 6, 7, 8, 9, 11, 13]

	sample_result = {}

	for i in range(200):
		random.seed(time.time())
		xs = random.randint(0,len(record)-1)
		# print(xs)
		if large_keys[record[xs]] not in sample_result:
			sample_result[large_keys[record[xs]]] = []

		while True:
			random.seed(time.time())
			dl = random.randint(1,int(large_values[record[xs]]))
			# print(dl)
			if dl not in sample_result[large_keys[record[xs]]]:
				sample_result[large_keys[record[xs]]].append(dl)
				break

	print(sample_result)


	# {'书剑恩仇录': [26, 15, 50, 63, 150, 136, 22, 174, 77, 129, 31, 164, 170, 117, 68, 167, 60, 178], 
	# '笑傲江湖': [83, 80, 4, 2, 41, 157, 9, 85, 30, 151, 71, 35, 94, 131, 101, 128, 89, 52, 147, 126, 19, 40, 113], 
	# '碧血剑': [89, 28, 26, 47, 31, 60, 84, 43, 13, 35, 104, 50, 57, 83, 96, 54, 15, 25, 23], 
	# '飞狐外传': [54, 81, 46, 92, 78, 27, 96, 86, 63, 69, 72, 16, 99, 6, 60], 
	# '天龙八部': [58, 50, 8, 30, 12, 25, 53, 13, 21, 59, 39, 17, 34, 11, 64, 47, 40, 23, 51], 
	# '射雕英雄传': [170, 177, 37, 142, 29, 78, 137, 94, 163, 198, 31, 164, 39, 110, 14, 129, 135, 41, 128, 151, 5, 95], 
	# '神雕侠侣': [57, 53, 55, 22, 45, 20, 50, 62, 63, 33, 36, 14, 5, 38, 64, 18, 9, 31, 19, 26], 
	# '鹿鼎记': [79, 82, 36, 97, 63, 76, 94, 11, 65, 55, 16, 114, 89, 110, 31, 3, 27, 6, 64, 66, 23, 109, 86, 70, 33, 2, 50, 8], 
	# '倚天屠龙记': [117, 121, 134, 63, 118, 87, 47, 85, 1, 73, 171, 164, 39, 57], 
	# '三十三剑客图': [5, 15, 7, 1, 19, 13, 8, 25, 12, 27, 18, 14, 21, 9, 10, 23, 26, 4, 6, 11, 3, 20]}

def get_sample():
	sample_dict = {'书剑恩仇录': [26, 15, 50, 63, 150, 136, 22, 174, 77, 129, 31, 164, 170, 117, 68, 167, 60, 178], 
	'笑傲江湖': [83, 80, 4, 2, 41, 157, 9, 85, 30, 151, 71, 35, 94, 131, 101, 128, 89, 52, 147, 126, 19, 40, 113], 
	'碧血剑': [89, 28, 26, 47, 31, 60, 84, 43, 13, 35, 104, 50, 57, 83, 96, 54, 15, 25, 23], 
	'飞狐外传': [54, 81, 46, 92, 78, 27, 96, 86, 63, 69, 72, 16, 99, 6, 60], 
	'天龙八部': [58, 50, 8, 30, 12, 25, 53, 13, 21, 59, 39, 17, 34, 11, 64, 47, 40, 23, 51], 
	'射雕英雄传': [170, 177, 37, 142, 29, 78, 137, 94, 163, 198, 31, 164, 39, 110, 14, 129, 135, 41, 128, 151, 5, 95], 
	'神雕侠侣': [57, 53, 55, 22, 45, 20, 50, 62, 63, 33, 36, 14, 5, 38, 64, 18, 9, 31, 19, 26], 
	'鹿鼎记': [79, 82, 36, 97, 63, 76, 94, 11, 65, 55, 16, 114, 89, 110, 31, 3, 27, 6, 64, 66, 23, 109, 86, 70, 33, 2, 50, 8], 
	'倚天屠龙记': [117, 121, 134, 63, 118, 87, 47, 85, 1, 73, 171, 164, 39, 57], 
	'三十三剑客图': [5, 15, 7, 1, 19, 13, 8, 25, 12, 27, 18, 14, 21, 9, 10, 23, 26, 4, 6, 11, 3, 20]}

	# tnum = 0
	# for i in sample_dict:
	# 	tnum += len(sample_dict[i])

	# print(tnum)
	sample_path = '/Users/huzikang/Desktop/jyxsqj_sample/'

	new_path = '/Users/huzikang/Desktop/jyxsqj_after_process/'

	reocrd_inf = []
	with open(new_path+'inf.txt','r',encoding='utf-8') as f:
		while True:
			lines = f.readline()
			reocrd_inf.append(lines)
			if not lines:
				break
	print(reocrd_inf)
	# 上述部分读取inf.txt记录的小说标题信息

	record_title = reocrd_inf[0].strip('\n').split(',')
	# print(len(record_title))
	# print(record_title)
	# 将标题记录为列表

	test_file = open(sample_path+'test.txt','w',encoding='utf-8')

	for i in range(len(record_title)):

		train_file = open(sample_path+'{}.txt'.format(record_title[i]),'w',encoding='utf-8')

		if record_title[i] in sample_dict:
			num = 0
			with open(new_path+record_title[i]+'.txt','r',encoding='utf-8') as f:
				while True:
					lines = f.readline().strip('\n')
					if len(jieba.lcut(lines)) >= 500:
						num += 1
						if num in sample_dict[record_title[i]]:
							test_file.write(lines+'huzikang'+record_title[i]+'\n')
						else:
							train_file.write(lines + '\n')
					else:
						train_file.write(lines + '\n')
					if not lines:
						break
			train_file.close()
		else:
			with open(new_path+record_title[i]+'.txt','r',encoding='utf-8') as f:
				while True:
					lines = f.readline().strip('\n')
					train_file.write(lines + '\n')
					if not lines:
						break
			train_file.close()
			pass
	test_file.close()

def see_sample():
	sample_path = '/Users/huzikang/Desktop/jyxsqj_sample/'

	new_path = '/Users/huzikang/Desktop/jyxsqj_after_process/'
	num = 0
	# with open(new_path+'sample.txt','r',encoding='utf-8') as f:
	with open(sample_path+'test.txt','r',encoding='utf-8') as f:
		while True:
			lines = f.readline().strip('\n')
			if len(lines) != 0:
				num += 1
			if not lines:
				break
	print(num)

def list_to_dict(input_list):

	output_dict = {}
	for i in range(len(input_list)):
		if input_list[i] not in output_dict:
			output_dict[input_list[i]] = []
			output_dict[input_list[i]].append(i+1)
		else:
			output_dict[input_list[i]].append(i+1)
	print(output_dict)
	return output_dict

def label_generate(input_string):
	txt_title = ['白马啸西风', '碧血剑', '飞狐外传', '连城诀', '鹿鼎记', '三十三剑客图', '射雕英雄传', '神雕侠侣', '书剑恩仇录', '天龙八部', '侠客行', '笑傲江湖', '雪山飞狐', '倚天屠龙记', '鸳鸯刀', '越女剑']
	# txt_vector = [0]*16
	# txt_vector[txt_title.index(input_string)] = 1
	# print(txt_vector)
	return txt_title.index(input_string)

def accuracy(result_list,lable_list):
	record = []
	if len(lable_list) == len(result_list):
		num = len(result_list)
		acc = 0
		for i in range(len(result_list)):
			if result_list[i] == lable_list[i]:
				record.append(lable_list[i])
				acc += 1
			else:
				pass
		print('accuracy',acc/num)
		# return acc/num
	else:
		print('sth wrong !')
	count_list_to_dict(record)



def count_list_to_dict(input_list):
	txt_title = ['白马啸西风', '碧血剑', '飞狐外传', '连城诀', '鹿鼎记', '三十三剑客图', '射雕英雄传', '神雕侠侣', '书剑恩仇录', '天龙八部', '侠客行', '笑傲江湖', '雪山飞狐', '倚天屠龙记', '鸳鸯刀', '越女剑']
	result = {}
	for i in set(input_list):
		result[txt_title[i]] = input_list.count(i)
	print(result)

'''
Ref.
https://www.jb51.net/article/208696.htm
'''


def our_lda():

	sample_path = '/Users/huzikang/Desktop/jyxsqj_sample/'
	new_path = '/Users/huzikang/Desktop/jyxsqj_after_process/'
	
	reocrd_inf = []
	with open(new_path+'inf.txt','r',encoding='utf-8') as f:
		while True:
			lines = f.readline()
			reocrd_inf.append(lines)
			if not lines:
				break
	print(reocrd_inf)
	# 上述部分读取inf.txt记录的小说标题信息
	record_title = reocrd_inf[0].strip('\n').split(',')

	#从文件导入停用词表
	# stpwrdpath = "/Users/huzikang/Desktop/jyxsqj_sample/stop_words.txt"
	# stpwrd_dic = open(stpwrdpath, 'r', encoding='gb18030')
	stpwrdpath = "/Users/huzikang/Desktop/jyxsqj_sample/hit_stopwords.txt"
	stpwrd_dic = open(stpwrdpath, 'r', encoding='utf-8')

	stpwrd_content = stpwrd_dic.read()
	#将停用词表转换为list  
	stpwrdlst = stpwrd_content.splitlines()

	print(stpwrdlst[0:100])
	stpwrd_dic.close()


	# count_v0 = CountVectorizer(stop_words = stpwrdlst)
	count_v0 = CountVectorizer()
	all_text = []
	for i in range(len(record_title)):
		print(record_title[i])
		with open(new_path+record_title[i]+'.txt','r',encoding='utf-8') as f:
			while True:
				lines = f.readline()
				if not lines:
					break
				else:
					lines_1 = jieba.lcut(lines.strip('\n'))
					filtered = [w for w in lines_1 if w not in stpwrdlst]
					all_text.append(' '.join(filtered))

	count_v0_ft = count_v0.fit_transform(all_text)


	# count_v1 = CountVectorizer(max_features = 5000)
	count_v1 = CountVectorizer(vocabulary=count_v0.vocabulary_)

	train_text = []
	train_label = []
	for i in range(len(record_title)):
		print(record_title[i])
		txt_str = ''
		# with open(sample_path+record_title[i]+'.txt','r',encoding='utf-8') as f:
		with open(new_path+record_title[i]+'.txt','r',encoding='utf-8') as f:
			
			# all_txt = f.read()
			# lines_2 = jieba.lcut(all_txt)
			# filtered = [w for w in lines_2 if w not in stpwrdlst]
			# train_text.append(' '.join(filtered))
			# train_label.append(label_generate(record_title[i]))

			while True:
				lines = f.readline()
				if not lines:
					if txt_str != '':
						lines_2 = jieba.lcut(txt_str)
						filtered = [w for w in lines_2 if w not in stpwrdlst]
						train_text.append(' '.join(filtered))
						train_label.append(label_generate(record_title[i]))
						txt_str = ''
					break
				else:
					lines_1 = lines.strip('\n')
					if len(jieba.lcut(txt_str+lines_1)) > 500:
						lines_2 = jieba.lcut(txt_str+lines_1)
						filtered = [w for w in lines_2 if w not in stpwrdlst]
						train_text.append(' '.join(filtered))
						train_label.append(label_generate(record_title[i]))
						txt_str = ''
					else:
						continue

	count_v1_ft = count_v1.fit_transform(train_text)

	print(len(count_v1_ft.toarray()))

	# for i in range(100):
	# 	print(len(count_v1_ft.toarray()[i]))
	# print('above')

	# count_v1 = CountVectorizer(vocabulary=count_v0.vocabulary_)

	# train_text = []
	# train_label = []
	# with open(sample_path+'test.txt','r',encoding='utf-8') as f:
	# 	while True:
	# 		lines = f.readline()
	# 		if not lines:
	# 			break
	# 		else:
	# 			lines_1 = lines.strip('\n').split('huzikang')

	# 			lines_2 = jieba.lcut(lines_1[0])
	# 			filtered = [w for w in lines_2 if w not in stpwrdlst]
	# 			train_text.append(' '.join(filtered))

	# 			train_label.append(label_generate(lines_1[1]))

	# count_v1_ft = count_v1.fit_transform(train_text)


	# count_v2 = CountVectorizer(max_features = 5000)
	count_v2 = CountVectorizer(vocabulary=count_v0.vocabulary_)

	test_text = []
	test_label = []
	with open(sample_path+'test.txt','r',encoding='utf-8') as f:
		while True:
			lines = f.readline()
			if not lines:
				break
			else:

				lines_1 = lines.strip('\n').split('huzikang')

				lines_2 = jieba.lcut(lines_1[0])
				filtered = [w for w in lines_2 if w not in stpwrdlst]
				test_text.append(' '.join(filtered))

				test_label.append(label_generate(lines_1[1]))

	count_v2_ft = count_v2.fit_transform(test_text)
	# 结果是词的位置，以及出现次数

	# 文本转为向量
	# print('count_v2_ft',count_v2_ft)
	# 结果是词的位置，以及出现次数
	
	# print(len(count_v2_ft.toarray()))
	# for i in range(200):
	# 	print(len(count_v2_ft.toarray()[i]))

	# # 154330 维数相同




	# acc_list = []
	# # n_list = [3,5,8,10,12,16,20,24,32,40,48]
	# # n_list = [3,5,8,10,12,16]
	# # n_list = [20,24,32,40,48]
	# n_list = [3,4,5,6,7,8,9,10]
	# # [0.124, 0.113, 0.104, 0.128, 0.105, 0.15, 0.108, 0.127]

	# kernel_list = ['rbf']
	# for k in range(1):
	# 	kernel = kernel_list[k]
		
	# 	for n in range(len(n_list)):
	# 		n_components  = n_list[n]

	# 		sum_acc = 0
	# 		for i in range(5):


	# 			lda = LatentDirichletAllocation(n_components = n_components, max_iter = 50, learning_method = 'batch',random_state = None)
	# 			# lda = LatentDirichletAllocation(n_components = 10, max_iter = 50, learning_method = 'batch')
	# 			# 参数 n_components 认为的隐含的主题数
				
	# 			X_train = lda.fit(count_v1_ft).transform(count_v1_ft)

	# 			# print('X_train',X_train)

	# 			X_test = lda.fit(count_v2_ft).transform(count_v2_ft)

	# 			# print('X_test',X_test)


	# 			# svclr = SVC(kernel = 'linear')
	# 			svclr = SVC(kernel = kernel)
	# 			# svclr.fit(count_v1_ft.toarray(),train_label)
	# 			# preds = svclr.predict(count_v2_ft.toarray())
	# 			svclr.fit(X_train,train_label)
	# 			preds = svclr.predict(X_test)

	# 			# print(preds)
	# 			acc = accuracy(list(preds),test_label)

	# 			sum_acc += acc

	# 		acc_list.append(sum_acc/5)

	# print(acc_list)

	# # accuracy 0.105,accuracy 0.15,accuracy 0.155,accuracy 0.19,accuracy 0.145
	# # accuracy 0.205,accuracy 0.065,accuracy 0.125,accuracy 0.13,accuracy 0.145
	# # accuracy 0.095,accuracy 0.12,accuracy 0.095,accuracy 0.14,accuracy 0.22
	# # accuracy 0.065,accuracy 0.1,accuracy 0.1,accuracy 0.085,accuracy 0.115
	# # accuracy 0.045,accuracy 0.07,accuracy 0.17,accuracy 0.08,accuracy 0.16
	# # accuracy 0.115,accuracy 0.1,accuracy 0.065,accuracy 0.095,accuracy 0.115
	# # accuracy 0.11,accuracy 0.145,accuracy 0.11,accuracy 0.145,accuracy 0.14
	# # accuracy 0.115,accuracy 0.09,accuracy 0.1,accuracy 0.12,accuracy 0.1
	# # accuracy 0.125,accuracy 0.06,accuracy 0.095,accuracy 0.085,accuracy 0.12
	# # accuracy 0.09,accuracy 0.07,accuracy 0.125,accuracy 0.085,accuracy 0.115
	# # accuracy 0.115,accuracy 0.09,accuracy 0.08,accuracy 0.065,accuracy 0.13
	# # accuracy 0.09,accuracy 0.095,accuracy 0.08,accuracy 0.065,accuracy 0.105

	# # accuracy 0.085,accuracy 0.11,accuracy 0.095,accuracy 0.145,accuracy 0.12
	# # accuracy 0.11,accuracy 0.125,accuracy 0.115,accuracy 0.09,accuracy 0.08
	# # accuracy 0.115,accuracy 0.135,accuracy 0.09,accuracy 0.09,accuracy 0.095
	# # accuracy 0.095,accuracy 0.125,accuracy 0.1,accuracy 0.12,accuracy 0.085
	# # accuracy 0.095,accuracy 0.115,accuracy 0.13,accuracy 0.06,accuracy 0.14

	# # accuracy 0.105,accuracy 0.15,accuracy 0.04,accuracy 0.085,accuracy 0.085
	# # accuracy 0.09,accuracy 0.13,accuracy 0.14,accuracy 0.12,accuracy 0.05
	# # accuracy 0.08,accuracy 0.085,accuracy 0.075,accuracy 0.165,accuracy 0.105
	# # accuracy 0.065,accuracy 0.07,accuracy 0.095,accuracy 0.085,accuracy 0.11
	# # accuracy 0.095,accuracy 0.11,accuracy 0.13,accuracy 0.145,accuracy 0.115
	# # [0.11100000000000002, 0.10399999999999998, 0.10499999999999998, 0.10500000000000001, 0.10800000000000001, 0.093, 0.10600000000000001, 0.10200000000000001, 0.08499999999999999, 0.119]






	lda = LatentDirichletAllocation(n_components = 10, max_iter = 50, learning_method = 'batch',random_state = 0)
	# lda = LatentDirichletAllocation(n_components = 10, max_iter = 50, learning_method = 'batch')
	# 参数 n_components 认为的隐含的主题数
	
	X_train = lda.fit(count_v1_ft).transform(count_v1_ft)

	print('X_train',X_train)

	X_test = lda.fit(count_v2_ft).transform(count_v2_ft)

	print('X_test',X_test)


	# svclr = SVC(kernel = 'linear')
	svclr = SVC(kernel = 'rbf')
	# svclr.fit(count_v1_ft.toarray(),train_label)
	# preds = svclr.predict(count_v2_ft.toarray())
	svclr.fit(X_train,train_label)
	preds = svclr.predict(X_test)

	print(preds)
	accuracy(list(preds),test_label)




	tf_feature_names = count_v1.get_feature_names()

	# print(tf_feature_names)
	n_top_words = 10

	plot_top_words(lda, tf_feature_names, n_top_words, "Topics in LDA model")


































	# corpus = []

	# for i in range(len(record_title)):
	# 	print(record_title[i])
	# 	with open(sample_path+record_title[i]+'.txt','r',encoding='utf-8') as f:
	# 		all_txt = f.read()
	# 		corpus.append(all_txt)

	# countv = CountVectorizer(stop_words = stpwrdlst)
	# countvf = countv.fit_transform(corpus)

	# # 上述代码把词转化为词频向量


	# lda = LatentDirichletAllocation(n_components = 16, max_iter = 50, learning_method = 'batch')
	# # 参数 n_topics 认为的隐含的主题数
	# result = lda.fit_transform(countvf)
	# print(result)





	

	# topic_list = []

	# for i in range(len(result)):
	# 	j = list(result[i]).index(max(result[i]))
	# 	topic_list.append(j)
	# 	print(i,'\t',j)

	# topic_list_d = list_to_dict(topic_list)

	# # {15: [1, 10], 10: [2], 2: [3, 7], 3: [4, 9, 11, 13], 8: [5], 11: [6], 14: [8], 5: [12, 14, 15], 0: [16]}

	# # {8: [1, 12], 14: [2, 3, 10, 11, 14], 15: [4, 8], 10: [5, 9], 12: [6], 4: [7], 1: [13, 15], 3: [16]}
	# print('词维度',len(lda.components_[0]))










	pass




def main():
	# preprocess()
	# 去除噪声数据

	# large_500 = tj_500()
	# 查看数据情况

	# random_output()
	# 预先确定随机抽取结果

	# len_txt()
	# 查看文本总长度

	# get_sample()
	# 获取训练集与测试集

	# see_sample()
	our_lda()

	pass





def test():
	# sample_path = '/Users/huzikang/Desktop/jyxsqj_sample/'

	# record_title = ['白马啸西风', '碧血剑', '飞狐外传', '连城诀', '鹿鼎记', '三十三剑客图', '射雕英雄传', '神雕侠侣', '书剑恩仇录', '天龙八部', '侠客行', '笑傲江湖', '雪山飞狐', '倚天屠龙记', '鸳鸯刀', '越女剑']

	# train_text = []

	# for i in range(len(record_title)):
	# 	if i == 1:
	# 		print(record_title[i])
	# 		with open(sample_path+record_title[i]+'.txt','r',encoding='utf-8') as f:
	# 			train_txt = f.read()
	# 			train_text.append(train_txt)
	# 			print(train_text)
	
	# test_text = []
	# with open(sample_path+'test.txt','r',encoding='utf-8') as f:
	# 	while True:
	# 		lines = f.readline()
	# 		if not lines:
	# 			break
	# 		else:
	# 			lines_1 = lines.strip('\n').split('huzikang')
	# 			print(lines_1)
	# 		# num += 

	#从文件导入停用词表
	stpwrdpath = "/Users/huzikang/Desktop/jyxsqj_sample/stop_words.txt"
	# stpwrdpath = "/Users/huzikang/Desktop/jyxsqj_sample/st_jc.txt"
	stpwrd_dic = open(stpwrdpath, 'r',encoding='gb18030')
	stpwrd_content = stpwrd_dic.read()
	#将停用词表转换为list  
	stpwrdlst = stpwrd_content.splitlines()
	stpwrd_dic.close()

	n_samples = 2000
	n_features = 1000
	n_components = 10
	n_top_words = 10



	sample_path = '/Users/huzikang/Desktop/jyxsqj_sample/'
	new_path = '/Users/huzikang/Desktop/jyxsqj_after_process/'
	
	record_title = ['白马啸西风', '碧血剑', '飞狐外传', '连城诀', '鹿鼎记', '三十三剑客图', '射雕英雄传', '神雕侠侣', '书剑恩仇录', '天龙八部', '侠客行', '笑傲江湖', '雪山飞狐', '倚天屠龙记', '鸳鸯刀', '越女剑']

	train_text = []
	for i in range(len(record_title)):
		print(record_title[i])
		# with open(sample_path+record_title[i]+'.txt','r',encoding='utf-8') as f:
		with open(new_path+record_title[i]+'.txt','r',encoding='utf-8') as f:
			# all_txt = f.read()
			# train_text.append(all_txt)
			while  True:
				lines = f.readline()
				if not lines:
					break
				else:
					lines_1 = lines.strip('\n')
					lines_2 = jieba.lcut(lines_1)

					filtered = [w for w in list(lines_2) if w not in stpwrdlst]
					lines_3 = ' '.join(filtered)
					# print(lines_3)

					train_text.append(lines_3)
			
	print('train_text',train_text)



	# Use tf (raw term count) features for LDA.
	print("Extracting tf features for LDA...")
	# tf_vectorizer = CountVectorizer(
	#     max_df=0.95, min_df=2, max_features=n_features, stop_words=stpwrdlst
	# )
	tf_vectorizer = CountVectorizer(
	    max_df=0.95, min_df=2, stop_words=stpwrdlst)


	t0 = time.time()

	tf = tf_vectorizer.fit_transform(train_text)


	print("done in %0.3fs." % (time.time() - t0))
	print()



	print(
	    "\n" * 2,
	    "Fitting LDA models with tf features, n_samples=%d and n_features=%d..."
	    % (n_samples, n_features),
	)

	lda = LatentDirichletAllocation(
	    n_components=n_components,
	    max_iter=5,
	    learning_method="online",
	    learning_offset=50.0,
	    random_state=0,
	)
	t0 = time.time()

	lda.fit(tf)
	print("done in %0.3fs." % (time.time() - t0))

	tf_feature_names = tf_vectorizer.get_feature_names()

	print(tf_feature_names)

	plot_top_words(lda, tf_feature_names, n_top_words, "Topics in LDA model")


def test2():

	#从文件导入停用词表
	stpwrdpath = "/Users/huzikang/Desktop/jyxsqj_sample/stop_words.txt"
	# stpwrdpath = "/Users/huzikang/Desktop/jyxsqj_sample/st_jc.txt"
	stpwrd_dic = open(stpwrdpath, 'r',encoding='gb18030')
	stpwrd_content = stpwrd_dic.read()
	#将停用词表转换为list  
	stpwrdlst = stpwrd_content.splitlines()
	stpwrd_dic.close()


	# corpus = [
	# '袁承志心想：“此人魔爪功练到此地步，也非一朝一夕之功，得给他留下颜面，如不让他一招，温青免不得还要说嘴。',
	# '”他自艺成下山，此刻是初次与人动手过招，决意遵照师父叮嘱，容让为先，眼见荣彩右手向自己肩头抓来，故意并不退避。',
	# '荣彩大喜，心中倒并不想伤他，只拟将他衣服撕破一块，就算赢了一招，哪知一抓到他的肩头，突觉他肌肉滑溜异常，竟像水中抓到一尾大鱼那样，一下子就被他滑了开去，正自一惊，袁承志已跳开两步，说道：“我输了！”',
	# '荣彩拱手道：“承让，承让！”温青道：“他是真的让你，你自知之明倒还有的，知道了就好啦！”',
	# '荣彩脸一板，正待发作，忽见岸上火光闪动，数十人手执兵刃火把，快步奔来。',
	# '当先一人叫道：“荣老爷子，已把那小子抓到了吧？咱们把这小子剐了，给沙老大报仇！”',
	# '温青见对方大队拥到，虽然胆大妄为，心中也不禁惴惴。荣彩叫道：“刘家兄弟，你们两人过来！”',
	# '岸上两人应声走到岸边，见大船离岸甚远，扑通两声跳入江内，迅速游到船边，水性极是了得，单手在船舷上一搭，扑地跳了上来。',
	# '荣彩道：“那包货色给这小子丢到江心去啦，你哥儿俩去捡起来！”说着向江心一指。刘氏兄弟跃落江中，潜入水内。',
	# '温青一扯袁承志的袖子，在他耳边低声说道：“快救救我吧，他们要杀我呢！”',
	# '袁承志回过头来，月光下见他容色愁苦，一副楚楚可怜的神气，便点了点头。',
	# '温青拉住他的手道：“他们人多势众。你想法子斩断铁链，咱们开船逃走。”',
	# '袁承志还未答应，只觉温青的手又软又腻，柔若无骨，甚感诧异：“这人的手掌像棉花一样，当真希奇。”',
	# '这时荣彩已留意到两人在窃窃私议，回头望来。温青把袁承志的手捏了一把，突然猛力举起船头桌子，向荣彩等三人推去。',
	# '那大汉与妇人正全神望着刘氏兄弟潜水取金，出其不意，背上被桌子一撞，惊叫一声，一齐掉下水去。',
	# '荣彩纵身跃起，伸掌抓出，五指嵌入桌面，用力一拉一掀，格格两声，温青握着的桌脚已然折断。',
	# '荣彩知道那大汉与妇人不会水性，这时江流正急，刘氏兄弟相距甚远，不及过来救援，忙把桌子抛入江中，让二人攀住了不致沉下，随即双拳呼呼两招，向温青劈面打来。',
	# '温青提了两条桌腿，护住面门，急叫：“快！你。”袁承志提起铁链，“混元功”内劲到处，一提一拉，那只大铁锚呼的一声，离岸向船头飞来。荣彩和温青大惊，忙向两侧跃开，回头看袁承志时，但见他手中托住铁锚，缓缓放在船头。',
	# '铁锚一起，大船登时向下游流去，与岸上众人慢慢远离。荣彩见他如此功力，料知若再逗留，决计讨不了好去，双足一顿，提气向岸上跃去。袁承志看他的身法，知他跃不上岸，提起一块船板，向江边掷去。',
	# '荣彩下落时见足底茫茫一片水光，正自惊惶，突见船板飞到，恰好落在脚下水面之上，当真大喜过望，左脚在船板上一借力。跃上了岸，暗暗感激他的好意，又不禁佩服他的功力，自己人先跃出，他飞掷船板，居然能及时赶到。'
	# '温青哼了一声，道：“不分青红皂白，便是爱做滥好人！到底你是帮我呢，还是帮这老头儿？让他在水里浸一下，喝几口江水不好吗？又不会淹死人。”'
	# ]

	corpus = ['This is the first document.',    'This document is the second document.',    'And this is the third one.',   'Is this the first document?']

	train_text = []

	for i in range(len(corpus)):
		corpus_list = jieba.lcut(corpus[i])
		filtered = [w for w in list(corpus_list) if w not in stpwrdlst]
		train_text.append(' '.join(filtered))

	print(train_text)

	vectorizer = CountVectorizer()
	X = vectorizer.fit_transform(train_text)
	print('0',X)
	# 打印的是词的位置，与数量

	vgf = vectorizer.get_feature_names()
	print('1',vgf)
	# 打印获得的所有的词，英文中可能就是按空格拆分

	print(X.toarray())
	# 简单的 0-1 向量表示


	vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2))

	X2 = vectorizer2.fit_transform(train_text)

	print('00',X2)

	v2gf = vectorizer2.get_feature_names()
	print('2',v2gf)

	print(X2.toarray())
	pass



def test3():

	sample_path = '/Users/huzikang/Desktop/jyxsqj_sample/'
	new_path = '/Users/huzikang/Desktop/jyxsqj_after_process/'
	
	record_title = ['白马啸西风', '碧血剑', '飞狐外传', '连城诀', '鹿鼎记', '三十三剑客图', '射雕英雄传', '神雕侠侣', '书剑恩仇录', '天龙八部', '侠客行', '笑傲江湖', '雪山飞狐', '倚天屠龙记', '鸳鸯刀', '越女剑']

	# train_text = []
	# for i in range(len(record_title)):
	# 	print(record_title[i])
	# 	# with open(sample_path+record_title[i]+'.txt','r',encoding='utf-8') as f:
	# 	with open(new_path+record_title[i]+'.txt','r',encoding='utf-8') as f:
	# 		all_txt = f.read()
	# 		train_text.append(all_txt)

	#从文件导入停用词表
	stpwrdpath = "/Users/huzikang/Desktop/jyxsqj_sample/stop_words.txt"
	# stpwrdpath = "/Users/huzikang/Desktop/jyxsqj_sample/st_jc.txt"
	stpwrd_dic = open(stpwrdpath, 'r', encoding='gb18030')
	stpwrd_content = stpwrd_dic.read()
	#将停用词表转换为list  
	stpwrdlst = stpwrd_content.splitlines()
	stpwrd_dic.close()

	print(stpwrdlst[0:100])

	with open(sample_path+'test.txt','r',encoding='utf-8') as f:
		lines = f.readline().strip('\n').split('huzikang')
		
		seg_list = list(jieba.cut(lines[0]))

		print(list(seg_list))
		print('\n'*3)



		filtered = [w for w in list(seg_list) if w not in stpwrdlst]

		# line = " ".join(seg_list)

		# text = jieba.lcut(lines[0])
		# print(line)
		print(filtered)

	pass



def test4():
	sample_dict = {'书剑恩仇录': [26, 15, 50, 63, 150, 136, 22, 174, 77, 129, 31, 164, 170, 117, 68, 167, 60, 178], 
	'笑傲江湖': [83, 80, 4, 2, 41, 157, 9, 85, 30, 151, 71, 35, 94, 131, 101, 128, 89, 52, 147, 126, 19, 40, 113], 
	'碧血剑': [89, 28, 26, 47, 31, 60, 84, 43, 13, 35, 104, 50, 57, 83, 96, 54, 15, 25, 23], 
	'飞狐外传': [54, 81, 46, 92, 78, 27, 96, 86, 63, 69, 72, 16, 99, 6, 60], 
	'天龙八部': [58, 50, 8, 30, 12, 25, 53, 13, 21, 59, 39, 17, 34, 11, 64, 47, 40, 23, 51], 
	'射雕英雄传': [170, 177, 37, 142, 29, 78, 137, 94, 163, 198, 31, 164, 39, 110, 14, 129, 135, 41, 128, 151, 5, 95], 
	'神雕侠侣': [57, 53, 55, 22, 45, 20, 50, 62, 63, 33, 36, 14, 5, 38, 64, 18, 9, 31, 19, 26], 
	'鹿鼎记': [79, 82, 36, 97, 63, 76, 94, 11, 65, 55, 16, 114, 89, 110, 31, 3, 27, 6, 64, 66, 23, 109, 86, 70, 33, 2, 50, 8], 
	'倚天屠龙记': [117, 121, 134, 63, 118, 87, 47, 85, 1, 73, 171, 164, 39, 57], 
	'三十三剑客图': [5, 15, 7, 1, 19, 13, 8, 25, 12, 27, 18, 14, 21, 9, 10, 23, 26, 4, 6, 11, 3, 20]}

	for i in sample_dict:
		print(i)
		print(len(sample_dict[i]))

if __name__ == '__main__':
	main()
	# test()
	# test2()
	# test3()
	# test4()
















