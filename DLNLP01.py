import jieba
import math


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







def main():
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



	record_other = []
	# 记录特殊字符

	record_statistic = []


	# num = 0
	# corespend_txt = ''


	train_set = []
	test_set = []


	for i in range(len(record_title)):
		record_content = []
		if True:
			print(record_title[i])
			with open(path+record_title[i]+'.txt','r',encoding='gb18030') as f:
				while True:
					
					lines = f.readline()
					# if num == 1598:
					# 	print(lines)
					# 	lines_1 = lines.strip('\n').strip('\u3000').strip('本书来自www.cr173.com免费txt小说下载站').strip('更多更新免费电子书请关注www.cr173.com')
					# 	print(lines_1)
					# 	lines_2 = reserve_special(lines_1)
					# 	print(lines_2)

					# 	num +=1
					# else:
					# 	num += 1
					# lines = lines.strip('\n').strip('\u3000').strip('本书来自www.cr173.com免费txt小说下载站').strip('更多更新免费电子书请关注www.cr173.com')
					# lines = reserve_special(lines)
					lines_1 = lines.strip('\n').strip('\u3000')
					# lines_1 = lines.strip()
					# lines_1 = lines

					if lines_1 == '本书来自www.cr173.com免费txt小说下载站' or lines_1 =='更多更新免费电子书请关注www.cr173.com':
						pass
					else:
						# .strip('本书来自www.cr173.com免费txt小说下载站').strip('更多更新免费电子书请关注www.cr173.com')

						lines_2 = reserve_special(lines_1)
						# lines_2 = lines_1

						if len(lines_2) != 0:
							record_content.append(lines_2)
							# record_content.append(lines.strip('\n').strip('\u3000').strip('本书来自www.cr173.com免费txt小说下载站').strip('更多更新免费电子书请关注www.cr173.com'))


					if not lines:
						break
			# with 结构与for 循环，按行读取所有文本信息，以列表存储


			# word_num = 0
			# sum_num = 0

			# for rc in range(len(record_content)):
			# 	sum_num += len(record_content[rc])
			# 	for c in record_content[rc]:
			# 		if is_chinese(c):
			# 			word_num += 1
			# record_statistic.append([record_title[i],len(record_content),word_num,sum_num])

			# # 记录标题，行数，总汉字数，总字符数



			# print(record_content)
			# for rc in range(len(record_content)):
			# # 	# if "十四个兵部尚" in record_content[rc]: 
			# # 		# print(rc)
			# 		print(record_content[rc])
			# print(len(record_content))

			# 打印或按行打印读取到的文本信息



			# record_clear = []

			# for j in range(len(record_content)):
			# 	for k in record_content[j]:
			# 		if is_chinese(k):
			# 			pass
			# 		else:
			# 			if k not in record_other:
			# 				record_other.append(k)

			# 记录所有文本中的非汉字字符

			if i == 15:
				test_set.append(record_content)
			train_set.append(record_content)
				

	


	# print(record_other)
	# print(len(record_other))
	# special_character = "：；、，。！？‘’“”（）《》…:;\,.!?''\"\"()"
	# print(len(special_character))

	# for l in record_other:
	# 	print(l)

	# # 查看所记录的非汉字字符

	# print(record_statistic)
	# # 统计数据打印，对应文中统计表

	# print(len(train_set))
	# print(len(test_set))



	character_model(train_set,test_set)
	words_model(train_set,test_set)

	# only_character_model(train_set,test_set)

	# 运行计算熵模型


def reserve_only_character(sample_list):
	'''
	以一本小说为例，拆解为纯汉字的字符
	'''
	result_list = []
	for sample in sample_list:
		temp_list = []
		temp_string = ''
		for c in sample:
			if is_chinese(c):
				temp_string += c
			else:
				if len(temp_string) != 0:
					temp_list.append(temp_string)
					temp_string = ''
				else:
					temp_string = ''
		
		if len(temp_string) != 0:
			temp_list.append(temp_string)
			temp_string = ''

		if len(temp_list) != 0:
			for i in temp_list:
				result_list.append(temp_list)
		else:
			print('maybe sth wrong !')
	return result_list






def only_character_model(train_set,test_set):
	'''
	该部分想去除所有非汉字字符，并将不相连的句子独立再计算熵
	'''
	new_train_set = []
	for t in train_set:
		new_list = reserve_only_character(t)
		new_train_set.append(new_list)

	'''
	偷懒，不写了
	'''

	pass



def together_string(sample_list):
	result_string = ''
	for s in sample_list:
		result_string = result_string + s

	return result_string

def split_str_in_character(sample_string):
	result_list = []
	for s in sample_string:
		result_list.append(s)

	return result_list



def character_model(train_set,test_set):
	'''
	按字计算
	'''
	# 先建立词典
	character_dict_2 = {}

	for t in train_set:
		train_list = split_str_in_character(together_string(t))
		for i in range(len(train_list)-1):
			if train_list[i]+train_list[i+1] not in character_dict_2:
				character_dict_2[train_list[i]+train_list[i+1]] = 1
			else:
				character_dict_2[train_list[i]+train_list[i+1]] = character_dict_2[train_list[i]+train_list[i+1]]+1

	print(len(character_dict_2))
	# print(character_dict_2)

	# result_dict = sorted(character_dict_2.items(),key=lambda t:t[1],reverse = True)
	# 仅查看用，返回元组


	# print(result_dict)
	# print(len(result_dict))
	# print(sum(character_dict_2.values()))

	character_dict_3 = {}

	for t in train_set:
		train_list = split_str_in_character(together_string(t))
		for i in range(len(train_list)-2):
			if train_list[i]+train_list[i+1]+train_list[i+2] not in character_dict_3:
				character_dict_3[train_list[i]+train_list[i+1]+train_list[i+2]] = 1
			else:
				character_dict_3[train_list[i]+train_list[i+1]+train_list[i+2]] += 1
	print(len(character_dict_3))

	Entropy_in_character = 0

	for t in test_set:
		test_list = split_str_in_character(together_string(t))
		for i in range(len(test_list)):
			if i == 0:
				mt1t2 = character_dict_2[test_list[i]+test_list[i+1]] / sum(character_dict_2.values())
				Entropy_in_character = Entropy_in_character + math.log(mt1t2,2)
			if i == 1:
				pass
			if i > 1:
				mti = character_dict_3[test_list[i-2]+test_list[i-1]+test_list[i]] / character_dict_2[test_list[i-2]+test_list[i-1]]
				Entropy_in_character = Entropy_in_character + math.log(mti,2)
	# print("前一阶段",Entropy_in_character)

	Entropy_in_character = -(1/len(test_list))*Entropy_in_character

	print('按字划分熵：',Entropy_in_character)

	# 按字划分熵：3.9437375493768028



def words_model(train_set,test_set):
	'''
	按词计算
	'''
	# 先建立词典
	character_dict_2 = {}

	for t in train_set:
		train_list = jieba.lcut(together_string(t))
		# 采用 jieba 划分为词

		for i in range(len(train_list)-1):
			if train_list[i]+train_list[i+1] not in character_dict_2:
				character_dict_2[train_list[i]+train_list[i+1]] = 1
			else:
				character_dict_2[train_list[i]+train_list[i+1]] = character_dict_2[train_list[i]+train_list[i+1]]+1

	print(len(character_dict_2))
	# print(character_dict_2)

	# result_dict = sorted(character_dict_2.items(),key=lambda t:t[1],reverse = True)
	# 仅查看用，返回元组


	# print(result_dict)
	# print(len(result_dict))
	# print(sum(character_dict_2.values()))

	character_dict_3 = {}

	for t in train_set:
		train_list = jieba.lcut(together_string(t))

		for i in range(len(train_list)-2):
			if train_list[i]+train_list[i+1]+train_list[i+2] not in character_dict_3:
				character_dict_3[train_list[i]+train_list[i+1]+train_list[i+2]] = 1
			else:
				character_dict_3[train_list[i]+train_list[i+1]+train_list[i+2]] += 1
	print(len(character_dict_3))

	Entropy_in_words = 0

	for t in test_set:
		test_list = jieba.lcut(together_string(t))

		for i in range(len(test_list)):
			if i == 0:
				mt1t2 = character_dict_2[test_list[i]+test_list[i+1]] / sum(character_dict_2.values())
				Entropy_in_words = Entropy_in_words + math.log(mt1t2,2)
			if i == 1:
				pass
			if i > 1:
				mti = character_dict_3[test_list[i-2]+test_list[i-1]+test_list[i]] / character_dict_2[test_list[i-2]+test_list[i-1]]
				Entropy_in_words = Entropy_in_words + math.log(mti,2)
	# print("前一阶段",Entropy_in_character)

	Entropy_in_words = -(1/len(test_list))*Entropy_in_words

	print('按词划分熵：',Entropy_in_words)

	# 按词划分熵： 2.993151420634368








def test():
	string = '本书来自www.cr173.com免费txt小说下载站,更多更新免费电子书请关注www.cr173.com'
	test_str = '小说《下载站》下载小说免费：“更多更新电子书，关注、书、免费、本书。来？自！站。'

	# jc = jieba.lcut(test_str)
	# # 不加参数默认为精确模式，返回列表

	# print(jc)

	# # for j in jc:
	# 	print(j)


	# result = test_str.split('：；、，。！？‘’“”（）《》…:;\,.!?''\"\"()')

	result = test_str.split("",)
	print(result)	

	pass






if __name__ == '__main__':
	main()
	# test()














