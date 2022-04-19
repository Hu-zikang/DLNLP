import time
import random
import sys
# 导入sys模块
# sys.setrecursionlimit(100)
# 将默认的递归深度修改为3000

from tqdm import tqdm


def parameter_random():
	# random.seed(1)
	random.seed(time.time())

	int1 = random.randint(1,98)
	# 上下限均包含，服从均匀分布
	# 放弃取极值，后续估计分母为0
	int2 = random.randint(int1+1,99)
	s1 = int1/100
	s2 = (int2-int1)/100
	s3 = (100-int2)/100
	# random.seed(s1)
	p = random.randint(1,99)/100
	# random.seed(s2)
	q = random.randint(1,99)/100
	# random.seed(s3)
	r = random.randint(1,99)/100
	# 设定随机种子后，若不重启，则全局随机都有迹可循
	print('随机序列生成依据点')
	print(s1,s2,s3,p,q,r)
	return s1,s2,s3,p,q,r

def start_random():
	random.seed(time.time())
	# random.seed(2)
	int1 = random.randint(1,98)
	int2 = random.randint(int1+1,99)
	s1 = int1/100
	s2 = (int2-int1)/100
	s3 = (100-int2)/100

	p = random.randint(1,99)/100
	q = random.randint(1,99)/100
	r = random.randint(1,99)/100

	print('随机出发点\n',s1,s2,s3,p,q,r)
	# return s1,s2,s3,p,q,r
	return [s1,s2,s3,p,q,r]


def random_sequence_output(s1,s2,s3,p,q,r):
	random.seed(time.time())
	# random.seed()
	seq_list = []
	print('生成数据中。。。')
	for i in tqdm(range(N)):
		coin_choice = random.random()
		# 返回实数，0-1之间，左闭右开
		if coin_choice >= 0 and coin_choice <= s1:
			H_or_T = random.random()
			if H_or_T <= p and H_or_T >= 0:
				seq_list.append(1)
			elif H_or_T <= 1 and H_or_T > p:
				seq_list.append(0)
			else:
				print('sth wrong !')

		elif coin_choice > s1 and coin_choice <= s1+s2:
			H_or_T = random.random()
			if H_or_T <= q and H_or_T >= 0:
				seq_list.append(1)
			elif H_or_T <= 1 and H_or_T > q:
				seq_list.append(0)
			else:
				print('sth wrong !')

		elif coin_choice > s1+s2 and coin_choice <= 1:
			H_or_T = random.random()
			if H_or_T <= r and H_or_T >= 0:
				seq_list.append(1)
			elif H_or_T <= 1 and H_or_T > r:
				seq_list.append(0)
			else:
				print('sth wrong !')

		else:
			print('sth wrong !')

	print(seq_list)
	print('生成随机数据结束。。。')
	return seq_list

def miu_1(para_list,x):
	if x == 1:
		probability = (para_list[0]*para_list[3])/(
			para_list[0]*para_list[3] + para_list[1]*para_list[4] + para_list[2]*para_list[5])
		# 简化计算公式，该种硬币的占比是不变的，分情况讨论时，因正反面不同，相应的正反面概率描述也需要变
	elif x == 0:
		probability = (para_list[0]*(1-para_list[3]))/(
			para_list[0]*(1-para_list[3]) + para_list[1]*(1-para_list[4]) + para_list[2]*(1-para_list[5]))
	else:
		print('sth wrong !')
	return probability

def miu_2(para_list,x):
	if x == 1:
		probability = (para_list[1]*para_list[4])/(
			para_list[0]*para_list[3] + para_list[1]*para_list[4] + para_list[2]*para_list[5])

	elif x == 0:
		probability = (para_list[1]*(1-para_list[4]))/(
			para_list[0]*(1-para_list[3]) + para_list[1]*(1-para_list[4]) + para_list[2]*(1-para_list[5]))
	else:
		print('sth wrong !')
	return probability

def miu_3(para_list,x):
	if x == 1:
		probability = (para_list[2]*para_list[5])/(
			para_list[0]*para_list[3] + para_list[1]*para_list[4] + para_list[2]*para_list[5])
		# 简化计算公式，该种硬币的占比是不变的，分情况讨论时，因正反面不同，相应的正反面概率描述也需要变
	elif x == 0:
		probability = (para_list[2]*(1-para_list[5]))/(
			para_list[0]*(1-para_list[3]) + para_list[1]*(1-para_list[4]) + para_list[2]*(1-para_list[5]))
	else:
		print('sth wrong !')
	return probability



def compare_list(list_1,list_2):

	# print('旧估计点\n',list_1)
	# print('新估计点\n',list_2)
	list_temp = [abs(list_1[i]-list_2[i]) for i in range(len(list_1))]
	# print(list_temp)
	# print(max(list_temp))
	return max(list_temp)

def distance_between_vectors(list_1,list_2):
	# dist = 0
	list_temp = [(list_1[i]-list_2[i])**2 for i in range(len(list_1))]
	dist = (sum(list_temp))**(0.5)
	return dist


def parameter_estiamte(para_list,seq_list):

	# s_s1,s_s2,s_s3,s_p,s_q,s_r = start_random()
	# old_para_list = para_list
	'''
	para_list，需要每次更新
	'''
	e_s1 = 0
	e_s2 = 0
	e_s3 = 0
	e_p_1 = 0
	e_q_1 = 0
	e_r_1 = 0
	for i in seq_list:
		e_s1 += miu_1(para_list,int(i))
		e_s2 += miu_2(para_list,int(i))
		e_s3 += miu_3(para_list,int(i))
		if i == 0:
			pass
		elif i == 1:
			e_p_1 += miu_1(para_list,int(i))
			e_q_1 += miu_2(para_list,int(i))
			e_r_1 += miu_3(para_list,int(i))
		else:
			print('sth wrong !')
	e_p = e_p_1/e_s1
	e_q = e_q_1/e_s2
	e_r = e_r_1/e_s3

	e_s1 = e_s1/N
	e_s2 = e_s2/N
	e_s3 = e_s3/N

	new_para_list = [e_s1,e_s2,e_s3,e_p,e_q,e_r]

	return new_para_list


	# if distance_between_vectors(para_list,new_para_list)> 1e-6:
	# # if compare_list([0.3,0.3,0.4,0.2,0.3,0.7],new_para_list)>0.01:
	# 	print('第{')
	# 	parameter_estiamte(new_para_list,seq_list)
	# else:
	# 	print('收敛')



def main():
	s1,s2,s3,p,q,r = parameter_random()
	# s1,s2,s3,p,q,r = 0.3,0.3,0.4,0.2,0.3,0.7
	gnr_para_list = [s1,s2,s3,p,q,r]
	
	seq_list = random_sequence_output(s1,s2,s3,p,q,r)
	# 生成获取随机数据

	para_list = start_random()
	# 随机获取一个初始点

	gs_list.append(distance_between_vectors(gnr_para_list,para_list))

	old_para_list = parameter_estiamte(para_list,seq_list)
	# 开始参数估计




	for i in range(1000):
		new_para_list = parameter_estiamte(old_para_list,seq_list)

		if distance_between_vectors(old_para_list,new_para_list) > 1e-6:
			if i < 999:
				print('第{}次估计未收敛'.format(i+2),new_para_list)
				old_para_list = new_para_list
			elif i == 999:
				sl_list.append(10000)
				dist_list.append(10000)
				away_list.append(10000)
				pass
		else:
			print('第{}次估计收敛'.format(i+2))
			sl_list.append(i+2)
			dist_list.append(distance_between_vectors(new_para_list,gnr_para_list))
			away_list.append(distance_between_vectors(new_para_list,para_list))
			print('上一次参数',old_para_list)
			print('新参数',new_para_list)
			break


def exp2():
	'''
	此部分验证多点取最优，但需注意，前提是知道何为最优
	'''

	standard = 9999

	s1,s2,s3,p,q,r = parameter_random()
	# s1,s2,s3,p,q,r = 0.3,0.3,0.4,0.2,0.3,0.7
	gnr_para_list = [s1,s2,s3,p,q,r]
	
	seq_list = random_sequence_output(s1,s2,s3,p,q,r)
	# 生成获取随机数据

	for rs in range(M):
		para_list = start_random()
		# 随机获取一个初始点
		# gs_list.append(distance_between_vectors(gnr_para_list,para_list))

		old_para_list = parameter_estiamte(para_list,seq_list)
		# 开始参数估计

		for n in range(1000):
			new_para_list = parameter_estiamte(old_para_list,seq_list)

			if distance_between_vectors(old_para_list,new_para_list) > 1e-6:
				# 这部分基本不经过
				if n < 999:
					print('第{}次估计未收敛'.format(n+2),new_para_list)
					old_para_list = new_para_list

				elif n == 999:
					print('sth wrong !')
					break
			else:
				print('第{}次估计收敛'.format(n+2))

				if distance_between_vectors(new_para_list,gnr_para_list) < standard:
					standard = distance_between_vectors(new_para_list,gnr_para_list)
				else:
					pass
				print('上一次参数',old_para_list)
				print('新参数',new_para_list)
				break

	m20_list.append(standard)
















if __name__ == '__main__':
	'''
	
	# experiment 1
	list_length = [20,50,100,500,1000,2000,5000,10000,50000,100000,500000,1000000]
	# N = list_length[-1]
	# N = 10

	sl_dict = {}
	dist_dict = {}
	away_dict = {}
	gs_dict = {}

	for i in range(len(list_length)):

		N  = list_length[i]

		sl_list = []
		dist_list = []
		away_list = []
		gs_list = []

		for j in range(100):
			main()

		if max(sl_list) == 10000:
			sl_dict[N] = -1
		else:
			sl_dict[N] = sum(sl_list)/100

		if max(dist_list) == 10000:
			dist_dict[N] = -1
		else:
			dist_dict[N] = sum(dist_list)/100

		if max(away_list) == 10000:
			away_dict[N] = -1
		else:
			away_dict[N] = sum(away_list)/100

		gs_dict[N] = sum(gs_list)/100

	print(sl_dict)
	print(dist_dict)
	print(away_dict)
	print(gs_dict)
	


	# s1,s2,s3,p,q,r = 0.3,0.3,0.4,0.2,0.3,0.7
	# 固定了生成参数
	# {20: 2.0, 50: 2.0, 100: 2.0, 500: 2.0, 1000: 2.0, 2000: 2.0, 5000: 2.0, 10000: 2.0, 50000: 2.0, 100000: 2.0, 500000: 2.0, 1000000: 2.0}
	# 随机参数
	# {20: 2.0, 50: 2.0, 100: 2.0, 500: 2.0, 1000: 2.0, 2000: 2.0, 5000: 2.0, 10000: 2.0, 50000: 2.0, 100000: 2.0, 500000: 2.0, 1000000: 2.0}

	
	# {20: 2.0, 50: 2.0, 100: 2.0, 500: 2.0, 1000: 2.0, 2000: 2.0, 5000: 2.0, 10000: 2.0, 50000: 2.0, 100000: 2.0, 500000: 2.0, 1000000: 2.0}
	# {20: 0.8030915504941109, 50: 0.7796643448924422, 100: 0.807116557020656, 500: 0.7960954468722712, 1000: 0.8380753225611977, 2000: 0.7749532437037374, 5000: 0.791534514378461, 10000: 0.7314694567558278, 50000: 0.8187337205720197, 100000: 0.7614734305867427, 500000: 0.7468928835805589, 1000000: 0.8112336045364208}
	# {20: 0.4315329678266357, 50: 0.3992197233636615, 100: 0.43135580921754735, 500: 0.3441696255812381, 1000: 0.34788375700495, 2000: 0.3680391307654935, 5000: 0.4145698252376351, 10000: 0.38833648191149855, 50000: 0.4458986492162406, 100000: 0.414842687554312, 500000: 0.3819723150805573, 1000000: 0.3978658838427536}
	# {20: 0.89709138233908, 50: 0.8800740423850344, 100: 0.8960472840941377, 500: 0.8647319151204007, 1000: 0.8798860222783117, 2000: 0.8367069777667168, 5000: 0.8710250194748521, 10000: 0.8333135499276362, 50000: 0.8987421205829901, 100000: 0.8592566668091368, 500000: 0.8469727279169287, 1000000: 0.8881250594653051}

	# {20: 2.0, 50: 2.0, 100: 2.0, 500: 2.0, 1000: 2.0, 2000: 2.0, 5000: 2.0, 10000: 2.0, 50000: 2.0, 100000: 2.0, 500000: 2.0, 1000000: 2.0}
	# {20: 0.8010001007043228, 50: 0.870248065802469, 100: 0.7777302249138726, 500: 0.8418301129972524, 1000: 0.7929063381484498, 2000: 0.8092867568817822, 5000: 0.8208195774008118, 10000: 0.7421441814785941, 50000: 0.7757495519433859, 100000: 0.810738033689948, 500000: 0.8082941311157523, 1000000: 0.7910709933684512}
	# {20: 0.37530557520943425, 50: 0.34471501148747485, 100: 0.3403119175299688, 500: 0.3609048486164348, 1000: 0.41155140133150175, 2000: 0.3528827804859248, 5000: 0.36813489489222745, 10000: 0.33054716552611096, 50000: 0.3662041247381253, 100000: 0.34857254038929986, 500000: 0.39127001189326754, 1000000: 0.328773794030586}
	# {20: 0.8468451829637176, 50: 0.9013135454022556, 100: 0.8475556876957561, 500: 0.93114239356126, 1000: 0.8951063803636333, 2000: 0.8778438838377681, 5000: 0.8909735154787151, 10000: 0.8360924111827278, 50000: 0.8343788018759538, 100000: 0.8768633504385217, 500000: 0.8602570638007261, 1000000: 0.8507542309845626}


	'''
	# experiment 2
	

	start_length = [10,20,50,100,200,500]

	list_length = [20,50,100,500,1000,2000,5000,10000]
	# N = list_length[-1]
	# N = 10

	# sl_dict = {}
	# dist_dict = {}
	# away_dict = {}
	# gs_dict = {}

	# tj_list = []
	tj_dict = {}
	res_dict = {}

	for i in range(len(list_length)):
		# 不同长度N
		
		N  = list_length[i]

		# dm_list = []

		for m in range(len(start_length)):
			# 不同的随机点数 M

			M = start_length[m]

			m20_list = []

			for j in range(20):
				# 做20次实验，每次实验都要寻M个随机点
				exp2()
			avg_dist = sum(m20_list)/20

			tj_dict[str(N)+str(M)] = m20_list
			res_dict[str(N)+str(M)] = avg_dist


	print(tj_dict)
	print(res_dict)



	'''
	{'2010': [0.5155632272193875, 0.5649034661123222, 0.381802941044358, 0.37704142632015736, 0.2920868195423493, 0.7234373098667938, 0.3784006354718793, 0.3786738253432121, 0.44039330781827296, 0.23266190694055933, 0.5143028330080315, 0.45731632274097334, 0.4745877693023332, 0.2404175949957699, 0.42178447467835084, 0.48623018162802667, 0.22043966210419644, 0.3210354330643882, 0.3042058986554567, 0.46532767506915834], 
	'2020': [0.2745882774464442, 0.7401824367559945, 0.21704683676171227, 0.1663095778417705, 0.4214677155876792, 0.32493180832353064, 0.392892674114373, 0.44282668658737667, 0.6178810447889381, 0.23786403428428254, 0.5709439448684986, 0.22707854093511715, 0.4706356205810276, 0.37175851149623235, 0.30923431697986764, 0.3103600227371997, 0.41518493045414717, 0.5184227861218723, 0.20558185157626505, 0.41351804159188194], 
	'2050': [0.2757742391478089, 0.19031605644291483, 0.37361707651126674, 0.36215357690707584, 0.4473891457626416, 0.29272955566948, 0.25087681049450256, 0.1203298844891323, 0.2865729956126023, 0.14229974440678253, 0.22734075769997508, 0.2071274490235706, 0.28898665060603035, 0.3537789974147729, 0.23745331682067836, 0.2297907218018133, 0.13903238040253965, 0.38061407847920986, 0.3669928745482395, 0.3683705501455217], 
	'20100': [0.17310985512926877, 0.19397929912663814, 0.3320802341867119, 0.15806637216039116, 0.19037041656059198, 0.16719094527614123, 0.25862100481052347, 0.14846653067184912, 0.32652604048826, 0.2882905647293919, 0.4350773024886525, 0.2675357861993945, 0.21114040482977262, 0.2225538631543809, 0.1580282698238579, 0.12376377328894324, 0.24734105747511131, 0.20072846972836808, 0.16556330859396795, 0.37553367929090625], 
	'20200': [0.25458390350147475, 0.19024778219225727, 0.23641124124362076, 0.2736949789745875, 0.10726097741169457, 0.11682223073563731, 0.2167146688780791, 0.09815311661551362, 0.1455768644820135, 0.15906291729957867, 0.13545195798433932, 0.20297526251175435, 0.1872834839026295, 0.6173290491631639, 0.14976673386398304, 0.23319270361314942, 0.31876537411696027, 0.2811129986357285, 0.49217924130236024, 0.19591809835220889], 
	'20500': [0.16341101466088154, 0.2696268708146573, 0.08258893443438017, 0.15010659596147627, 0.10366841214136537, 0.11059444837717168, 0.15811848420704663, 0.08477188467443353, 0.11151221585188473, 0.1977235640948967, 0.3387258583053293, 0.20960638958514305, 0.12064514063154817, 0.20390658919593954, 0.12768689299640867, 0.2009153642173501, 0.09737213907277321, 0.12763385713546108, 0.20882859121763664, 0.15996443941917954], 
	'5010': [0.4398400282544007, 0.40625194992793257, 0.6194965445540015, 0.2001023189439644, 0.4002135940315564, 0.3661184475594431, 0.4713862828166724, 0.285217754850436, 0.38219896774266593, 0.6195696844791224, 0.44798496749682987, 0.4135066571102598, 0.4723678015365942, 0.6909905552659298, 0.2579091036040025, 0.3082728757871508, 0.1714712942139869, 0.513268079771435, 0.3435568413404579, 0.23398419313010496], 
	'5020': [0.3298191787063934, 0.3094658140982352, 0.27577921925459375, 0.2264182151794262, 0.3176601112728887, 0.5552085713939413, 0.25236659365947667, 0.3832222383163437, 0.6146444641069987, 0.31345112804004954, 0.5815887896270154, 0.19721002108633656, 0.27570683252573547, 0.5658407642242267, 0.4900039008715243, 0.2907054968216551, 0.35941211618189056, 0.24939257345813326, 0.5349203431006598, 0.5234585001888509], 
	'5050': [0.17089704143966228, 0.3393558717860297, 0.27391567448726484, 0.23000975650109826, 0.30445817245763185, 0.2293302158975395, 0.24865738242623728, 0.36098594336557055, 0.19557867441998264, 0.22193575254499506, 0.1569382737052849, 0.09364356100639222, 0.37391943121327076, 0.16905196830809904, 0.3168665099090018, 0.2286215596307089, 0.12985807476695377, 0.20240772638209537, 0.26003265433098033, 0.21133811544006867], 
	'50100': [0.2409203518331383, 0.28670517388458494, 0.34147844233374397, 0.26124188022571104, 0.28275731896760725, 0.2283533993373289, 0.22228713877116357, 0.15189942527026906, 0.22147330200609042, 0.21584733152912855, 0.21941619818184052, 0.3271570487520192, 0.2526521814691275, 0.17094590234747908, 0.2915851176587169, 0.16026678602563593, 0.10806356051963621, 0.44017771781517057, 0.06686211576809879, 0.4311274920580753], 
	'50200': [0.36615465907398964, 0.24028071731010436, 0.2675817712828076, 0.14608678775440043, 0.2452455110892734, 0.14308330062579716, 0.1287055345843491, 0.28603656320834436, 0.214509052813955, 0.1364838635783109, 0.07025107792368504, 0.1754011721692216, 0.19692072112307935, 0.26344409548035747, 0.32150078769412854, 0.13846202750574707, 0.11718155094204527, 0.116840898472525, 0.2447508693212248, 0.0700566982820299], 
	'50500': [0.19219184102803333, 0.12759929793964833, 0.13654255065391305, 0.19229376559614095, 0.10648967271412692, 0.15966959038856157, 0.04887225078306043, 0.20844571024391043, 0.1437281913661586, 0.16434932304263977, 0.12002405743374164, 0.11880136421411139, 0.1181294487893307, 0.2217429165749678, 0.2708543329883182, 0.1424571269994398, 0.13612143289973477, 0.17382653332342224, 0.17361081983629637, 0.14576111185789373], 
	'10010': [0.19731837592484175, 0.2464573288402978, 0.6973344423985928, 0.5512850425507003, 0.33901704630547447, 0.881273225397682, 0.353049456480147, 0.506062140305179, 0.3419600283138483, 0.41830442706549026, 0.27715682614316156, 0.580779075867839, 0.48309913925784365, 0.4615256613434641, 0.6546618528089039, 0.43089798548172153, 0.3654037907799554, 0.5504008384480125, 0.9951454138921783, 0.24643923170086018], 
	'10020': [0.23152870455924104, 0.18611569410612921, 0.18521299069555106, 0.16129742818416098, 0.4594537395746895, 0.43686338931770263, 0.5483877533712729, 0.3895663234232408, 0.3108267695407922, 0.4066491194302334, 0.4946446910062365, 0.37636977855092774, 0.19467366359172839, 0.3779707100903627, 0.4111226003109833, 0.7107892115318006, 0.16785089536937942, 0.17825695409070771, 0.27416748380671885, 0.4399059207137567], 
	'10050': [0.17179978664838916, 0.12736236742019846, 0.2027295686947312, 0.12577968602752132, 0.2628461104847668, 0.32633874244884276, 0.30173009137817836, 0.2751427895646366, 0.275698356703337, 0.43156437938480724, 0.24564971956308748, 0.2092747428919656, 0.36636837823886864, 0.12086906089239152, 0.3646637298912071, 0.3198125104549161, 0.39284648855629867, 0.31325132113992255, 0.13288764310603293, 0.1982862075124654], 
	'100100': [0.2069539554653909, 0.23620374747248885, 0.19793257717693702, 0.28423417404549317, 0.23457051727772274, 0.33138334509648376, 0.2938161175010687, 0.21084252514930127, 0.1421819935900164, 0.21535648296154264, 0.2576166697068743, 0.24570375710994558, 0.1851340755971774, 0.13058734596972701, 0.13422681119819646, 0.2555687750623497, 0.14719806624791787, 0.3804100054022249, 0.1173739698639199, 0.2833845316269515], 
	'100200': [0.20896242348410515, 0.07075731295842469, 0.07109933020199534, 0.16465993344354293, 0.26594701255579317, 0.14338386889798732, 0.19037087292373792, 0.15028953131491496, 0.12866201400821328, 0.15234927761586747, 0.11318605993724382, 0.2410364987109978, 0.23471537670292672, 0.10736673773415881, 0.16149298002904175, 0.13405030582600935, 0.10112670252780408, 0.12448089804959808, 0.24444946205442955, 0.08065423416464723], 
	'100500': [0.1374520748430895, 0.10127188537032183, 0.0600915141402696, 0.1482016220688251, 0.12594976612675288, 0.15828656308474565, 0.11082462515700371, 0.1499108000037226, 0.1774632443021115, 0.16113051555278166, 0.12750573844783022, 0.06674636055025261, 0.1030896797883674, 0.14424812598500875, 0.17311241697553056, 0.18491444136675356, 0.17258117692234654, 0.1624223810411833, 0.12571382169691495, 0.1403213059908812], 
	'50010': [0.4267764795888597, 0.2788005753794787, 0.3995542823464293, 0.06231522610146962, 0.21205884569310052, 0.387006407629399, 0.14535650512677356, 0.1647386712967546, 0.3297904414181914, 0.4859596338911261, 0.39286351859715, 0.3014811946619058, 0.6001332117814401, 0.3051956039651558, 0.5613180405609828, 0.26837857539527954, 0.3946425665435497, 0.8327454070267917, 0.4076090067206225, 0.19050591284248417], 
	'50020': [0.25873024362439667, 0.21686890757699132, 0.31370336152621664, 0.32092464861209, 0.2736603281789892, 0.3247388506129712, 0.32163959730513547, 0.5434684375836731, 0.23874889976488364, 0.2736869602418817, 0.29119562509599883, 0.5077191268055256, 0.45431563085757676, 0.2141613083565852, 0.4278562533196519, 0.341703137905873, 0.47378404158358417, 0.4020705236453771, 0.34256940795495155, 0.30300242848749026], 
	'50050': [0.23362439774455615, 0.3393342396697456, 0.3345829204348068, 0.18206894233219054, 0.28674832799814426, 0.27806569357885536, 0.18659606613916077, 0.15906513100301983, 0.3003365091100812, 0.3571882230971839, 0.2012461227328646, 0.13411944841669102, 0.3623980406237155, 0.37331546899040047, 0.14976231720816202, 0.21433100439917482, 0.14966720156499055, 0.06885175612974205, 0.266834394145625, 0.1704521707120388], 
	'500100': [0.18853374884585128, 0.16401331829776472, 0.29470749631817295, 0.13806732073646, 0.25873681115362435, 0.17262585607734465, 0.2058134237002108, 0.0558142277608899, 0.17386882891228062, 0.19811656676946957, 0.1800295570728441, 0.22803234680793982, 0.16365315914955164, 0.21415115051303135, 0.19862704629758912, 0.16405911915366217, 0.10114601706374518, 0.16697421133565538, 0.17569586784346006, 0.18612842456741613], 
	'500200': [0.19813877255559312, 0.18700056392024703, 0.16426247774215003, 0.154980709741186, 0.23743585293709293, 0.15006126996905503, 0.12385584569994722, 0.29980198562980115, 0.14409142864537217, 0.175471929543614, 0.2469542876379808, 0.16043064084554076, 0.11654761060201095, 0.24560616798238194, 0.15680872060116616, 0.17118822327232933, 0.21971527789628678, 0.2522964759006206, 0.06362335233418333, 0.061471004365954506], 
	'500500': [0.10834085103044433, 0.10852379472126202, 0.09213971933137233, 0.1129011137535575, 0.10880656001813846, 0.06688489033406682, 0.24966466350159663, 0.1245383201734375, 0.13128425666551047, 0.16266409989644398, 0.18215637912801735, 0.10309894433025642, 0.09785863546764227, 0.10506138569865461, 0.09826060010814014, 0.14505045153956891, 0.09113561478305633, 0.12798299705244617, 0.14205592390796193, 0.20091085549695406], 
	'100010': [0.3888455043486171, 0.42280285408745655, 0.1852317936548034, 0.6022289777806784, 0.3505337326064872, 0.752411726195559, 0.5611716458946359, 0.999889531552756, 0.35662024476578336, 0.40599239037483154, 0.4745621562137364, 0.7181502668237583, 0.3049173481688347, 0.3814567858686497, 0.5499068015129678, 0.36461646422301625, 0.18786453977758286, 0.3867715415396763, 0.705591881706167, 0.30279254400590366], 
	'100020': [0.33987561914016146, 0.27545354280785783, 0.13576175578363459, 0.22589554646717813, 0.6136287195474628, 0.3772252861124011, 0.1790160136163859, 0.3908979232038, 0.48927425066863567, 0.055459789182934226, 0.1521622358382337, 0.5806231317784172, 0.4415949169232782, 0.5962477687488013, 0.14227470002084508, 0.224134976477044, 0.3698148033320592, 0.29375317601655354, 0.2746900470621895, 0.43575947091207384], 
	'100050': [0.2698701637800338, 0.1970437125283099, 0.15634415445642189, 0.13521009919602872, 0.20533394981581335, 0.2681832693588526, 0.3554589199725156, 0.21211827223687746, 0.09897531044584576, 0.32693227298112393, 0.23175688434686512, 0.274540339777029, 0.19596421377335937, 0.20776480325252164, 0.21589204699325537, 0.3779351644938025, 0.24220504482580016, 0.11419149598233731, 0.23572673980454234, 0.18959748792395612], 
	'1000100': [0.13959437911064732, 0.061701940489781996, 0.21354868209383682, 0.24973867453154958, 0.14938911305350494, 0.1882074446292519, 0.1615374744421998, 0.1338939658698899, 0.11615725612416211, 0.17733776128524073, 0.19600667918011153, 0.1306423972340003, 0.19122788763470255, 0.1472613733940852, 0.15433858156816932, 0.34035909522088426, 0.19482374335627414, 0.30296057567361784, 0.16856287022534033, 0.20722120642082076], 
	'1000200': [0.17334077070962708, 0.2939745364509668, 0.18463357409489642, 0.10608666641920822, 0.13812799931097872, 0.1538229353175292, 0.17659614085356065, 0.18858743485064539, 0.1378325139204672, 0.23920255821229994, 0.1477733152602714, 0.15936975447948495, 0.09747552244922619, 0.36316622739481114, 0.10836101966387225, 0.38574167964700573, 0.2567982399652222, 0.08547116795138873, 0.13021752351361432, 0.11527152680823195], 
	'1000500': [0.10953418380954236, 0.08982495108444222, 0.10262001549585188, 0.06569383042072499, 0.09855740494968021, 0.1887775065139812, 0.06680868959455988, 0.1355357558964286, 0.1457864832790562, 0.08856394080318125, 0.17196772553912937, 0.13318812818190767, 0.12294262277769656, 0.1641170996609259, 0.15063640500311345, 0.14947929379415778, 0.10680501714428481, 0.14298888289955822, 0.14375089660477966, 0.07460517545337958], 
	'200010': [0.4270356148571393, 0.4018888927865927, 0.3013034189446318, 0.4564112992659574, 0.7326087919503279, 0.21902651496880457, 0.5518999094531913, 0.27634739881899884, 0.279765334327552, 0.5795156037306162, 0.4135653791261115, 0.4390604644755013, 0.47362194199138974, 0.305761226903299, 0.3899045553136326, 0.41865278640340964, 0.6611468959547004, 0.36040946904172116, 0.27506916992327934, 0.7126479660895737], 
	'200020': [0.4064613374025749, 0.24262434701618826, 0.3123371732929033, 0.393812859587409, 0.25658152663906014, 0.285069178752105, 0.7524782420322527, 0.17668991409977727, 0.27377908755821073, 0.311839343099288, 0.3633944201168298, 0.43155259259320233, 0.4024567481481379, 0.11242012689432286, 0.2820548407245268, 0.28207311307016614, 0.323038205636519, 0.421813588342349, 0.27149735853278684, 0.3575949888960733], 
	'200050': [0.3997858130305394, 0.346292080825341, 0.20826245716216715, 0.22033049035938643, 0.21304117882224566, 0.17604065733105495, 0.08586522732605625, 0.3408451491463666, 0.39740467039933003, 0.10934249379872402, 0.24616989840812634, 0.1407468068813374, 0.2340801784239809, 0.3343708634540579, 0.3348043836130404, 0.23108518083026758, 0.1320745666901055, 0.3344325419002561, 0.2791417004431571, 0.09199087433518642], 
	'2000100': [0.23578008445349039, 0.3884845793796587, 0.03679390988573334, 0.31209796623310504, 0.10211565710748045, 0.20467859585844964, 0.19648014650173182, 0.2979637003623347, 0.264413072744354, 0.2475160540748027, 0.20582530767785204, 0.16904724056756643, 0.3936768466807058, 0.31046718885507574, 0.2622315108458738, 0.2389805494749232, 0.13095563517511558, 0.3514046151329738, 0.2595023004606879, 0.14983388735909015], 
	'2000200': [0.2917376839195965, 0.09477044231669268, 0.1347641176060975, 0.1412911900854965, 0.11276950147122426, 0.09684609747549745, 0.20695277139828636, 0.2456999974526898, 0.10941411166677388, 0.08721568357807903, 0.15037833324572916, 0.2186801092122449, 0.10300359576525885, 0.17879129475020805, 0.09331009365568554, 0.24513727818776887, 0.13278501125957728, 0.16931731782688683, 0.1448867655564327, 0.19054133066641435], 
	'2000500': [0.09765911986426369, 0.11711957331327942, 0.07570448355594264, 0.12467641050891069, 0.15276276359616286, 0.12143497523527665, 0.1559634865696889, 0.10524893885387177, 0.09705635174247634, 0.21823521189788336, 0.05918894178803323, 0.17314095252998404, 0.11949098178571103, 0.1407667514799049, 0.22031459246552648, 0.09865486816312886, 0.10449304247810395, 0.07857476839611273, 0.10172738246624885, 0.16860039309672775], 
	'500010': [0.10535438904645218, 0.28769912511681095, 0.2519526363849853, 0.3136073820721963, 0.38083008608843005, 0.23493098834639503, 0.2616838042553424, 0.43303271442261676, 0.40537241116041595, 0.36840898448796794, 0.5075210160691823, 0.20844776675851245, 0.3706542525982224, 0.27489884308303003, 0.3591641782729468, 0.19987683054651503, 0.19860122981103356, 0.3270235689431272, 0.17720642632324807, 0.6487601700319073], 
	'500020': [0.3376252525640168, 0.31822843750792745, 0.2684999205629267, 0.1638378418351123, 0.28471182120861327, 0.31668847755570634, 0.5116963919769735, 0.24072219704620995, 0.18328943795797562, 0.3026854111124389, 0.08635661432741695, 0.10416862885305468, 0.2457244989803944, 0.21578948038097107, 0.3514639104845185, 0.589254627711207, 0.4763959121604469, 0.44087325501176505, 0.7851409530974848, 0.2791411184149149], 
	'500050': [0.29247202597515376, 0.26709849784020556, 0.19005091647161057, 0.22837124775889034, 0.15212371159531993, 0.29195172762011184, 0.25566718842294767, 0.27153647556194355, 0.39654173671210724, 0.1778431877644541, 0.09156483230722859, 0.38147462731733084, 0.23205534401940894, 0.12430111992042762, 0.2130806697932716, 0.07907169896549503, 0.1356273763044451, 0.1270705887228573, 0.3196336977402309, 0.1489231571399457], 
	'5000100': [0.05131114874472002, 0.2147992354239553, 0.20221623869333744, 0.2689033999219547, 0.34783202223681325, 0.2778617869291462, 0.10661936466078933, 0.17933654391038886, 0.1487287684275632, 0.3279944272217858, 0.1747582123577273, 0.08110103152067615, 0.2444282771182142, 0.23756236960268792, 0.11793283228395891, 0.18200087621444325, 0.1267689799294768, 0.27269782239370494, 0.22832880451740323, 0.15031673905947018], 
	'5000200': [0.09566094850744368, 0.12250870530168925, 0.08224540546539627, 0.20104185235468788, 0.08366838501631157, 0.23363772935153218, 0.08905469835057103, 0.16760468608892146, 0.1585135458710174, 0.08193614535803255, 0.21838089427615692, 0.37087253358479977, 0.09471575938746964, 0.2371905237008669, 0.20197480376156218, 0.1643725387196977, 0.20588000077712015, 0.11054010788707182, 0.15601068573099106, 0.14325823065127366], 
	'5000500': [0.19323915223330212, 0.14147041205007502, 0.14470658001383127, 0.11603001284005571, 0.08523486046662035, 0.11932264448579612, 0.18337110529629636, 0.15291270755265224, 0.11926938375547942, 0.10524227420934433, 0.06543488612613298, 0.132256909169942, 0.13010394386082647, 0.13316714654108786, 0.23584596600996086, 0.1571682535507922, 0.16526856663290926, 0.05339738164410387, 0.07839577900063428, 0.16003359929297226], 
	'1000010': [0.46173051341951804, 0.5715270580903571, 0.49423446152802525, 0.449405845075639, 0.2513323436173263, 0.46240686500319944, 0.5973490955034642, 0.3998861249902708, 0.4882505216740123, 0.4087862104945557, 0.6590458367631787, 0.3168842677924541, 0.17488988896351096, 0.3472077783457229, 0.5282475021707448, 0.21453260590391376, 0.3316183952455308, 0.11579808117127989, 0.5301158968790728, 0.6383897364539696], 
	'1000020': [0.637646736598196, 0.19069360905626975, 0.3785469689852712, 0.36324633108974713, 0.32010675266060107, 0.6213255850780133, 0.21933467929698894, 0.2356480649578316, 0.3105877589953779, 0.15494538115936513, 0.523152483061467, 0.17863503985644283, 0.3626713276193017, 0.4106747424850051, 0.24838220672792582, 0.2707704479660183, 0.3006567558434413, 0.19782276235814644, 0.19685133181297582, 0.1265196417692117], 
	'1000050': [0.21808226797186472, 0.18907810924075388, 0.30455324716085835, 0.3141881652909363, 0.1825206681484952, 0.7598412649024487, 0.19865324908633436, 0.21953425105083973, 0.3330057735678831, 0.2532012773915574, 0.11583042496140995, 0.45377650089656857, 0.145531010448747, 0.3413477546119008, 0.24483233006166918, 0.2808282311898126, 0.20418383281555166, 0.19164778362802637, 0.2325724021232814, 0.25736853863215653], 
	'10000100': [0.16192165156506036, 0.08228537569704877, 0.4388358714628576, 0.17787519002208838, 0.14812841347554853, 0.1817049729622255, 0.3760885777110835, 0.16202967199107166, 0.351637684799094, 0.33503207123457956, 0.12980562250900818, 0.15760568556721974, 0.29234780702221064, 0.12311126770391737, 0.24124599638161706, 0.20755327835016, 0.22460073999137786, 0.2655947802063723, 0.11752748866086363, 0.10599634490290925], 
	'10000200': [0.1759423064170004, 0.11112833644749144, 0.1781718913198736, 0.12297322828841772, 0.2180660984992661, 0.21718251796578905, 0.22279161651105958, 0.21489099005315782, 0.18856171252837486, 0.12403142913208819, 0.1117093074463717, 0.1438366391744951, 0.17501706481430798, 0.14874578336411662, 0.19866117574741504, 0.1136302919432942, 0.1776649325900314, 0.1555685476236522, 0.11958430757669565, 0.2533814217064463], 
	'10000500': [0.05596142127195654, 0.131507513498706, 0.08104543307741223, 0.12332499580446943, 0.16453320343328975, 0.11870974238549285, 0.14392321496308513, 0.07978987404900012, 0.12266363581878277, 0.09781083433048339, 0.22748419900175432, 0.1058413189543754, 0.1521060399273979, 0.12605722670078703, 0.18201482000667882, 0.19439283379598193, 0.1221040875452332, 0.13571029113367894, 0.10818839667235798, 0.2636553810878724]}
	

	'''
	'''
	{'2010': 0.40953063554629876, '2020': 0.3824354829917106, '2050': 0.27707734311932797, '20100': 0.23219835890065615, '20200': 0.23062517923903672, '20500': 0.16137038434974818, 
	'5010': 0.4021853971208473, '5020': 0.3823137436057188, '5050': 0.23589011800094334, '50100': 0.2460608942377283, '50200': 0.1944488830117688, '50500': 0.1550755669336725, 
	'10010': 0.4788785664653097, '10020': 0.34708269106328077, '10050': 0.25824508405012825, '100100': 0.22453397217608648, '100200': 0.154452041657072, '100500': 0.13656190297073464, 
	'50010': 0.3573615053283472, '50020': 0.34222738595199215, '50050': 0.2374294188015575, '500100': 0.18143972491884816, '500200': 0.17648712989112567, '500500': 0.1279660028469264, 
	'100010': 0.4701179365550951, '100020': 0.3296771836819974, '100050': 0.2255522172972646, '1000100': 0.18122555507690358, '1000200': 0.18209255536366542, '1000500': 0.12260920044531909, 
	'200010': 0.43378213171632146, '200020': 0.3329784496217342, '200050': 0.24280536065903635, '2000100': 0.23791244244155027, '2000200': 0.15741463635483205, '2000500': 0.12654069948936192, 
	'500010': 0.31575134019096684, '500020': 0.32511470943750376, '500050': 0.2188229913976693, '5000100': 0.19707494405841083, '5000200': 0.16095340900713062, '5000500': 0.13359357823664075, 
	'1000010': 0.4220819514542874, '1000020': 0.31241093036887996, '1000050': 0.2720288541590548, '10000100': 0.21404642461081572, '10000200': 0.1685769799574673, '10000500': 0.13684122317293979}
	'''














