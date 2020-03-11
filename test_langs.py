"""
test languages
"""
import random
import os
import numpy as np
#japanese:

from util import get_episode_generator
generate_episode_train, generate_episode_test, input_lang, output_lang, prog_lang = get_episode_generator("wordToNumber")
input_symbols = input_lang.symbols
	
MAXLEN = 8
MAXNUM = 99999999

def parseExamples(lang):

	return {'ja': parseExamples_JA,
			'ko': parseExamples_KO,
			'vi': parseExamples_VI,
			'it': parseExamples_IT,
			'es': parseExamples_ES,
			'zh': parseExamples_ZH,
			'en': parseExamples_EN,
			'fr': parseExamples_FR,
			'el': parseExamples_EL,
			}[lang]()


def myDist(maxLen, training_dist=False):
	if maxLen == 0: return 0
	size = random.choice(list(range(0, maxLen+1)))
	if size == 0: return 0
	num = random.choice(range(1, 10)) * 10**(size-1)

	if training_dist:
		return num + myDist(size-1, training_dist=True)
	else:
		return num + myDist(maxLen-1)

def printInts_JA():
	necessaryNums = list(range(1, 11)) + [100, 1000] + [10000, 1000000]
	nums = necessaryNums.copy()
	size = 110
	while True:
		newNum = myDist(MAXLEN)
		if newNum in nums or newNum > MAXNUM or newNum == 0: continue
		nums.append(newNum)
		if len(nums) == size: break


	os.system(f"php convertNum.php ja {' '.join([ str(n) for n in nums])} > testnums/japanese_out.txt")

	return necessaryNums

def parseExamples_JA():
	necessary = printInts_JA()
	# output should be a list of tuples: (d_in: string, d_out: int)
	convert_list = [
		'一',
		'二',
		'三',
		'四',
		'五',
		'六',
		'七',
		'八',
		'九',
		'十',
		'千',
		'百',
		'万',
		]

	in_symbols = input_symbols.copy()
	random.shuffle(in_symbols)
	convert_dict = {}

	for word in convert_list:
		symbol = in_symbols.pop()
		convert_dict[word] = symbol


	D = []
	f = open("testnums/japanese_out.txt", 'r')
	num_list = f.readlines()
	#assert len(num_list) == 100
	for ex in num_list:
		num, word = ex.split(',')
		num = int(num.strip())
		word = word.strip()
		word =  ' '.join([convert_dict[c] for c in word])
		D.append((word, num))
	return D, len(necessary)


def printInts_KO():
	necessaryNums = list(range(1, 11)) + [100, 1000] +[10000, 1000000]
	nums = necessaryNums.copy()
	size = 110
	
	while True:
		#newNum = np.random.logseries(p=.999)
		newNum = myDist(MAXLEN)
		if newNum in nums or newNum > MAXNUM or newNum == 0: continue
		nums.append(newNum)
		if len(nums) == size: break


	os.system(f"php convertNum.php ko {' '.join([ str(n) for n in nums])} > testnums/korean_out.txt")

	return necessaryNums

def parseExamples_KO():
	necessary = printInts_KO()
	# output should be a list of tuples: (d_in: string, d_out: int)
	convert_list = [
		'일',
		'이',
		'삼',
		'사',
		'오',
		'육',
		'칠',
		'팔',
		'구',
		'십',
		'백',
		'천',
		'만'
		]

	in_symbols = input_symbols.copy()
	random.shuffle(in_symbols)
	convert_dict = {}

	for word in convert_list:
		symbol = in_symbols.pop()
		convert_dict[word] = symbol


	D = []
	f = open("testnums/korean_out.txt", 'r')
	num_list = f.readlines()
	#assert len(num_list) == 100
	for ex in num_list:
		num, word = ex.split(',')
		num = int(num.strip())
		word = word.strip()
		word =  ' '.join([convert_dict[c] for c in word if c is not ' '])
		D.append((word, num))
	return D, len(necessary)

def printInts_VI():
	necessaryNums = list(range(0, 11)) + [100, 1000] + [10000, 1000000]
	nums = necessaryNums.copy()
	size = 110
	
	while True:
		#newNum = np.random.logseries(p=.999)
		newNum = myDist(MAXLEN)
		if newNum in nums or newNum > MAXNUM or newNum == 0: continue
		nums.append(newNum)
		if len(nums) == size: break


	os.system(f"php convertNum.php vi {' '.join([ str(n) for n in nums])} > testnums/VI_out.txt")


	return necessaryNums
#grep -o -E '\w+' testnums/VI_out.txt | sort -u -f | grep -v -E '[0123456789]+'

def parseExamples_VI():
	necessary = printInts_VI()
	# output should be a list of tuples: (d_in: string, d_out: int)

	convert_list = [
		('ba',),
		('bảy',)
		('bốn',)
		('chín',)
		('hai', 'lăm'),
		('không',),
		('trăm',),
		('lẻ',),
		('mốt', 'một'),
		('mươi', 'mười'),
		('năm',)
		('nghìn',),
		('sáu',),
		('tám',),
		('triệu',)
			]

	in_symbols = input_symbols.copy()
	random.shuffle(in_symbols)
	convert_dict = {}

	for group in convert_list:
		symbol = in_symbols.pop()
		for word in group:
			convert_dict[word] = symbol


	D = []
	f = open("testnums/VI_out.txt", 'r')
	num_list = f.readlines()
	#assert len(num_list) == 100
	for ex in num_list:
		num, stringOfWords = ex.split(',')
		num = int(num.strip())
		stringOfWords = stringOfWords.strip()
		words = stringOfWords.split()

		lst = [convert_dict[word] for word in words]
		convertedWords =  ' '.join(lst)
		#print(convertedWords)
		D.append((convertedWords, num))
	return D, len(necessary)


def printInts_IT():
	necessaryNums = list(range(1, 20)) + [i*10 for i in range(2, 10)] + [i*100 for i in range(2, 10)]+ [100, 1000] +[10000, 1000000]
	nums = necessaryNums.copy()
	size = 110
	
	while True:
		#newNum = np.random.logseries(p=.999)
		newNum = myDist(MAXLEN)
		if newNum in nums or newNum > MAXNUM or newNum == 0: continue
		nums.append(newNum)
		if len(nums) == size: break


	os.system(f"php convertNum.php it {' '.join([ str(n) for n in nums])} > testnums/IT_out.txt")

	return necessaryNums
#grep -o -E '\w+' testnums/IT_out.txt | sort -u -f | grep -v -E '[0123456789]+' | xargs printf '"%s":\n'

def parseExamples_IT():
	necessary = printInts_IT()
	# output should be a list of tuples: (d_in: string, d_out: int)
	# we are ignoring tuples

	convert_list = [
			("cent", "cento",),
			("cinquanta", "cinquant"),
			("cinque",),
			("diciannove",),
			("diciassette",),
			("diciotto",),
			("dieci",),
			("dodici",),
			("due",),
			("mila", "mille"), #TODO fix this
			("novant", "novanta"),
			("nove",),
			("ottant", "ottanta"),
			("otto"),
			("quaranta","quarant"),
			("quattordici",),
			("quattro",),
			("quindici",),
			("sedici",),
			("sei",),
			("sessanta", "sessant"),
			("settant", "settanta"),
			("sette",),
			("tre", "tré"),
			("tredici",),
			("trenta", "trent"),
			("undici",),
			("uno", "un"),
			("vent", "venti", "venta"),
			("milione", "milione", "milioni"),
		]

	in_symbols = input_symbols.copy()
	random.shuffle(in_symbols)
	convert_dict = {}

	for group in convert_list:
		symbol = in_symbols.pop()
		for word in group:
			convert_dict[word] = symbol


	D = []
	f = open("testnums/IT_out.txt", 'r')
	num_list = f.readlines()
	#assert len(num_list) == 100
	for ex in num_list:
		num, stringOfWords = ex.split(',')
		num = int(num.strip())
		stringOfWords = stringOfWords.strip()
		words = [s for st in  stringOfWords.split('\xad') for s in st.split() ]
		#print(word)
		convertedWords =  ' '.join([convert_dict[word] for word in words])
		D.append((convertedWords, num))
	return D, len(necessary)

def printInts_ES():
	necessaryNums = list(range(1, 30)) + [i*10 for i in range(3, 10)] + [i*100 for i in range(2, 10)]+ [100, 1000] +[10000, 1000000]
	nums = necessaryNums.copy()
	size = 110
	
	while True:
		#newNum = np.random.logseries(p=.999)
		newNum = myDist(MAXLEN)
		if newNum in nums or newNum > MAXNUM or newNum == 0: continue
		nums.append(newNum)
		if len(nums) == size: break


	os.system(f"php convertNum.php es {' '.join([ str(n) for n in nums])} > testnums/ES_out.txt")

	return necessaryNums
#grep -o -E '\w+' testnums/ES_out.txt | sort -u -f | grep -v -E '[0123456789]+' | xargs printf '"%s":\n'

def parseExamples_ES():
	necessary = printInts_ES()
	# output should be a list of tuples: (d_in: string, d_out: int)
	# we are ignoring tuples
	#convert_list = ( )
	convert_list = [
			"catorce",
			"cien",
			"ciento",
			"cinco",
			"cincuenta",
			"cuarenta",
			"cuatro",
			"cuatrocientos",
			"diecinueve",
			"dieciocho",
			"dieciséis",
			"diecisiete",
			"diez",
			"doce",
			"dos",
			"doscientos",
			"mil",
			"novecientos",
			"noventa",
			"nueve",
			"ochenta",
			"ocho",
			"ochocientos",
			"once",
			"quince",
			"quinientos",
			"seis",
			"seiscientos",
			"sesenta",
			"setecientos",
			"setenta",
			"siete",
			"trece",
			"treinta",
			"tres",
			"trescientos",
			"uno",
			"veinte",
			"veinticinco",
			"veintinueve",
			"veintiuno",
			"veintidós",
			"veintitrés",
			"veintiséis",
			"veinticuatro",
			"veintisiete",
			"veintiocho",
			"y",
			"millón",
			"millones",
			"un",
			"veintiún"]

	in_symbols = input_symbols.copy()
	random.shuffle(in_symbols)
	convert_dict = {}
	for word in convert_list:
		if word == 'ciento':
		 	convert_dict['ciento'] = convert_dict['cien'] #TODO
		elif word == 'millones':
		 	convert_dict['millones'] = convert_dict["millón"]
		elif word == 'un':
		 	convert_dict['un'] = convert_dict["uno"]
		elif word == 'un':
		 	convert_dict['veintiún'] = convert_dict["veintiuno"]
		else:
			convert_dict[word] = in_symbols.pop()

	D = []
	f = open("testnums/ES_out.txt", 'r')
	num_list = f.readlines()
	for ex in num_list:
		num, stringOfWords = ex.split(',')
		num = int(num.strip())
		stringOfWords = stringOfWords.strip()
		words = stringOfWords.split(' ')
		#print(word)
		lst = []
		for word in words:
			if convert_dict.get(word, None):
				lst.append(convert_dict[word])
			elif '\xad' in word:
				newWord = convert_dict[''.join(word.split('\xad'))]
			else: assert False, f"bad word: {word}"
		#convert_dict.get(word) for word in words]
		convertedWords =  ' '.join(lst)
		D.append((convertedWords, num))
	return D, len(necessary)


def printInts_ZH():
	necessaryNums = list(range(0, 20)) + [100, 1000] + [10000, 1000000]
	nums = necessaryNums.copy()
	size = 110
	
	while True:
		#newNum = np.random.logseries(p=.999)
		newNum = myDist(MAXLEN)
		if newNum in nums or newNum > MAXNUM or newNum == 0: continue
		nums.append(newNum)
		if len(nums) == size: break


	os.system(f"php convertNum.php zh {' '.join([ str(n) for n in nums])} > testnums/zh_out.txt")
	return necessaryNums

def parseExamples_ZH():
	necessary = printInts_ZH()
	# output should be a list of tuples: (d_in: string, d_out: int)
	convert_list = [
		'〇',
		'一',
		'二',
		'三',
		'四',
		'五',
		'六',
		'七',
		'八',
		'九',
		'十',
		'千',
		'百',
		'万',
		]


	in_symbols = input_symbols.copy()
	random.shuffle(in_symbols)
	convert_dict = {}

	for word in convert_list:
		symbol = in_symbols.pop()
		convert_dict[word] = symbol

	D = []
	f = open("testnums/zh_out.txt", 'r')
	num_list = f.readlines()
	#assert len(num_list) == 100
	for ex in num_list:
		num, word = ex.split(',')
		num = int(num.strip())
		word = word.strip()
		word =  ' '.join([convert_dict[c] for c in word])
		D.append((word, num))
	return D, len(necessary)


def printInts_EN():
	necessaryNums = list(range(1, 20)) + [i*10 for i in range(2, 10)] + [100, 1000] + [10000, 1000000]
	nums = necessaryNums.copy()
	size = 110
	
	while True:
		#newNum = np.random.logseries(p=.999)
		newNum = myDist(MAXLEN)
		if newNum in nums or newNum > MAXNUM or newNum == 0: continue
		nums.append(newNum)
		if len(nums) == size: break


	os.system(f"php convertNum.php en {' '.join([ str(n) for n in nums])} > testnums/EN_out.txt")

	return necessaryNums
#grep -o -E '\w+' testnums/EN_out.txt | sort -u -f | grep -v -E '[0123456789]+' | xargs printf '"%s",\n'

def parseExamples_EN():
	necessary = printInts_EN()
	# output should be a list of tuples: (d_in: string, d_out: int)
	# we are ignoring tuples
	convert_list = ["eight",
					"eighteen",
					"eighty",
					"eleven",
					"fifteen",
					"fifty",
					"five",
					"forty",
					"four",
					"fourteen",
					"hundred",
					"nine",
					"nineteen",
					"ninety",
					"one",
					"seven",
					"seventeen",
					"seventy",
					"six",
					"sixteen",
					"sixty",
					"ten",
					"thirteen",
					"thirty",
					"thousand",
					"three",
					"twelve",
					"twenty",
					"two",
					"million"]

	in_symbols = input_symbols.copy()
	random.shuffle(in_symbols)
	convert_dict = {}
	for word in convert_list:
		convert_dict[word] = in_symbols.pop()

	D = []
	f = open("testnums/EN_out.txt", 'r')
	num_list = f.readlines()
	#assert len(num_list) == 100
	for ex in num_list:
		num, stringOfWords = ex.split(',')
		num = int(num.strip())
		stringOfWords = stringOfWords.strip()
		words = [s for stringOfWord in stringOfWords.split() for s in stringOfWord.split('-')]
		#print(words)
		convertedWords =  ' '.join([convert_dict.get(word, word) for word in words])
		D.append((convertedWords, num))
	return D, len(necessary)


def printInts_FR():
	necessaryNums = list(range(1, 20)) + [i*10 for i in range(2, 10)] + [100, 1000] +  [10000, 1000000]
	nums = necessaryNums.copy()
	size = 110
	
	while True:
		#newNum = np.random.logseries(p=.999)
		newNum = myDist(MAXLEN)
		if newNum in nums or newNum > MAXNUM or newNum == 0: continue
		nums.append(newNum)
		if len(nums) == size: break


	os.system(f"php convertNum.php fr {' '.join([ str(n) for n in nums])} > testnums/FR_out.txt")
	return necessaryNums
#grep -o -E '\w+' testnums/EN_out.txt | sort -u -f | grep -v -E '[0123456789]+' | xargs printf '"%s",\n'

def parseExamples_FR():
	necessary = printInts_FR()
	# output should be a list of tuples: (d_in: string, d_out: int)
	# we are ignoring tuples
	convert_list = [("cent", "cents",),
					("cinq",),
					("cinquante",),
					("deux",),
					("dix",),
					("douze",),
					("et",),
					("huit",),
					("mille",),
					("neuf",),
					("onze",),
					("quarante",),
					("quatorze",),
					("quatre",),
					("quinze",),
					("seize",),
					("sept",),
					("six",),
					("soixante",),
					("treize",),
					("trente",),
					("trois",),
					("un",),
					("vingt", "vingts",),
					("million", "millions"),
						]

	in_symbols = input_symbols.copy()
	random.shuffle(in_symbols)
	convert_dict = {}

	for group in convert_list:
		symbol = in_symbols.pop()
		for word in group:
			convert_dict[word] = symbol

	D = []
	f = open("testnums/FR_out.txt", 'r')
	num_list = f.readlines()
	#assert len(num_list) == 100
	for ex in num_list:
		num, stringOfWords = ex.split(',')
		num = int(num.strip())
		stringOfWords = stringOfWords.strip()
		words = [s for stringOfWord in stringOfWords.split() for s in stringOfWord.split('-')]
		#print(words)
		convertedWords =  ' '.join([convert_dict.get(word, word) for word in words])
		D.append((convertedWords, num))
	return D, len(necessary)

def printInts_EL():
	necessaryNums = list(range(1, 30)) + [i*10 for i in range(3, 10)] + [i*100 for i in range(2, 10)]+ [100, 1000] +  [10000, 1000000]
	nums = necessaryNums.copy()
	size = 110
	
	while True:
		#newNum = np.random.logseries(p=.999)
		newNum = myDist(MAXLEN)
		if newNum in nums or newNum > MAXNUM or newNum == 0: continue
		nums.append(newNum)
		if len(nums) == size: break


	os.system(f"php convertNum.php el {' '.join([ str(n) for n in nums])} > testnums/EL_out.txt")

	return necessaryNums
#grep -o -E '\w+' testnums/EL_out.txt | sort -u -f | grep -v -E '[0123456789]+' | xargs printf '"%s",\n'
def parseExamples_EL():

	necessary = printInts_EL()
	# output should be a list of tuples: (d_in: string, d_out: int)
	# we are ignoring tuples
	convert_list = [("δεκα","δέκα",),
					("διακόσια",),
					("δύο",),
					("δώδεκα",),
					("εβδομήντα",),
					("είκοσι",),
					("εκατό","εκατόν",),
					("ένα", "μία"), 
					("εννέα",),
					("εννενήντα",),
					("εννιακόσια","εννιακόσιες"),
					("έντεκα",),
					("εξακόσια", "εξακόσιες"),
					("εξήντα",),
					("έξι",""),
					("επτά",),
					("επτακόσια", "επτακόσιες"),
					("ογδόντα",),
					("οκτακόσια", "οκτακόσιες"),
					("οκτώ",),
					("πενήντα",),
					("πεντακόσια","πεντακόσιες"),
					("πέντε",),
					("σαράντα",),
					("τέσσερα","τέσσερις"),
					("τετρακόσια",),
					("τρεις","τρία",),
					("τριακόσια","τριακόσιες"),
					("τριάντα",),
					("χίλια",),
					("χίλιάδες",),
					("δεκα\xadτρία",),
					("δεκα\xadτέσσερα",),
					("δεκα\xadπέντε",),
					("δεκα\xadέξι",),
					("δεκα\xadεπτά",),
					("δεκα\xadοκτώ",),
					("δεκα\xadεννέα",),
					("δεκα­\xadτέσσερις","δεκα­τέσσερις"),
					("εκατομμύριο", "εκατομμύρια"),
					]

	in_symbols = input_symbols.copy()
	random.shuffle(in_symbols)
	convert_dict = {}

	for group in convert_list:
		symbol = in_symbols.pop()
		for word in group:
			convert_dict[word] = symbol

	D = []
	f = open("testnums/EL_out.txt", 'r')
	num_list = f.readlines()
	#assert len(num_list) == 100
	for ex in num_list:
		num, stringOfWords = ex.split(',')
		num = int(num.strip())
		stringOfWords = stringOfWords.strip()
		#words = [s for stringOfWord in stringOfWords.split() for s in stringOfWord.split('\xad')]
		words = [s for s in stringOfWords.split()]
		#print(words)
		lst = []
		for word in words:
			if word in convert_dict:
				lst.append(convert_dict[word])
			elif word[-3:] == 'ιες':
				newWord = word[:-3]+'ια'
				lst.append(convert_dict[newWord])
			elif word[-3:] == 'εις':
				newWord = word[:-3]+'ία'
				lst.append(convert_dict[newWord])
			else:
				assert False, f"{word}"
		convertedWords =  ' '.join(lst)
		D.append((convertedWords, num))
	return D, len(necessary)

if __name__=='__main__':
	printInts_FR()
	D = parseExamples_FR()
	for d in D: print(d)