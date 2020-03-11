from copy import deepcopy
import numpy as np
import re


def fullmatch(regex, string, flags=0):
	# emulation of python 3 regexp fullmatch
	return re.match("(?:" + regex + r")\Z", string, flags=flags)

def is_prim(s):
	# detect if string is a primitive name, such as 'u12' (u followed by optional number)
	# ignore the interpretation symbols	
	s = int_strip(s)
	pattern = fullmatch('u[0-9]*',s)
	return bool(pattern)

def is_var(s):
	# detect if string is a variable name, such as 'x12' (x followed by optional number)
	# ignore the interpretation symbols	
	s = int_strip(s)
	pattern = fullmatch('x[0-9]*',s)
	return bool(pattern)

def is_emptyvar(s):
	# detect if string is an empty variable name, such as 'x12' (x followed by optional number)
	# ignore the interpretation symbols	
	s = int_strip(s)
	pattern = fullmatch('y[0-9]*',s)
	return bool(pattern)

def int_strip(s):
	# strip the interpretation symbols from a string
	s = s.replace('[','')
	s = s.replace(']','')
	return s

def to_interpet(s):
	# does s still need interpretation?
	return s[0] == '[' and s[-1] == ']'

class IncompleteError(Exception):
	pass

class NumberGrammar():

	max_recursion = 50 # maximum number of recursive calls
	count_recursion = 0 # recursive counters
	rules = []

	def __init__(self,rules,list_prims):
		# rules is a list of Rule objects
		# list_prims : list of primitive symbols
		self.rules = deepcopy(rules)
		for r in self.rules:
			r.set_primitives(list_prims)

		self.var_regexp = self.rules[-1].var_regexp #MAJOR HACK

	def apply(self,s, max_recursion_count=50):
		self.max_recursion = max_recursion_count
		self.count_recursion = 0
		x = self._apply_helper(s)
		# if self.count_recursion >= self.max_recursion:
		# 	print(x)
		return x

	def _apply_helper(self,s, return_list=False):
		#print("APPLY HELPER", repr(s))
		s = s.strip()
		self.count_recursion += 1
		if s == '': return 0
		valid = []
		myrule = None
		for r in self.rules:
			valid.append(r.applies(s))
		# if np.count_nonzero(valid) != 1: # check that only one rule applies
			# assert False
		if not any(valid):
			#import pdb
			#pdb.set_trace()
			raise IncompleteError
			# print("NO VALID RULES FOR", s)
			# print(s == ' ')
			# return s
		myrule = self.rules[valid.index(True)]

		# run 'apply' recursively

		out = myrule.apply(s, self) #Pass grammar to rule, so we can apply it inside
		return out

		# for idx,o in enumerate(out):
		# 	if to_interpet(o) and self.count_recursion < self.max_recursion:
		# 			# print("WARNING: recursion depth exceeded")
		# 			# break
		# 		out[idx] = self._apply_helper(int_strip(o))
		# if return_list: 
		# 	return out
		# else:
		# 	return ' '.join(out)

	def __str__(self):
		s = ''
		for r in self.rules:
			s += str(r)+'\n'
		return s

	def apply_once(self, s):
		self.max_recursion = 1
		self.count_recursion = 0
		return self._apply_helper(s, return_list=True)
	
def mod10(d, g): return 10*g._apply_helper(d['x1'])
def mod100(d, g): return 100*g._apply_helper(d['x1'])
def mod1000(d, g): return 1000*g._apply_helper(d['x1'])
def mod10000(d, g): return 10000*g._apply_helper(d['x1'])
def mod1000000(d, g): return 1000000*g._apply_helper(d['x1'])
def denote_x1(d, g): return g._apply_helper(d['x1'])
def denote_u1(d, g): return g._apply_helper(d['u1'])
def denote_y1(d, g): return g._apply_helper(d['y1'])
def zeroFn(d, g): return 0

RHS_DICT = {}
RHS_DICT['[x1]*10'] = mod10
RHS_DICT['[x1]*100'] = mod100
RHS_DICT['[x1]*1000'] = mod1000
RHS_DICT['[x1]*10000'] = mod10000
RHS_DICT['[x1]*1000000'] = mod1000000
RHS_DICT['[x1]'] = denote_x1

RHS_DICT['[u1]'] = denote_u1
RHS_DICT['[y1]'] = denote_y1

RHS_DICT[''] = zeroFn
RHS_DICT[' '] = zeroFn

def lambdaMaker(i):
	a = i
	return lambda d, g: a

for i in range(1, 10):
	RHS_DICT['1000000*'+str(i)] = lambdaMaker(i*1000000)
	RHS_DICT['10000*'+str(i)] = lambdaMaker(i*10000)
	RHS_DICT['1000*'+str(i)] = lambdaMaker(i*1000)
	RHS_DICT['100*'+str(i)] = lambdaMaker(i*100)
	RHS_DICT['10*'+str(i)] = lambdaMaker(i*10)

class Rule():

	# left-hand-side
	LHS_str = ''
	LHS_list = []
	LHS_regexp = ''

	# right-hand-side
	RHS_str = ''
	RHS_list = []

	# 
	valid_rule = True #this may be annoying but can't see a way around it
	var_regexp = '([ a-zA-Z0-9]+)'

	# var_regexp = '([ a-z]+)' # define acceptable string for a variable to hold
	
	def __init__(self, LHS, RHS):
		# LHS : string with variables (no interpretation symbols [ or ] )
		# RHS : string with variables (can have interpretation symbols for recursive computation)

		self.LHS_str = LHS
		self.LHS_list = LHS.split()
		self.RHS_str = RHS
		self.RHS_list = RHS.split()

		self.RHS_exp = self._detokenize(self.RHS_list)

		# self.RHS_list = split_special(RHS)	

	def __getstate__(self):
		return (self.LHS_str, self.RHS_str)

	def __setstate__(self, state):
		if type(state)==dict:
			pass

		else:
			LHS, RHS = state
			self.__init__(LHS, RHS)


	def _detokenize(self, RHS): #Maybe more stuff
			# an RHS_exp takes a grammar and a vdict and returns an int
			lst = []
			try:
				i = 0
				while i < len(RHS):
					
					token = RHS[i]
					if token in ['1000000*', '10000*', '1000*', '100*', '10*']:
						token = token+RHS[i+1]
						lst.append( RHS_DICT.get(token, lambda v, g: int(token)) )
						i+=2
					else:
						lst.append( RHS_DICT.get(token, lambda v, g: int(token)) )
						i+=1

				#return lambda vdict, grammar: sum( tok(vdict, grammar) for tok in lst )
				def semantics(vdict, grammar):
					return sum( tok(vdict, grammar) for tok in lst )
				return semantics
			except (ValueError, IndexError):
				raise IncompleteError

	def set_primitives(self,list_prims):
		# list_prims : list of the primitive symbols

		self.list_prims = list_prims
		self.prim_regexp = '(' + '|'.join(self.list_prims) + ')'  # define acceptable string for a primitive
		self.var_regexp = '([(' + '|'.join(self.list_prims + [' ']) + ')]+)'
		#emptyvars match empty as well as non-empty
		self.emptyvar_regexp = '([(' + '|'.join(self.list_prims + [' ']) + ')]*)' #i hope this will match empty string

		# get list of all variables in LHS
		self.vars = [v for v in self.LHS_list if is_prim(v) or is_var(v) or is_emptyvar(v)]
			
		self.valid_rule = True #all([v in self.vars for v in rhs_vars])

		# Compute the regexp for checking whether the rule is active
		mylist = deepcopy(self.LHS_list)

		self.LHS_regexp = ''
		for i,x in enumerate(mylist):
			if is_prim(x):
				mylist[i] = self.prim_regexp
			elif is_var(x):
				mylist[i] = self.var_regexp

			#TODO NEW:
			elif is_emptyvar(x):
				mylist[i] = self.emptyvar_regexp

				self.LHS_regexp += mylist[i]
				continue

			if self.LHS_regexp == '': 
				self.LHS_regexp += mylist[i]
			else:
				self.LHS_regexp += ' ' + mylist[i]

		#self.LHS_regexp = ' '.join(mylist)

	def applies(self,s):
		# return True if the re-write rule applies to this string
		return self.valid_rule and bool(fullmatch(self.LHS_regexp,s))


	def apply(self, s, grammar):
		# apply rule to string s
		assert self.applies(s)
		assert self.valid_rule
		# if not self.valid_rule:
		# 	print(self)
		# 	assert False

		# extract variables from LHS
		m = fullmatch(self.LHS_regexp,s)
		#import pdb; pdb.set_trace()
			# if the expression has two variables "x1 x2", it returns the first split #TODO for var consistency
		mygroups = m.groups()
		assert len(mygroups) == len(self.vars), f"{mygroups}, {self.vars}, s={s}, rule={str(self)}"
		# if len(mygroups) != len(self.vars):
		# 	assert len(mygroups) + 1 == len(self.vars)
		# 	vdict = dict(zip(self.vars, mygroups))
		# 	del vdict['y1'] 

		# else:
		
		vdict = dict(zip(self.vars, mygroups))

		try:
			return self.RHS_exp(vdict, grammar) #is a fn from list of strs and a grammar to an integer
		except (ValueError, KeyError, RecursionError) as e:
			print('error in interpret grammar, fn: apply')
			print(e)
			if '(invalid)' in str(e): assert False
			print('error in application of gramamr')
			#import pdb; pdb.set_trace()
			raise IncompleteError


	def __str__(self):
		if self.valid_rule:
			val_tag = ''
		else:
			val_tag = ' (invalid)'
		return str(self.LHS_str) + ' -> ' + str(self.RHS_str) + val_tag

wrongRules = [
	['1', '->', 'one '],
	['2', '->', 'two '],
	['3', '->', 'three '],
	['4', '->', 'four '],
	['5', '->', 'five '],
	['6', '->', 'six '],
	['7', '->', 'seven '],
	['8', '->', 'eight '],
	['9', '->', 'nine '],
	['10', '->', 'ten '],
	['11', '->', 'eleven '],
	['12', '->', 'twelve '],
	['13', '->', 'thirteen '],
	['14', '->', 'forteen '],
	['15', '->', 'fifteen '],
	['16', '->', 'siksteen '],
	['17', '->', 'sebenteen '],
	['18', '->', 'eiteen '],
	['19', '->', 'ninteen '],
	#can give these too ..
	['20', '->', 'twenty '],
	['30', '->', 'thirty '],
	['40', '->', 'forty '],
	['50', '->', 'fifty '],
	['60', '->', 'sicksty '],
	['70', '->', 'sebenty '],
	['80', '->', 'eaity '],
	['90', '->', 'ninnty '],
	['100','->', 'one hundred '],
]

numRules = [Rule(rule[2].strip(), rule[0]) for rule in wrongRules[:-1]]

chineseCompound = [
			Rule('liang thousand y1', '1000* 2 [y1]'),
			Rule('x1 thousand y1', '[x1]*1000 [y1]'),
			Rule('liang hundred y1', '100* 2 [y1]'),
			Rule('x1 hundred y1', '[x1]*100 [y1]'),
			Rule('ten y1', '10* 1 [y1]'),
			Rule('x1 ten y1', '[x1]*10 [y1]'),
			Rule('zero', '0')
				]

hundredThousand = [Rule('one thousand', '1000'), Rule('one hundred', '100')]

concatRule = Rule('u1 x1', '[u1] [x1]')

ChineseRules = numRules[:10] +hundredThousand + chineseCompound + [concatRule]
chinesePrims = ['thousand', 'hundred', 'liang', 'ten', 'zero'] + [w[-1].strip() for w in wrongRules[:9] if ' ' not in w[-1].strip()]

ChineseG = NumberGrammar(ChineseRules, chinesePrims)

tokenizedRules = [
		['1', '->', 'one'],
		['2', '->', 'two'],
		['3', '->', 'three'],
		['4', '->', 'four'],
		['5', '->', 'five'],
		['6', '->', 'six'],
		['7', '->', 'seven'],
		['8', '->', 'eight'],
		['9', '->', 'nine'],
		['10', '->', 'ten'],
		[ '//1000==2' , '->', 'liang', 'thousand', '[x%1000]' ],
		[ '>1000' , '->', '[x//1000]', 'thousand', '[x%1000]' ],
		['//100==2', '->', 'liang', 'hundred', '[x%100]'],
		[ '>100' , '->', '[x//100]', 'hundred', '[x%100]' ],
		['//10==1', '->', 'ten', '[x%10]'],
		['>10', '->', '[x//10]', 'ten', '[x%10]'],
		['0', '->', 'zero']

			]


from number_words import IntGrammar

ChineseIntG = IntGrammar(tokenizedRules, zeroRule=True)



if __name__ == "__main__":


	G = EnglishG





	print(G.apply('one hundred fifteen'))
	print(G.apply('five hundred fifteen thousand'))

	# print(G.rules[-3].applies('five hundred fifteen thousand'))
	# s = 'five hundred fifteen thousand'
	# print(G.apply(s))
	lst = [
		"fifty thousand five",
		"fifty thousand six hundred",
		"fifty thousand six hundred two",
		"fifty thousand six hundred twenty two",
		"fifty four thousand six hundred twenty two",
		"two hundred thousand six hundred twenty two",
		"two hundred fifty thousand six hundred twenty two",
		"two hundred fifty thousand",
	]
	for word in lst:
		print(word, G.apply(word))

	for rule in G.rules: print(rule)
	#print('Variable detector has passed tests.')