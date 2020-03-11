from copy import deepcopy
import numpy as np
import re

# Class of grammars for interpreting SCAN commands
# Rules are applied in sequential order (first come first serve)

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

def int_strip(s):
	# strip the interpretation symbols from a string
	s = s.replace('[','')
	s = s.replace(']','')
	return s

def to_interpet(s):
	# does s still need interpretation?
	return s[0] == '[' and s[-1] == ']'




class Grammar():

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
		x = self.__apply_helper(s)
		# if self.count_recursion >= self.max_recursion:
		# 	print(x)
		return x

	def __apply_helper(self,s, return_list=False):
		self.count_recursion += 1
		valid = []
		myrule = None
		for r in self.rules:
			valid.append(r.applies(s))
		# if np.count_nonzero(valid) != 1: # check that only one rule applies
			# assert False
		if not any(valid):
			return s
		myrule = self.rules[valid.index(True)]

		# run 'apply' recursively
		out = myrule.apply(s)
		for idx,o in enumerate(out):
			if to_interpet(o) and self.count_recursion < self.max_recursion:
					# print("WARNING: recursion depth exceeded")
					# break
				out[idx] = self.__apply_helper(int_strip(o))
		if return_list: 
			return out
		else:
			return ' '.join(out)

	def __str__(self):
		s = ''
		for r in self.rules:
			s += str(r)+'\n'
		return s

	def apply_once(self, s):
		self.max_recursion = 1
		self.count_recursion = 0
		return self.__apply_helper(s, return_list=True)


	def apply_repl(self, s, max_recursion_count=50):
		self.max_recursion = max_recursion_count
		self.count_recursion = 0
		return self.__apply_helper_repl(s)



	def __apply_helper_repl(self, s, return_list=False):
		self.count_recursion += 1
		valid = []
		myrule = None
		for r in self.rules:
			valid.append(r.applies(' '.join(s)))
		# if np.count_nonzero(valid) != 1: # check that only one rule applies
			# assert False
		if not any(valid):
			#print("no valid rules for", s)
			if fullmatch(self.var_regexp, ' '.join(s)):
				#if self.count_recursion == 1:
				#	pass
				#else:
				return ['['] + s + [']'] #['[' + ' '.join(s) + ']']
			return s
		myrule = self.rules[valid.index(True)]

		# run 'apply' recursively
		out = myrule.apply(' '.join(s))
		for idx,o in enumerate(out):
			if to_interpet(o) and self.count_recursion < self.max_recursion:
				#if not self.count_recursion < self.max_recursion:
					# print(f'max recursion of {self.max_recursion} exceeded for miniscan repl')
					# import pdb; pdb.set_trace()

				out[idx] = self.__apply_helper_repl(int_strip(o).split(' '))
			else:
				out[idx] = o.split(' ')
		#if return_list: 
		return [token for oo in out for token in oo]
		#else:
			#return ' '.join(out)
	





class Rule():

	# left-hand-side
	LHS_str = ''
	LHS_list = []
	LHS_regexp = ''

	# right-hand-side
	RHS_str = ''
	RHS_list = []

	# 
	valid_rule = False
	var_regexp = '([ a-zA-Z0-9]+)'

	# var_regexp = '([ a-z]+)' # define acceptable string for a variable to hold
	
	def __init__(self,LHS,RHS):
		# LHS : string with variables (no interpretation symbols [ or ] )
		# RHS : string with variables (can have interpretation symbols for recursive computation)

		self.LHS_str = LHS
		self.LHS_list = LHS.split()
		self.RHS_str = RHS
		self.RHS_list = RHS.split()
		# self.RHS_list = split_special(RHS)		

	def set_primitives(self,list_prims):
		# list_prims : list of the primitive symbols

		self.list_prims = list_prims
		self.prim_regexp = '(' + '|'.join(self.list_prims) + ')'  # define acceptable string for a primitive
		self.var_regexp = '([(' + '|'.join(self.list_prims + [' ']) + ')]+)'
		#print("hit this guy")
		#print(self.prim_regexp)
		#print(self.var_regexp)
		# get list of all variables in LHS
		self.vars = [v for v in self.LHS_list if is_prim(v) or is_var(v)]
			
		# sanity check 
		rhs_vars  = [int_strip(v) for v in self.RHS_list if is_prim(v) or is_var(v)]
		self.valid_rule = all([v in self.vars for v in rhs_vars])

		# Compute the regexp for checking whether the rule is active
		mylist = deepcopy(self.LHS_list)
		for i,x in enumerate(mylist):
			if is_prim(x):
				mylist[i] = self.prim_regexp
			elif is_var(x):
				mylist[i] = self.var_regexp
		self.LHS_regexp = ' '.join(mylist)

	def applies(self,s, var_consistancy=False):
		# return True if the re-write rule applies to this string
		if not var_consistancy:
			return self.valid_rule and bool(fullmatch(self.LHS_regexp,s))
		else:
			if not self.valid_rule: return False
			m = fullmatch(self.LHS_regexp,s)
			if not bool(m): return False
			mygroups = m.groups()
			if not len(mygroups) == len(self.vars): return False
			vdict = dict(zip(self.vars,mygroups))
			for var, group in zip(self.vars, mygroups):
				if vdict[var] != group: return False
			return True


	def apply(self,s, var_consistancy=False):
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
		assert(len(mygroups) == len(self.vars)), f"{mygroups}, {self.vars}, s={s}, rule={str(self)}"
		vdict = dict(zip(self.vars,mygroups))
		if var_consistancy:
			for var, group in zip(self.vars, mygroups):
				if vdict[var] != group: assert False

		# replace RHS with variable values
		mylist = deepcopy(self.RHS_list)
		for i,x in enumerate(mylist):
			if is_var(x) or is_prim(x):
				#import pdb; pdb.set_trace()
				mylist[i] = x.replace(int_strip(x), vdict[int_strip(x)])
		return mylist

	def __str__(self):
		if self.valid_rule:
			val_tag = ''
		else:
			val_tag = ' (invalid)'
		return str(self.LHS_str) + ' -> ' + str(self.RHS_str) + val_tag



if __name__ == "__main__":
 
	myrules = [Rule('walk','WALK'), Rule('u left','LTURN [u]'), Rule('x twice','[x] [x]')]
	G = Grammar(myrules,['walk','left'])
	print(G.apply('walk left twice'))
	print('Testing variable detector..')
	assert is_var('x')
	assert is_var('x0')
	assert is_var('x10')
	assert not is_var('X10')
	assert not is_var('10x')
	assert not is_var('y10')
	assert not is_var(' x10')
	assert not is_var('x10 ')
	print('Variable detector has passed tests.')