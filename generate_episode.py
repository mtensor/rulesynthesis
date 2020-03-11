from interpret_grammar import Grammar, Rule
import random
import numpy as np
from copy import deepcopy
import re
from collections import Counter

class DataSamplingError(Exception):
    pass

p_lhs_onearg = 0.4 # probability that we have a single argument. Otherwise, two arguments
p_stop_rhs = 0.6 # prob. of stopping on the right hand side #0.4 for longtail
vars_input = ['u1','u2','x1','x2']
vars_output = ['['+v+']' for v in vars_input]
icon_concat_rule = Rule('u1 x1','[u1] [x1]')

def sample_data(ndat,G,maxlen_input=6,maxlen_output=6,maxntry=300, out_lang=None, input_lang=None):
	# Input
	#  ndat : number of input sequences
	#  G : grammar
	#  maxlen : maximum length for input or output sequence


	CFG = make_pcfg_for_data_gen(G)
	D = set([])
	ntry = 0
	while len(D)<ndat:
		dat_in = sample_from_pcfg(CFG, maxlen_input)
		dat_out = G.apply(dat_in)
		ntry += 1
		if out_lang:
			if not all( d in out_lang.symbols for d in dat_out.split()):
				continue
		if (dat_in != '') and (dat_out != '') and (len(dat_in.split()) <= maxlen_input) and (len(dat_out.split()) <= maxlen_output):
				D.add((dat_in, dat_out))		
			#ntry = 0
		if ntry > maxntry:
			raise DataSamplingError	
	return list(D)
	
def make_pcfg_for_data_gen(G):
	# Transform the rules into a PCFG that defines a dist. over valid input strings to create the data set
	LHS_list = [r.LHS_str for r in G.rules]
	LHS_list = [s.replace('u1','U') for s in LHS_list]
	LHS_list = [s.replace('u2','U') for s in LHS_list]
	LHS_list = [s.replace('x1','X') for s in LHS_list]
	LHS_list = [s.replace('x2','X') for s in LHS_list]
	CFG = {}
	CFG['U'] = [s for s in LHS_list if len(s.split())==1]
	CFG['X'] = ['U'] + [s for s in LHS_list if len(s.split())>1] # TODO maybe put a hack in for this so more freqently simple
	return CFG

def sample_from_pcfg(CFG,maxlen, mystr='X'): 
	#  CFG : context-free grammar we want to sample from
	#  maxlen : maximum length of sampled string
	# 
	# If we sample a string that is too long, we return an empty string
	#
	#mystr = 'X' # starting symbol
	while True: 
		list_expand = [] # expansion of each current symbol
		all_term = True # all terminals?
		for symb in mystr.split():						
			if symb in CFG:
				all_term = False # we made an expansion
				options = CFG[symb]
				symb = random.choice(options)
			list_expand.append(symb)

		# if we are over the allowed length
		if len(list_expand) > maxlen:
			return ''

		mystr = ' '.join(list_expand)
		if all_term:
			break
	return mystr

def exact_perm_rules():
	return [
		Rule('run', 'RUN'),
		Rule('jump', 'JUMP'), 
		Rule('walk', 'WALK'),
		Rule('look', 'LOOK'),
		Rule('turn right', 'RTURN'),
		Rule('u1 right', 'RTURN [u1]'),
		Rule('turn opposite right', 'RTURN RTURN'),
		Rule('u1 opposite right', 'RTURN RTURN [u1]'),
		Rule('turn around right', 'RTURN RTURN RTURN RTURN'),
		Rule('u1 around right', 'RTURN [u1] RTURN [u1] RTURN [u1] RTURN [u1]'),
		Rule('turn left', 'LTURN'),
		Rule('u1 left', 'LTURN [u1]'),
		Rule('turn opposite left', 'LTURN LTURN'),
		Rule('u1 opposite left', 'LTURN LTURN [u1]'),
		Rule('turn around left', 'LTURN LTURN LTURN LTURN'),
		Rule('u1 around left', 'LTURN [u1] LTURN [u1] LTURN [u1] LTURN [u1]'),
		Rule('x1 and x2', '[x1] [x2]'),
		Rule('x2 after x1', '[x1] [x2]'),
		Rule('x1 twice', '[x1] [x1]'),
		Rule('x1 thrice', '[x1] [x1] [x1]')
	]

def exact_perm_doubled_rules():
	return [
		Rule('run', 'RUN'),
		Rule('jump', 'JUMP'), 
		Rule('walk', 'WALK'),
		Rule('look', 'LOOK'),
		Rule('turn', ''),

		Rule('right', 'RTURN'),
		Rule('left', 'LTURN'),

		Rule('u1 opposite u2', '[u2] [u2] [u1]'),
		Rule('u1 around u2', '[u2] [u1] [u2] [u1] [u2] [u1] [u2] [u1]'),
		Rule('x1 and x2', '[x1] [x2]'),
		Rule('x2 after x1', '[x1] [x2]'),
		Rule('x1 twice', '[x1] [x1]'),
		Rule('x1 thrice', '[x1] [x1] [x1]'),
		Rule('u1 u2','[u2] [u1]')
	]

def generate_scan_rules(nprims, nurules, nxrules, input_symbols, output_symbols, u_type='random'):

	# for synthesis: generate a scan grammar
	# nprims : number of primitives
	# nurules : number of rules with u
	# nxrules : number of rules with x

	assert u_type == 'random'
	return generate_random_scan_rules(nprims, nurules, nxrules, input_symbols, output_symbols)


def generate_random_scan_rules(nprims, nurules, nxrules, input_symbols, output_symbols):

	# for synthesis: generate a scan grammar
	# nprims : number of primitives - can be 4-8? need to add some nonce actions ...
	# nurules : number of rules with u - must be 2
	# nxrules : number of rules with x or u - should be 3-7 i think
	# also needs to have extended decay rate to hit around rule ... unless there's a cleverer way 

	assert nurules == 0
	####
	#random shuffling
	input_symbol_options = input_symbols.copy()
	output_symbol_options = output_symbols.copy()
	random.shuffle(input_symbol_options)
	random.shuffle(output_symbol_options)

	rules, input_symbol_options, output_symbol_options = generate_prims(nprims, input_symbol_options, output_symbol_options, blank_prim=True)

	# if random.random() < 0.5:
	# 	u_rules, input_symbol_options, output_symbol_options = generate_opposite_around_rules(nurules, input_symbol_options, output_symbol_options)
	# 	rules.extend(u_rules)

	for i in range(nxrules):
		#sample_lhs makes sure not to use words twice for simple defs
		input_symbol = input_symbol_options[i]
		LHS, used_vars_lhs = sample_LHS(input_symbol, only_x=False)
		RHS = sample_RHS(used_vars_lhs, p_stop_rhs=0.35)
		rules.append(Rule(LHS,RHS))

	#rules.extend(u_rules)
	#also need to ensure that both u and x are used, or else there will be no good compositionality ...
	#can do this by having 6 prims, 10 fns with u, and 4 with x or x1 x2!!!

	if random.random() < 0.5:
		#scan:
		rules.append(Rule('u1 u2','[u2] [u1]'))
	else:
		rules.append(icon_concat_rule)

	return Grammar(rules, input_symbols) #todo


def generate_random_rules(nprims,nrules,input_symbols,output_symbols, sort_prims=False):
	# nprims : number of primitives
	# nrules : number of rules
	assert(nprims+nrules <= len(input_symbols))
	input_symbol_options = np.copy(np.array(input_symbols))
	output_symbol_options = np.copy(np.array(output_symbols))
	np.random.shuffle(input_symbol_options)
	np.random.shuffle(output_symbol_options)
	rules, input_symbol_options, output_symbol_options = generate_prims(nprims, input_symbol_options, output_symbol_options)
	if sort_prims:
		rules = sorted(rules, key=lambda rule: rule.LHS_str )
	primitive_input_symbols = list(set(input_symbols) - set(input_symbol_options))
	for i in range(nrules):
		input_symbol = input_symbol_options[i]
		LHS,used_vars_lhs = sample_LHS(input_symbol)
		RHS = sample_RHS(used_vars_lhs)
		rules.append(Rule(LHS,RHS))
	rules.append(icon_concat_rule)
	return Grammar(rules,input_symbols)

def generate_prims(nprims,input_symbol_options,output_symbol_options, blank_prim=False):
	# generate the rules for the primitives
	# input
	#   nprims : number of primitives
	#   _symbol_options : available input and output symbols
	# return :  rules and updated list of input and output symbols

	#note that the blank thing was hacked in, so the output symbol which was assigned to that one is no longer there ...

	rules = []
	nblank = 0
	for i in range(nprims):
		if blank_prim and random.random() < 1./7:
			rules.append( Rule(input_symbol_options[i], '') )
			nblank += 1
		else:
			rules.append( Rule(input_symbol_options[i],output_symbol_options[i]) )

	return rules, input_symbol_options[nprims:], output_symbol_options[nprims:]

def sample_LHS(func_name, p_lhs_onearg=0.4, only_x=False):
	# Sample either a one (x func_name) or two (x func_name x) argument rule
	if only_x:
		vars_options = vars_input[2:] #only x1 and x2
	else:
		vars_options = vars_input[:]
	np.random.shuffle(vars_options) # randomize variables
	if random.random() < p_lhs_onearg:
		arr = [vars_options.pop(), func_name]
	else:
		arr = [vars_options.pop(), func_name, vars_options.pop()]
	used_vars = list(set(arr).intersection(set(vars_input)))
	used_vars = ['['+u+']' for u in used_vars] # variables we used
	return ' '.join(arr), used_vars

def sample_RHS(vars_in_lhs, p_stop_rhs=0.6):
	# vars_in_lhs : variables that were used to construct the left hand side
	arr = vars_in_lhs[:]
	while True:
		if random.random() < p_stop_rhs:
			break
		item = np.random.choice(vars_in_lhs)
		arr.append(item)
	np.random.shuffle(arr) # randomize RHS				
	return ' '.join(arr)

def load_scan_file(mytype,split):
	# Load SCAN tasks from file
	#
	# Input
	#  mytype : type of SCAN experiment
	#  split : 'train' or 'test'
	#
	# Output
	#  commands : list of input/output strings
	assert mytype in ['simple','addprim_jump','length','addprim_turn_left','all','template_around_right','viz','examine']
	assert split in ['train','test']
	fn = 'data/tasks_' + split + '_' + mytype + '.txt'
	fid = open(fn,'r')
	lines = fid.readlines()
	fid.close()
	lines = [l.strip() for l in lines]
	lines = [l.lstrip('IN: ') for l in lines]
	commands = [l.split(' OUT: ') for l in lines]
	#import pdb; pdb.set_trace()
	return commands

def sample_traditional_scan(nsupport,nquery,scan_tuples,
											upweight_privileged_words=True):
	# Randomly sample a set of SCAN episodes
	#
	# Input
	#  nsupport: number of support items
	#  nquery: number of query items
	#  scan_tuples : list of input/output tuples to draw from 
	return sample_traditional_scan_dist(nsupport, nquery, scan_tuples, upweight_privileged_words=upweight_privileged_words)


def sample_traditional_scan_dist(nsupport,nquery,scan_tuples, use_out=False, upweight_privileged_words=False):
	# Randomly sample a set of SCAN episodes
	#
	# Input
	#  nsupport: number of support items
	#  nquery: number of query items
	#  scan_tuples : list of input/output tuples to draw from 

	out_dist = {
		1: 8.25,
		2: 15.833333333333334,
		3: 17.416666666666668,
		4: 9.083333333333334,
		5: 3.3333333333333335,
		6: 7.416666666666667,
		7: 1.6666666666666667,
		8: 14.583333333333334,
		9: 4.0,
		10: 3.1666666666666665,
		11: 1.6666666666666667,
		12: 3.5,
		13: 0.75,
		14: 0.5,
		15: 0.5,
		16: 2.4166666666666665,
		17: 0.9166666666666666,
		18: 1.0,
		19: 0.4166666666666667,
		20: 0.4166666666666667,
		21: 0.08333333333333333,
		22: 0.3333333333333333,
		23: 0.08333333333333333,
		24: 1.8333333333333333,
		25: 0.08333333333333333,
		26: 0.08333333333333333,
		28: 0.5,
		29: 0.16666666666666666,
		}

	out_dist = {
		1: 8.25,
		2: 15.833333333333334,
		3: 17.416666666666668,
		4: 9.083333333333334,
		5: 3.3333333333333335,
		6: 7.416666666666667,
		7: 1.6666666666666667,
		8: 14.583333333333334,
		9: 4.0,
		10: 3.1666666666666665,
		11: 1.6666666666666667,
		12: 3.5,
		13: 0.75,
		14: 0.5,
		15: 0.5,
		16: 2.4166666666666665,
		17: 0.9166666666666666,
		18: 1.0,
		19: 0.4166666666666667,
		20: 0.4166666666666667,
		21: 0.08333333333333333,
		22: 3.0833333333333335
		}


	in_dist = {
		1 : 5.416666666666667,
		2 : 14.416666666666666,
		3 : 33.833333333333336,
		4 : 12.083333333333334,
		5 : 5.833333333333333,
		6 : 8.25,
		7 : 9.0,
		8 : 3.0,
		9 : 1.9166666666666667,
		10 : 1.6666666666666667,
		11 : 1.75,
		12 : 0.3333333333333333,
		13 : 0.5833333333333334,
		14 : 0.5,
		15 : 0.4166666666666667,
		16 : 0.5,
		17 : 0.08333333333333333,
		18 : 0.16666666666666666,
		19 : 0.08333333333333333,
		22 : 0.08333333333333333,
		23 : 0.08333333333333333,}
	#make compatible
	in_dist = {
		1 : 5.416666666666667,
		2 : 14.416666666666666,
		3 : 33.833333333333336,
		4 : 12.083333333333334,
		5 : 5.833333333333333,
		6 : 8.25,
		7 : 9.0,
		8 : 3.0,
		9 : 8.166666666666668,}


	didactic_in_dist = {
		}

	dist = in_dist
	if use_out: dist=out_dist

	#step 1: sample nums from dist
	keys, weights = list(zip(*dist.items()))

	lengths = random.choices(keys, weights=weights, k=nsupport)
	length_dict = Counter(lengths)

	if upweight_privileged_words:
		print("UPWEIGHTING PRIVILEGED WORDS")
		privileged_words = ['opposite', 'around']

	D_support = set()
	for length, count in length_dict.items():
	#for i in range(nsupport):
		if use_out:
			ex_of_len = [dat for dat in scan_tuples if len(dat[1].split(' ')) == length]
		else:
			ex_of_len = [dat for dat in scan_tuples if len(dat[0].split(' ')) == length]


		if upweight_privileged_words:

			ex_of_len_privileged = []
			for dat in ex_of_len:
				if any(word in dat[0] for word in privileged_words):
					ex_of_len_privileged.append(dat)
			#ex_of_len_privileged = [dat for dat in ex_of_len if any(word in dat[0] for word in privileged_words)]
			
			#import pdb; pdb.set_trace()
			ex_of_len.extend(ex_of_len_privileged*10)

		if len(ex_of_len) < count:
			print(f"number of exs={len(ex_of_len)}, count={count}, len={length}")
			data_samples = ex_of_len
	
		else:
			data_samples = random.sample(ex_of_len, k=count)
		#convert to set so it is okay
		data_samples = {tuple(d) for d in data_samples}
		#print("data_samples", data_samples)
		D_support = D_support.union( data_samples )

	D_query = []
	for i in range(nquery):
		dat = random.choice(scan_tuples)
		D_query.append(dat)
	return list(D_support),D_query


if __name__ == "__main__":
	pass 
