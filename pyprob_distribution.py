#import pyro
#import pyro.poutine as poutine
import pyprob
from pyprob import Model
import torch
import math

from interpret_grammar import Rule, Grammar

import random #AHHH
"""
nxrules = random.choice((3,4,5,6,7))
nurules = 0
nprims = random.choice(range(4,9))

	input_symbols = input_lang.symbols
	output_symbols = output_lang.symbols
"""

#helper funs:
p_lhs_onearg = 0.4 # probability that we have a single argument. Otherwise, two arguments
p_stop_rhs = 0.6 # prob. of stopping on the right hand side #0.4 for longtail
vars_input = ['u1','u2','x1','x2']
vars_output = ['['+v+']' for v in vars_input]
icon_concat_rule = Rule('u1 x1','[u1] [x1]')

def sample_bernoulli(p):
	return pyprob.sample( pyprob.distributions.Categorical(torch.tensor([1-p, p])))

def getNPrims(g):
	nPrims = 0
	for rule in g.rules:
		if len(rule.LHS_list) == 1:
			nPrims += 1
	return nPrims

def getNHOrules(g):
	nHOrules = 0
	for rule in g.rules:
		if len(rule.LHS_list) > 1:
			nHOrules += 1

	return nHOrules - 1

def popFromList(lst):	
	size = len(lst)
	idx = pyprob.sample(pyprob.distributions.Categorical(torch.tensor([1.]*size))).long().item()
	popped = lst[idx]
	newLst = lst[:idx] + lst[idx+1:]
	return popped, newLst

def selectFromList(lst):
	size = len(lst)
	idx = pyprob.sample(pyprob.distributions.Categorical(torch.tensor([1.]*size))).long().item()
	popped = lst[idx]
	return popped

def rangeSample(span):
	mini = span[0]
	maxi = span[1]
	size = maxi - mini + 1
	samp = pyprob.sample(pyprob.distributions.Categorical(torch.tensor([1.]*size)))
	n = samp + mini
	return n

def generate_prim(i, input_symbol_options, output_symbol_options, blank_prim=True, obs_prim=None):
	if obs_prim:
		LHS_obs = obs_prim.LHS_str
		RHS_obs = obs_prim.RHS_str
		is_blank = RHS_obs == ''


	input_symbol, input_symbol_options = popFromList(input_symbol_options)

	if blank_prim and sample_bernoulli(1/7.):
		rule = Rule(input_symbol, '') 
	else:
		output_symbol, output_symbol_options = popFromList(output_symbol_options)
		rule = Rule(input_symbol,output_symbol)

	return rule, input_symbol_options, output_symbol_options

def sample_LHS(i, func_name, p_lhs_onearg=0.4, obs_rule=None):
	if obs_rule:
		LHS_len = len(obs_rule.LHS_list)

	#if pyro.sample(f'one_arg_{i}', pyro.distributions.Bernoulli(p_lhs_onearg), obs=int(LHS_len==2) if obs_rule else None):
	if sample_bernoulli(p_lhs_onearg):
		var, _ = popFromList(vars_input)
		arr = [var, func_name]
	else:
		var1, rest_vars_input  = popFromList(vars_input)
		var2, _ = popFromList(rest_vars_input)
		arr = [var1, func_name, var2]

	used_vars = list(set(arr).intersection(set(vars_input)))
	used_vars = ['['+u+']' for u in used_vars] # variables we used
	return ' '.join(arr), used_vars

def multiShuffle(i, AdditionalLength, vars_in_lhs):
	while True:
		RHS = []
		for i in range(AdditionalLength + len(vars_in_lhs)):
			x = selectFromList(vars_in_lhs)
			RHS.append(x)

		if all( v in RHS for v in vars_in_lhs):
			break

	return RHS


def sample_RHS(i, vars_in_lhs, p_stop_rhs=0.6):
	# vars_in_lhs : variables that were used to construct the left hand side

	# pyro geometric
	#obs_len = torch.tensor(len(obs_rule.RHS_list) - len(vars_in_lhs)) if obs_rule else None

	AdditionalLength = 0

	while not sample_bernoulli(p_stop_rhs):
		AdditionalLength += 1

	#AdditionalLength = pyprob.sample(pyro.distributions.Geometric(torch.tensor(p_stop_rhs)))
	#arr = obs_rule.RHS_list
	arr = multiShuffle(i, AdditionalLength, vars_in_lhs)

	return ' '.join(arr)

def genModel(input_symbols, output_symbols):
	input_symbol_options = input_symbols
	output_symbol_options = output_symbols

	nPrims = rangeSample([4,9]).long().item()
	rules = []
	for i in range(nPrims):
		rule, input_symbol_options, output_symbol_options = generate_prim(i, input_symbol_options, 
																			output_symbol_options,
																			blank_prim=True)
		rules.append(rule)

	nxrules = rangeSample([3,7]).long().item()

	for i in range(nxrules):
		#sample_lhs makes sure not to use words twice for simple defs
		input_symbol, input_symbol_options = popFromList(input_symbol_options)

		LHS, used_vars_lhs = sample_LHS(i, input_symbol)
		RHS = sample_RHS(i, used_vars_lhs, p_stop_rhs=0.35)
		
		rules.append(Rule(LHS,RHS))
	
	last_rule = sample_bernoulli(0.5)
	if last_rule == 1.0: #TODO
		#scan:
		rules.append(Rule('u1 u2','[u2] [u1]'))
	else:
		rules.append(icon_concat_rule)

	return Grammar(rules, input_symbols) #todo

def compute_score(g, IO):
	score = 0
	for xs, ys in zip(*IO):
		x = ' '.join(xs)
		y = ' '.join(ys)

		if g.apply(x) == y:
			score += 1

	
	prop_correct = score/len(IO[0])
	#print("prop correct", prop_correct)

	dist = 1/prop_correct**4 if prop_correct > 0. else 10**16 #TODO change this??
	return dist


def make_uniform_under_dist(distance):
	return pyprob.distributions.Uniform(0., distance) #distributions

class FullModel(Model):
	def __init__(self, IO, input_symbols, output_symbols):
		super(FullModel, self).__init__(name='model') # give the model a name
		self.IO = IO
		self.input_symbols = input_symbols
		self.output_symbols = output_symbols

	def forward(self, output_distance=True):
		g = genModel(self.input_symbols, self.output_symbols)
		distance = compute_score(g, self.IO)
		dist_obj = make_uniform_under_dist(distance)
		pyprob.observe(dist_obj, name="dist")
		if output_distance:
			return g, distance
		else:
			return g

#for mcmc:

# z = 0.3
# trace = poutine.trace(model).get_trace(params, grammar)
# print(trace.log_prob_sum())

def dummySample(i):
	return pyprob.sample( pyprob.distributions.Categorical(torch.tensor([0.5,0.5])))

class SimpleModel(Model):
	def __init__(self):
		super(SimpleModel, self).__init__(name='Gaussian with unknown mean') # give the model a name
		self.distance = 5

	def forward(self):
		nPrims = rangeSample([4,9], "nPrims").long().item()
		outRules = []
		#print(nPrims)
		for i in range(nPrims):
			#print(len(rules))
			outRules.append(dummySample(i))

		dist_obj = make_uniform_under_dist(self.distance)
		pyprob.observe(dist_obj, name="dist")
		#last = pyprob.observe(pyprob.distributions.Categorical(torch.tensor([0.5,0.5])), name="dist")
		#outRules.append(last)
		
		return tuple(outRules)

if __name__== '__main__':
	from util import get_episode_generator
	generate_episode_train, generate_episode_test, input_lang, output_lang, prog_lang = get_episode_generator("scan_random")
	sample = generate_episode_train(set())
	grammar = sample['grammar']
	print(grammar)

	#TRUE
	# trueRules = [1., 0., 1., 0., 1.]

	# model = SimpleModel()

	IO = (sample['xs'], sample['ys'])
	model = FullModel(IO, input_lang.symbols, output_lang.symbols)

	for i in range(10):
		g, distance = model.forward(output_distance=True)
		print("iiii", i)


	posterior = model.posterior_results(
	                     num_traces=1500, # the number of samples estimating the posterior
	                     inference_engine=pyprob.InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS, # specify which inference engine to use
	                     observe={'dist': 0.} # assign values to the observed values
	                     )

	mode = posterior.mode
	print(mode)

