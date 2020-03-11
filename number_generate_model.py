
import copy
import random
import math
from generate_episode import DataSamplingError
from number_word_interpret_grammar import NumberGrammar, Rule, IncompleteError

import pyro_num_distribution


def generate_lang_test_episode(nSupp,
								nQuery,
								input_lang,
								output_lang,
								prog_lang, 
								tabu_examples,
								lang=''):
	from test_langs import parseExamples
	max_try_novel = 10
	input_symbols = input_lang.symbols #i think this is fine
	count = 0
	while True:
		D, n_necessary = parseExamples(lang)
		#debug thingy
		#print("WARNING: JAPANESE ONLY LESS THAN ONE THOUSAND")
		#D = [d for d in D if d[1] < 10000]

		#random.shuffle(D)
		unseenRules = [ Rule(datum[0], str(datum[1])) for datum in D if len(datum[0].split()) == 1]# and datum[1] != 0 ] #TODO
		for r in unseenRules:
			r.valid_rule = True

		x_total = [d[0].split() for d in D]
		y_total = [[d[1]] for d in D]

		from util import make_hashable
		myhash = make_hashable('', sort=True)
		#ruleset is the set of rules I want it to predict
		#helped by tokenize_target_rule
		x_support = x_total[:n_necessary + nSupp]
		y_support = y_total[:n_necessary + nSupp]
		x_query = x_total
		y_query = y_total


		if myhash not in tabu_examples:
			break
		count += 1
		if count > max_try_novel:
			raise Exception("not able to generate a fitting grammar")
	#assert False, "grammar is not g, it's a subset here for during training"
		# during testing, want grammar you start with to have the other rules
	from util import build_sample
	return build_sample(x_support,
						y_support,
						x_query,
						y_query,
						input_lang,
						output_lang, 
						prog_lang, 
						myhash,
						grammar='',
						#seenRules=seenRules,
						unseenRules=unseenRules, integerIO=True)


#TODO
def generate_wordToNumber_episode(nSupp,
								nQuery,
								input_lang,
								output_lang,
								prog_lang, 
								tabu_examples,
								debug=False):
	max_try_novel = 10
	input_symbols = input_lang.symbols #i think this is fine
	count = 0
	while True:
		if debug:
			nSupp = 200
			from number_word_interpret_grammar import ChineseG, ChineseIntG
			g = ChineseG
			intG = ChineseIntG
		else:
			g, intG = pyro_num_distribution.generate_number_grammar(input_symbols)

		seenRules = [rule for rule in g.rules if len(rule.RHS_list)>1] #or rule.RHS_str =='0']
		unseenRules = [rule for rule in g.rules if len(rule.RHS_list) == 1] # and rule.RHS_str != '0'] #TODO
		
		try:
			D = sample_examples_gen(nSupp+nQuery, g, intG, maxIntSize=8)
		except DataSamplingError:
			continue

		x_total = [d[0].split() for d in D]
		y_total = [[d[1]] for d in D]

		from util import make_hashable
		myhash = make_hashable(g, sort=True)

		#ruleset is the set of rules I want it to predict
		#helped by tokenize_target_rule
		x_support = x_total[:nSupp]
		y_support = y_total[:nSupp]
		x_query = x_total
		y_query = y_total
		if myhash not in tabu_examples:
			break
		if debug: break
		count += 1
		if count > max_try_novel:
			raise Exception("not able to generate a fitting grammar")

	#assert False, "grammar is not g, it's a subset here for during training"
		# during testing, want grammar you start with to have the other rules
	from util import build_sample
	return build_sample(x_support,
						y_support,
						x_query,
						y_query,
						input_lang,
						output_lang, 
						prog_lang, 
						myhash,
						grammar=g,
						seenRules=seenRules,
						unseenRules=unseenRules, integerIO=True)

def sample_examples_gen(ndat, G, intG, maxIntSize=8):
	from test_langs import myDist #or whatever
	necessaryNums = list(range(1,21))
	for rule in G.rules:
		if len(rule.RHS_list)==1: #and rule.RHS_str != 0: # and len(rule.LHS_list)==1 and
			n = G.apply(rule.LHS_str)
			if n not in necessaryNums:
				necessaryNums.append(n)


	nums = necessaryNums
	while True:
		newNum = myDist(maxIntSize, training_dist=True)
		if newNum in nums or newNum == 0: continue 
		nums.append(newNum)
		if len(nums) == ndat: break

	dat = [(intG.evaluate(num), num) for num in nums]
	random.shuffle(dat)
	return dat

if __name__=='__main__':
	pass

