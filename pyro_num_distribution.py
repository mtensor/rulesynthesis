
import pyro
import pyro.poutine as poutine
import torch
import math
import copy

from number_word_interpret_grammar import Rule, NumberGrammar
from number_words import IntGrammar


def popFromList(lst, name=None, obs=None):	
	size = len(lst)
	idx = pyro.sample(name, pyro.distributions.Categorical(torch.tensor([1.]*size)), obs=torch.tensor(lst.index(obs)) if obs else None)
	popped = lst[idx]
	newLst = lst[:idx] + lst[idx+1:]
	return popped, newLst

def selectFromList(lst, name=None, obs=None):
	size=len(lst)	
	idx = pyro.sample(name, pyro.distributions.Categorical(torch.tensor([1.]*size)), obs=torch.tensor(lst.index(obs)) if obs else None)
	popped = lst[idx]
	return popped

def generateOneToTen(input_symbol_options):
	rules = []
	intRules = []
	for i in range(1, 10):
				num = str(i)
				word, input_symbol_options = popFromList(input_symbol_options, name=f"word_{i}", obs=None)
				if i == 1: oneWord = word
				rules.append(Rule(word, num))
				intRules.append([ str(num), '->', word])
	return rules, intRules, input_symbol_options, oneWord

def generate_number_grammar(input_symbols, grammar=None):
	zeroRule = False
	connectingWordsProb = 0.2
	ZeroProb = 0.2
	exceptionProb = 0.3
	tenThousandWordProb = 0.3

	input_symbol_options = copy.deepcopy(input_symbols)
	#random.shuffle(input_symbol_options)

	rules, intRules, input_symbol_options, oneWord = generateOneToTen(input_symbol_options)

	for base in [10000, 1000, 100, 10]:

		if base == 10 and tp == 'irregular':
			tp = 'irregular'
		else:
			if base in [100, 10]:
				tp = selectFromList(['regular', 'irregular'], f"regularity_{base}", obs=None)
			else: 
				tp = 'regular'


		# IMPORTANT: if 100s are irregular 10s cant be regular ... probably fine
		# TODO teens and twenties?
		# maybe teens are irregular if 10s are? 

		if tp == 'irregular':
			for i in range(1, 10):
				num = str(i*base)
				word, input_symbol_options = popFromList(input_symbol_options, name=f"irreg_word_{base}_{i}", obs=None)
				rules.append(Rule(word, num))
				intRules.append([ str(num), '->', word])
				#french situation??

			#do irregular teens half the time:
			if base == 10 and pyro.sample('irreg_teens', pyro.distributions.Bernoulli(0.5)):
				for i in range(11, 20):
					num = str(i)
					word, input_symbol_options = popFromList(input_symbol_options, name=f"teen_{i}", obs=None)
					rules.append(Rule(word, num))
					intRules.append([ str(num), '->', word])

			intRules.append(['>'+str(base), '->' , '[x - x%'+str(base)+']', '[x%'+str(base)+']' ] )

		elif tp == 'regular':

			#we have a word for ten thousand sometimes
			if base == 10000:
				if not pyro.sample(f'ten_thousand_word', pyro.distributions.Bernoulli(tenThousandWordProb), obs=None):
					base = 1000000

			baseWord, input_symbol_options = popFromList(input_symbol_options, name=f"word_{base}", obs=None)

			rule = Rule('x1 '+baseWord+' y1', '[x1]*'+str(base)+' [y1]')
			rules.append(rule)
			intRules.append(['>'+str(base), '->', '[x//'+str(base)+']', baseWord, '[x%'+str(base)+']'])


			if pyro.sample(f'one_exception_{base}', pyro.distributions.Bernoulli(exceptionProb), obs=None): #for now, only one exception
				oneException = True
				if pyro.sample(f'one_exception_change_{base}', pyro.distributions.Bernoulli(0.3), obs=None): #for now, only one exception
					oneExceptionWord, input_symbol_options = popFromList(input_symbol_options, name=f"oneWord_{base}", obs=None)
					exceptionRule = Rule(' '.join([oneExceptionWord, 'y1']), str(base)+'* 1' +' [y1]')
					intRule = ['//'+str(base)+'=='+str(1), '->', oneExceptionWord, '[x%'+str(base)+']']				
				else:
					exceptionRule = Rule(' '.join([baseWord, 'y1']), str(base)+'* 1' +' [y1]')
					intRule = ['//'+str(base)+'=='+str(1), '->', baseWord, '[x%'+str(base)+']']
				
				rules.insert(-1, exceptionRule)
				intRules.insert(-1, intRule)

				explicitExceptionRule = Rule(baseWord, str(base))
			else:
				explicitExceptionRule = Rule(' '.join([oneWord, baseWord]), str(base))
			rules.insert(0, explicitExceptionRule)


			if pyro.sample(f'exception_{base}', pyro.distributions.Bernoulli(exceptionProb), obs=None): #for now, only one exception
				exceptionNum = selectFromList(list(range(2, 10)), name=f"exception_num_{base}" , obs=None)
				exceptionWord, input_symbol_options = popFromList(input_symbol_options, name=f"exception_name_{base}", obs=None)
				exceptionRule = Rule(' '.join([exceptionWord, baseWord, 'y1']), str(base)+'* '+str(exceptionNum)+' [y1]')
				intRule = ['//'+str(base) +'=='+ str(exceptionNum), '->', exceptionWord, baseWord, '[x%'+str(base)+']']
				#TODO other direction

				#TODO
				rules.insert(-1, exceptionRule)
				intRules.insert(-1, intRule) 
		else: assert False

	if pyro.sample(f'connecting_word', pyro.distributions.Bernoulli(connectingWordsProb), obs=None):
		connectingWord, input_symbol_options = popFromList(input_symbol_options, name=f"connecting_word_val", obs=None)
		rules.append(Rule('u1 '+connectingWord+' x1', '[u1] [x1]') )
		intRules.append( ['>10', '->', '[x - x%10]' , connectingWord,  '[x%10]'] )
		#assert False, "need to figure out where to put connectingWord"

	if pyro.sample(f'zero', pyro.distributions.Bernoulli(ZeroProb), obs=None):
		zeroWord, input_symbol_options = popFromList(input_symbol_options, name=f"zero_word", obs=None)
		rules.append(Rule(zeroWord, '0') )
		intRules.append(['0', '->', zeroWord])
		zeroRule = True

	concatRule = Rule('u1 x1', '[u1] [x1]')
	rules.append(concatRule)

	#can shuffle those rules, but it doesn't matter, do that at example sampling time
	return NumberGrammar(rules, input_symbols), IntGrammar(intRules, zeroRule) #makeIntG(intRules, intG)


if __name__== '__main__':
	# rules = [1., 0., 1., 0., 1.]
	
	from util import get_episode_generator
	generate_episode_train, generate_episode_test, input_lang, output_lang, prog_lang = get_episode_generator("wordToNumber")
	#sample = generate_episode_train(set())

	#grammar = sample['grammar']
	#print(grammar)

	#trace = poutine.trace(model).get_trace(input_lang.symbols, output_lang.symbols, grammar)
	#print(trace.log_prob_sum())


	#trace = poutine.trace(generate_number_grammar).get_trace(input_lang.symbols, None)
	#print(trace.log_prob_sum())


	g, intG = generate_number_grammar(input_lang.symbols, None)

	from number_word_interpret_grammar import ChineseG, ChineseIntG
	#g = EnglishG
	#g = SpanishG
	g = ChineseG
	#g = simpChineseG
	intG = ChineseIntG

	#print(g)
	#for r in intG.tokenizedRules: print(r)

	x = [10, 100, 1000, 101, 203, 554, 23, 10, 11, 27, 4302, 1415, 2303, 414, 901]
	#print(g.apply('one hundred ling one'))
	for num in x:
		print(num,'   ', intG.evaluate(num))
		assert g.apply(intG.evaluate(num)) == num, f"fail on {x}"

