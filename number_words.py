#rewrite system


def intToList(num):
	return [c for c in str(num)]

def lambdaMaker(i):
	a = i
	return lambda x: x == a

def exceptionMaker(i,j):
	a = i
	b = j
	return lambda x: x//b == a

class IntGrammar:
	def __init__(self, tokenizedRules, zeroRule=False):
		self.LHS_DICT = {}
		for i in range(101):
			self.LHS_DICT[ str(i) ] = lambdaMaker(i) #TODO

		for i in range(1, 10):
			for j in [10, 100, 1000, 10000, 1000000]:
				self.LHS_DICT['//'+str(j)+'=='+str(i)] = exceptionMaker(i, j)

		self.LHS_DICT['>10'] = lambda x: x>= 10
		self.LHS_DICT['>100'] = lambda x: x>= 100
		self.LHS_DICT['>1000'] = lambda x: x>= 1000
		self.LHS_DICT['>10000'] = lambda x: x>= 10000
		self.LHS_DICT['>1000000'] = lambda x: x>= 1000000

		self.RHS_DICT = {}
		self.RHS_DICT['[x//1000000]'] = lambda x: self.evaluate(x//1000000)
		self.RHS_DICT['[x//10000]'] = lambda x: self.evaluate(x//10000)
		self.RHS_DICT['[x//1000]'] = lambda x: self.evaluate(x//1000)
		self.RHS_DICT['[x//100]'] = lambda x: self.evaluate(x//100)
		self.RHS_DICT['[x//10]'] = lambda x: self.evaluate(x//10)

		self.RHS_DICT['[x%1000000]'] = lambda x: self.evaluate(self._modfun(x,1000000))
		self.RHS_DICT['[x%10000]'] = lambda x: self.evaluate(self._modfun(x,10000))
		self.RHS_DICT['[x%1000]'] = lambda x: self.evaluate(self._modfun(x,1000))
		self.RHS_DICT['[x%100]'] = lambda x: self.evaluate(self._modfun(x,100))
		self.RHS_DICT['[x%10]'] = lambda x: self.evaluate(self._modfun(x, 10))

		self.RHS_DICT['[x - x%10]'] = lambda x: self.evaluate(x - x%10)
		self.RHS_DICT['[x - x%100]'] = lambda x: self.evaluate(x - x%100)
		self.RHS_DICT['[x - x%1000]'] = lambda x: self.evaluate(x - x%1000)
		self.RHS_DICT['[x - x%10000]'] = lambda x: self.evaluate(x - x%10000)
		self.RHS_DICT['[x - x%1000000]'] = lambda x: self.evaluate(x - x%1000000)

		#'[x - x%10]'
		self.rules = [self._detokenize(rule) for rule in tokenizedRules]
		self.tokenizedRules = tokenizedRules
		self.zeroRule = zeroRule

	def _modfun(self, x, base):
			ret = x%base
			if ret == 0:
				return ''
			if self.zeroRule:
				if ret < base/10:
					return " ".join([self.evaluate(0), self.evaluate(ret)])
			return ret
			

	def apply(self, x):
		x = self._digitListToInt(x)
		return self.evaluate(x)

	def evaluate(self, x):
		for lhs, rhs in self.rules:
	 		if type(x)!= str and lhs(x):
	 			return rhs(x)
		#print(x)
		return x

	def _digitListToInt(self, lst):
		return int("".join(lst))

	def _detokenize(self, tokenizedRule):
		splitIdx = tokenizedRule.index('->')
		lhsTokens, rhsTokens = tokenizedRule[:splitIdx], tokenizedRule[splitIdx+1:]

		#for now, very simple parsing ...
		assert len(lhsTokens) == 1
		lhs = self.LHS_DICT.get(lhsTokens[0], lambda x: int(lhsTokens[0])==x)

		rhs = lambda x: " ".join(" ".join([self.RHS_DICT.get(token, lambda y: token )(x) for token in rhsTokens]).split())
		return lhs, rhs

	def __getstate__(self):
		return (self.tokenizedRules, self.zeroRule)

	def __setstate__(self, state):
		if type(state)==dict:
			pass
		else:
			tokenizedRules, zeroRule = state
			self.__init__(tokenizedRules, zeroRule=zeroRule)

if __name__=='__main__':
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
		['11', '->', 'eleven'],
		['12', '->', 'twelve'],
		['13', '->', 'thirteen'],
		['14', '->', 'fourteen'],
		['15', '->', 'fifteen'],
		['12', '->', 'twelve'],
		['13', '->', 'thirteen'],
		['14', '->', 'fourteen'],
		['15', '->', 'fifteen'],
		['16', '->', 'sixteen'],
		['17', '->', 'seventeen'],
		['18', '->', 'eighteen'],
		['19', '->', 'nineteen'],
		#can give these too ..
		['20', '->', 'twenty'],
		['30', '->', 'thirty'],
		['40', '->', 'forty'],
		['50', '->', 'fifty'],
		['60', '->', 'sixty'],
		['70', '->', 'seventy'],
		['80', '->', 'eighty'],
		['90', '->', 'ninety'],
		['100','->', 'one hundred'],

		[ '>1000' , '->', '[x//1000]', 'thousand', '[x%1000]' ],
		[ '>100' , '->', '[x//100]', 'hundred', '[x%100]' ],
		#[ '%==0', '100' , '->', '[x//10]', 'ty' ],
		['>10', '->', '[x - x%10]', '[x%10]']
		#['>', '2', '0', '->', '[x - x%10]', '[x%10]'],
		#['>', '1', '5', '->', '[x%10]', 'teen']
	]

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
		#['10', '->', 'ten'],
		[ '//1000==2' , '->', '[x//1000]', 'thousand', '[x%1000]' ],
		[ '>1000' , '->', '[x//1000]', 'thousand', '[x%1000]' ],
		['//100==2', '->', 'liang', 'hundred', '[x%100]'],
		[ '>100' , '->', '[x//100]', 'hundred', '[x%100]' ],
		['>10', '->', '[x//10]', 'ten', '[x%10]'],
		['0', '->', 'ling']

			]


	g = IntGrammar(tokenizedRules, zeroRule=True)



	print(1, "goes to", g.apply(['1']))
	print(12, "goes to", g.apply(['1', '2']))
	print(123, "goes to", g.apply(['1', '2', '3']))
	print(11, "goes to", g.apply(['1', '1']))

	print(101, "goes to", g.apply(['1', '0', '1']))

	print("your turn!")
	while True:
		print("write an integer:")
		x = input()
		print(g.apply([c for c in x]))
# Need Grammar object

# Need rule object

# grammar.apply(i)


