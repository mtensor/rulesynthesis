#agent.py
#imports
from interpret_grammar import Grammar, Rule

class ParseError(Exception):
    pass


class Example:
    def __init__(self, current, target):
        self.current = tuple(current)
        self.target = tuple(target)
        self.modified_current = tuple()

    def __hash__(self):
        return hash((self.current, self.target))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __str__(self):
        return str((self.current, self.target))

    def __repr__(self):
        return self.__str__()

class State:

    @staticmethod
    def new(examples):
        rules = []
        return State(examples, rules)

    def __init__(self, examples, rules):
        #perhaps check that they are in the right family or something #TODO
        self.examples = set(examples)
        self.rules = rules
        #self.g = build_g_from_rule_list


# class Model:
#     def __init__(self):
#         pass
#     def __call__(self, state, action):
#         pass #return new_state


def parse_rules(rules, input_symbols=None ):
    assert input_symbols
    Rules = []
    for rule in rules:
        #split into two on arrow
        if '->' in rule:
            idx = rule.index('->')
        else:
            raise ParseError
        lhs = rule[:idx]
        rhs = rule[idx+1:]

        lhs = ' '.join(lhs)
        rhs = ' '.join(rhs)
        Rules.append(Rule(lhs,rhs))

    #create grammar from Rules:
    #list_prims = ['dax', 'lug', 'fep', 'blicket', 'kiki', 'tufa', 'gazzer', 'zup', 'wif']#input_lang.symbols #TODO this is a major hack
    return Grammar(Rules, input_symbols)


def grammar_to_rule_list(g):
    rules = []
    for r in g.rules:
        rules.append( str(r).split(' ') )
    return rules


def remove_prefix(cur, tgt):
    cur = tuple(cur)
    tgt = tuple(tgt)
    if cur and tgt and tgt[0] == cur[0]:
        return remove_prefix(cur[1:], tgt[1:])
    else:
        return cur, tgt

def remove_suffix(cur, tgt):
    cur = tuple(cur)
    tgt = tuple(tgt)
    if cur and tgt and tgt[-1] == cur[-1]:
        return remove_suffix(cur[:-1], tgt[:-1])
    else:
        return cur, tgt

def show(state):
    print("current rules:")
    for r in state.rules: print(r)
    print()
    print("current examples:")
    for ex in state.examples:
        print(ex.current, ex.target)
    print()
    print()

if __name__ == '__main__':
    pass