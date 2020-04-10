#imports:
from program import Primitive, Program, prettyProgram
from grammar import Grammar
from type import tlist, tint, tbool, arrow, baseType #, t0, t1, t2
import math

def multiArrow(i_tp, o_tp, n):
    tpLst = [i_tp for _ in range(n)] + [o_tp]
    return arrow(*tpLst)

#semantics
def _list_constructor(n):

    def fn(i, lst):
        if i == 0:
            return lst
        else:
            return lambda x: fn(i-1, [x]+lst)
    return fn(n, [])

def _prim_rule(x):
    return lambda y: x + ['->'] + y

def _grammar_constructor(primLst): lambda HOLst: lambda lastRule: primLst + HOLst + [lastRule]

def _suffix_rule(lhsToken):
    def fs(token, var, n):
        in_v = var+'1'
        out_v = f"[{var}1]"
        rule = [in_v, token, '->'] + [out_v for _ in range(n)]
        return rule
    return lambda v: lambda i: fs(lhsToken, v, i) 

def _infix_rule(lhsToken):
    def fi(token, v1, v2, lst):
        in_v1 = f"{v1}1"
        in_v2 = f"{v2}2"
        out_vs = [f"[{in_v1}]", f"[{in_v2}]"]

        rule = [in_v1, token, in_v2, '->'] + [ out_vs[i] for i in lst]
        return rule
    return lambda v: lambda v2: lambda lst: fi(lhsToken, v1, v2, lst)

_concat_rule = ['u1', 'x1', '->', '[u1]', '[x1]']
_u_rule = ['u1, u2', '->'. '[u2]', '[u1]']

#types:
tRHSToken = baseType("RHSToken")
tLHSToken = baseType("LHSToken")

tPrimRule = baseType("PrimRule")
tHORule = baseType("HORule")
tLastRule = baseType("LastRule")

tPrimList = baseType("PrimList")
tHOList = baseType("HOList")

tGrammar = baseType("Grammar")
tVar = baseType("Var")
tPos = baseType("Pos")


#params
RHSTokens = []
LHSTokens = [""]
n_prims = range(3, 9)
n_ho_rules = range(3, 9)

#syntax:

rhs_token_prims = [Primitive(token, tRHSToken, token) for token in RHSTokens]
lhs_token_prims = [Primitive(token, tLHSToken, token) for token in LHSTokens]
prim_list_constructors = [Primitive(f"{i}_prims", 
    multiArrow(tPrimRule, tPrimList, i), _list_constructor(i)) for i in n_prims]
ho_list_constructors = [Primitive(f"{i}_ho_rules", 
    multiArrow(tHORule, tHOList, i), _list_constructor(i)) for i in n_ho_rules]
ints = [Primitive(str(i), tint, i) for i in range(1, 4)]

Primitives = [
    Primitive("grammar", arrow(tPrimList, tHOList, tLastRule, tGrammar), _grammar_constructor),
    #primitive rules:
    ] + prim_list_constructors + [
    Primitive("prim_rule", arrow(tLHSToken, tRHSToken, tPrimRule), _prim_rule ),
    #Higher order rules:
    ] + ho_list_constructors + [
    rhs_token_prims,
    lhs_token_prims,
    Primitive("suffix_rule", arrow(tLHSToken, tVar, tint,  tHORule), _suffix_rule),
    Primitive("infix_rule", arrow( tLHSToken, tVar, tVar, tlist(tPos), tHORule), _infix_rule),
    Primitive("x", tVar, "x"),
    Primitive("u", tVar, "u"),
    ] + ints + [
    Primitive("pos0", tPos, 0),
    Primitive("pos1", tPos, 1),
    Primitive("pos_append", arrow(tPos, tlist(tPos),  tlist(tPos)),  lambda x: lambda lst: [x] + lst ),
    Primitive("pos_start", arrow(tPos, tPos, tlist(tPos)) , lambda x: lambda y: [x, y] ),
    #last Rules:
    Primitive("concat_rule", tLastRule, _concat_rule),
    Primitive("u_rule", tLastRule, _u_rule),
    ]

def constructList(lst):
    s1, s2 = lst[0], lst[1]
    lst = lst[2:]

    e = buildFromArgs( Primitive.GLOBALS["pos_start"], 
        [Primitive.GLOBALS[f"pos{s1}"], Primitive.GLOBALS[f"pos{s2}"] ] 
        )
    
    for i in lst:
        e = buildFromArgs( Primitive.GLOBALS["pos_append"]
            [Primitive.GLOBALS[f"pos{i}"] , e])
    return e 

def buildFromArgs(fn, args):
    e = fn
    for arg in args:
        e = Application(e, arg)
    return e

def buildPrimRule(prim):
    lhs, _ , rhs = prim
    return buildFromArgs(Primitive.GLOBALS["prim_rule"],
        [Primitive.GLOBALS[lhs] , Primitive.GLOBALS[rhs]] )

def buildHORule(r):
    if r.index('->') == 2:
        #suffix rule
        var, token = r[0], r[1]
        var = var[0] #ignore the 1 or 2 at the end
        n = len(r) - 3
        return buildFromArgs(Primitive.GLOBALS["suffix_rule"], 
            [Primitive.GLOBALS[token], Primitive.GLOBALS[var], Primitive.GLOBALS[str(n)]] )

    elif r.index('->') == 3:
        #infix rule
        var1, token, var2 = r[:3]
        lst = r[4:]
        v1,v2 = var1[0], var2[0]

        d = {f"[{var2}]" : 0 , f"[{var2}]" : 1 }

        return buildFromArgs(Primitive.GLOBALS["infix_rule"].
            [Primitive.GLOBALS[], Primitive.GLOBALS[v1], Primitive.GLOBALS[v2] , ] 
            constructList( [d[l] for l in lst ] ))

    else: assert 0


def rulesToECProg(rules):

    g = Primitive.GLOBALS["grammar"]

    prims = [ rule for rule in rules if rule.index('->') == 1 ]
    ho_rules = rules[ len(prims):-1]
    last_rule = rules[-1]

    #primRules
    n = len(prims)
    primRules = buildFromArgs(Primitive.GLOBALS(f"{n}_prims"), [buildPrimRule(prim) for prim in prims] )
    
    #HO rules
    n = len(ho_rules)
    HORules = buildFromArgs(Primitive.GLOBALS(f"{n}_ho_rules"), [buildHORule(r) for r in ho_rules] )

    #last rule
    lastRule = buildLastRule(last_rule) #TODO

    expr = buildFromArgs(g, [primRules, HORules, lastRule] )
    return expr


if __name__ == "__main__":
    #g = Grammar.uniform(deepcoderPrimitives())
    # g = Grammar.fromProductions(deepcoderProductions(), logVariable=.9)
    # request = arrow(tlist(tint), tint, tint)
    # p = g.sample(request)
    # print("request:", request)
    # print("program:")
    # print(prettyProgram(p))
    # print("flattened_program:")
    # flat = flatten_program(p)
    # print(flat)

    f = _list_constructor(4)
    a = f(1)(2)(3)(4)
    print(a)
