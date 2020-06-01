#imports

from dreamcoder.task import EvaluationTimeout
import signal

import time
from type import Context
from scanPrimitives import tGrammar
from util import cuda_a_dict
from agent import parse_rules
import random

def compute_score(g, IO):
    score = 0.
    for xs, ys in zip(*IO):
        x = ' '.join(xs)
        y = ' '.join(ys)

        #import pdb; pdb.set_trace()
        if g.apply(x) == y:
            score += 1.    

    prop_correct = score/len(IO[0])

    return prop_correct



def test_deepcoder(sample, dc_model, model,
                   enum_only=False,
                   symbols=None,
                   timeout=30,
                   mdlIncrement=0.5,
                   examples=None,
                   query_examples=None,
                   cegis=False):

    #states, _ = model.sample_to_statelist(sample)
    #initial_state = states[0]
    #query_examples = {Example(cur, tgt) for cur, tgt in zip(sample['xq'], sample['yq']) if cur not in sample['xs'] }

    #import pdb; pdb.set_trace()
    if examples or query_examples:
        assert False, "not implemented"

    else:
        IO = (sample['xs'], sample['ys'])
        query_examples = [(cur, tgt) for cur, tgt in zip(sample['xq'], sample['yq']) if cur not in sample['xs'] ]
        test_IO = list(zip(*query_examples)) #check it out

        if cegis:
            xs = sample['xs']
            ys = sample['ys']
            cegis_IO = (xs[:cegis], ys[:cegis])


    states, rules = model.sample_to_statelist(sample)
    #for state, rule in zip(states, rules):
    assert len(rules) == 1
    running_sample = model.state_rule_to_sample(states[0], [] ) 
    running_sample = cuda_a_dict(running_sample)


    #states, _ = model.sample_to_statelist(sample)
    #initial_state = states[0]


    #setup

    if enum_only:
        g = dc_model.grammar
    else: g = dc_model.infer_grammar(running_sample)

    print(g)

    startTime = time.time()
    #i = 0
    lowerBound = 0
    upperBound = mdlIncrement

    stats = {
        'nodes_expanded': 0,
        'policy_runs': 0,
        'policy_gpu_runs': 0,
        'value_runs': 0,
        'value_gpu_runs': 0,
        'start_time': startTime
            }

    best = None
    bestScore = 0
    bestCegisScore = 0

    try:
        def timeoutCallBack(_1, _2): raise EvaluationTimeout()
        signal.signal(signal.SIGVTALRM, timeoutCallBack)
        signal.setitimer(signal.ITIMER_VIRTUAL, timeout)   

        while True:
            for ll, _, expr, _ in g.enumeration(Context.EMPTY,[], tGrammar, upperBound,
                                   lowerBound=lowerBound,
                                   maximumDepth=20, #TODO
                                   uniquePrims=True):
                #print("ll",ll, flush=True)  
                stats['nodes_expanded'] += 1
                intGrammar = expr.evaluate([])
                intGrammar = parse_rules(intGrammar, input_symbols=symbols) #TODO

                #import pdb; pdb.set_trace()

                if cegis:
                    cegisScore = compute_score(intGrammar, cegis_IO)
                    if cegisScore > 0:
                        score = compute_score(intGrammar, IO)
                    else: 
                        score = 0
                else:
                    score = compute_score(intGrammar, IO)

                #print(score)

                if score > bestScore:
                    bestScore = score
                    best = intGrammar 

                    if bestScore == 1.0: break

                if time.time() - startTime > timeout: break

            if time.time() - startTime > timeout or bestScore==1.0: break

            lowerBound += mdlIncrement
            upperBound += mdlIncrement

            print("lower bound", lowerBound)


    except EvaluationTimeout:
        print("Timed out while evaluating")
        #return False, NEGATIVEINFINITY
    finally:
        signal.signal(signal.SIGVTALRM, lambda *_: None)
        signal.setitimer(signal.ITIMER_VIRTUAL, 0)

        if best:
            stats['fraction_query_hit'] = compute_score(best, test_IO)
        else: 
            stats['fraction_query_hit'] = 0.

        solution = best
        hit = stats['fraction_query_hit'] == 1.0
        return hit, solution, stats
