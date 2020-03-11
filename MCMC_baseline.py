#MCMC baseline.py

from pyprob_distribution import FullModel, genModel, compute_score
from util import get_episode_generator
import math
import argparse
import os
import dill
import pyprob
import time


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--timeout', type=int, default=30)
    parser.add_argument('--num_traces', type=int, default=1500)
    parser.add_argument('--savefile', type=str, default="results/smcREPL.p")
    parser.add_argument('--mode', type=str, default="MCMC")
    parser.add_argument('--load_data', type=str, default='')
    args = parser.parse_args()

    _, _, input_lang, output_lang, prog_lang = get_episode_generator("scan_simple_original")
    

    if args.load_data:
        if os.path.isfile(args.load_data):
            print('loading test data ... ')
            with open(args.load_data, 'rb') as h:
                test_samples = dill.load(h)

        else: assert False





    frac_exs_hits = []
    for i, sample in enumerate(test_samples):

        IO = (sample['xs'], sample['ys'])
        model = FullModel(IO, input_lang.symbols, output_lang.symbols)

        if args.mode == 'MCMC':
            posterior = model.posterior_results(
                             num_traces=args.num_traces, # the number of samples estimating the posterior
                             inference_engine=pyprob.InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS, # specify which inference engine to use
                             observe={'dist': 0.} # assign values to the observed values
                             )
            best, _ = posterior.mode 
        
        elif args.mode == 'sample':
            best_dist = float('inf')
            best_g = None
            start_time = time.time()

            n_samp = 0
            while time.time() - start_time <= args.timeout:
                num_samples = 5
                gs = model.prior_results(num_samples).get_values()
                #import pdb; pdb.set_trace()
                for g, distance in gs:

                    if distance < best_dist:
                        best_dist = distance
                        best_g = g
                    n_samp += 1

            best = best_g 
            print("sampled", n_samp, "grammars")
        

        #n_ex = []
        frac_ex = 0.
        for x_list, y_list in zip(sample['xq'], sample['yq']):

            if best.apply(' '.join(x_list) ) == ' '.join(y_list):
                frac_ex += 1.

        print(best)

        frac_ex = frac_ex/len(sample['xq'])
        frac_exs_hits.append(frac_ex)

        print(f"fraction hits: {frac_ex}")

    avg = sum(frac_exs_hits)/len(frac_exs_hits)
    print(f"AVERAGE {avg*100} % examples")

    variance = sum([(f - avg)**2 for f in frac_exs_hits ])/len(frac_exs_hits)
    print(f"standard error: {math.sqrt(variance)/math.sqrt(len(frac_exs_hits))*100}")