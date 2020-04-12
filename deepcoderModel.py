#deepcoderModel.py
import torch
import torch.nn as nn

from scanPrimitives import rulesToECProg, tGrammar
from util import cuda_a_dict

import sys
import os
#sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./ec'))
try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from dreamcoder.grammar import Grammar

def deepcoder_training_step(samples, dc_model):
    request = tGrammar
    loss = 0
    samples = [cuda_a_dict(sample) for sample in samples]
    for sample in samples:
        rec_input = [sample]
        rules = [r.split() for r in ' '.join(sample['grammar']).split('\n') ]
        program = rulesToECProg(rules) #TODO
        loss += dc_model.optimizer_step(rec_input, program, request)
    return loss/len(samples)

def dc_eval_ll(samples, dc_model):
    dc_model.eval()
    request = tGrammar
    loss = 0
    samples = [cuda_a_dict(sample) for sample in samples]
    for sample in samples:
        rec_input = [sample]
        #print(sample['grammar'])
        rules = [r.split() for r in ' '.join(sample['grammar']).split('\n') ]
        program = rulesToECProg(rules) #TODO
        loss += dc_model.loss(rec_input, program, request).data.item()
    dc_model.train()
    return loss/len(samples)
    

class DeepcoderRecognitionModel(nn.Module):
    def __init__(
            self,
            featureExtractor,
            grammar,  
            hidden=[256, 256, 256], #or whatever it is
            inputDimensionality=256): #TODO implement this
        super(DeepcoderRecognitionModel, self).__init__()
        self.grammar = grammar
        self.use_cuda = torch.cuda.is_available()

        self.featureExtractor = featureExtractor
        # Sanity check - make sure that all of the parameters of the
        # feature extractor were added to our parameters as well
        if hasattr(featureExtractor, 'parameters'):
            for parameter in featureExtractor.parameters():
                assert any(
                    myParameter is parameter for myParameter in self.parameters())

        hiddenLayers = []
        #inputDimensionality = featureExtractor.outputDimensionality
        for h in hidden:
            layer = nn.Linear(inputDimensionality, h)
            hiddenLayers.append(layer)
            hiddenLayers.append(nn.ReLU())
            inputDimensionality = h
        self.hiddenLayers = nn.Sequential(*hiddenLayers)

        self.logVariable = nn.Linear(inputDimensionality, 1)
        self.logProductions = nn.Linear(inputDimensionality, len(self.grammar))
        # if cuda:
        #     self.logVariable = self.logVariable.cuda()
        #     self.logProductions = self.logProductions.cuda()

        #add optimizer
        self.opt = torch.optim.Adam(
            self.parameters(), lr=0.0001, eps=1e-3, amsgrad=True)

        self.num_pretrain_episodes = 0

        if self.use_cuda:
            self.cuda()

    def forward(self, features):
        h = self.hiddenLayers(features)
        # added the squeeze
        return self.logVariable(h), self.logProductions(h)

    def _run(self, rec_input):
        # TODO, may want to do this 
        encoder_embedding, _, _ = self.featureExtractor(rec_input)
        return self(encoder_embedding)

    def loss(self, rec_input, program, request):
        g = self.infer_grammar(rec_input)
        try:
            return - g.logLikelihood(request, program)
        except:
            rules = [r.split() for r in ' '.join(rec_input[0]['grammar']).split('\n') ]
            print(rules)
            print(program)
            import pdb; pdb.set_trace()

    def optimizer_step(self, rec_input, program, request): 
        self.opt.zero_grad()
        loss = self.loss(rec_input, program, request) #can throw in a .mean() here when you are batching
        loss.backward()
        self.opt.step()
        return loss.data.item()

    def infer_grammar(self, rec_input):
        variables, productions = self._run(rec_input)
        #import pdb; pdb.set_trace()
        g = Grammar(
            variables, [
                (productions.squeeze(0)[k], t, prog) for k, (_, t, prog) in enumerate(
                    self.grammar.productions)])
        return g