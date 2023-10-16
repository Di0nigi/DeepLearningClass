import numpy as np
import perceptronV2 as p
import random as rand
import matplotlib.pyplot as Matplot
import os


class ForwardPassNN:
    
    def __init__(self, hlayers,distribution,learning_rates=0.01,epochs=10000):
        self.it=epochs
        self.ln=learning_rates
        self.hidden=hlayers
        self.neurons=[]
        self.out=p.Perceptron(learning_rates=self.ln, n_iters=self.it)
        self.NperL=distribution

        return
    def initP(self):
        for elem in range(self.hidden):
            layer=[]
            for i in range(self.NperL[elem]):
                pn=p.Perceptron(self.ln,self.it)
                layer.append(pn)
            self.neurons.append(layer)
        return
    def passForward(self,layer1,layer2):
        return
    def train(self):
        return
    