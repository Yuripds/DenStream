import numpy as np


class MicroCluster:
    def __init__(self, lambd, creation_time):
        self.lambd = lambd
        self.decay_factor = 2 ** (-lambd)
        self.mean = 0
        self.variance = 0
        self.sum_of_weights = 0
        self.creation_time = creation_time
        self.gainChannel = []
        self.ganhoTempo = []

    def insert_sample(self, sample,estimacaoGanhoCanal, weight):
        print("teeeeeeeeeeeeeeeeeeeeeeeeeste33333333333333333333333333333333333:",estimacaoGanhoCanal)
        if self.sum_of_weights != 0:
            # Update sum of weights
            old_sum_of_weights = self.sum_of_weights
            new_sum_of_weights = old_sum_of_weights * self.decay_factor + weight

            # Update mean
            old_mean = self.mean
            new_mean = old_mean + (weight / new_sum_of_weights) * (sample - old_mean)
            
            # Update variance
            old_variance = self.variance
            new_variance = old_variance * ((new_sum_of_weights - weight)
                                           / old_sum_of_weights) + weight * (sample - new_mean) * (sample - old_mean)

            # Update Ganho de canal
            gc_aux = self.gainChannel
            gc_aux.append(sample)
            self.gainChannel = gc_aux

            # Update Ganho no tempo

            gt_aux = self.ganhoTempo
            gt_aux.append(estimacaoGanhoCanal)
            self.ganhoTempo= gt_aux

            self.mean = new_mean
            self.variance = new_variance
            self.sum_of_weights = new_sum_of_weights
        else:
            
            self.mean = sample
            self.sum_of_weights = weight


    def delete_sample(self,sample,index,weight):
        print("Essa função foi chamada")
        # Update sum of weights
        old_sum_of_weights = self.sum_of_weights
        new_sum_of_weights = old_sum_of_weights * self.decay_factor - weight

        # Update mean
        old_mean = self.mean
        new_mean = old_mean - (weight / new_sum_of_weights) * (sample - old_mean)
           
        # Update variance
        old_variance = self.variance
        new_variance = old_variance * ((new_sum_of_weights - weight)
                                           / old_sum_of_weights) - weight * (sample - new_mean) * (sample - old_mean)

        # Update Ganho de canal
        self.gainChannel.pop(index)

        # Update Ganho no tempo
        self.ganhoTempo.pop(index)

        self.mean = new_mean
        self.variance = new_variance
        self.sum_of_weights = new_sum_of_weights
       
        

    def radius(self):
        if self.sum_of_weights > 0:
            return np.linalg.norm(np.sqrt(self.variance / self.sum_of_weights))
        else:
            return float('nan')

    def center(self):
        return self.mean

    def weight(self):
        return self.sum_of_weights

    def getGainChannel(self):
        return self.gainChannel
    
    def getGanhoTempo(self):
        return self.ganhoTempo

    def __copy__(self):
        new_micro_cluster = MicroCluster(self.lambd, self.creation_time)
        new_micro_cluster.sum_of_weights = self.sum_of_weights
        new_micro_cluster.variance = self.variance
        new_micro_cluster.mean = self.mean
        return new_micro_cluster

