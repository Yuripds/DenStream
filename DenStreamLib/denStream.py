import sys
import numpy as np
from sklearn.utils import check_array
from copy import copy
from DenStreamLib import microCluster
from math import ceil
from sklearn.cluster import DBSCAN


class DenStream:

    def __init__(self, lambd=1, eps=1, beta=2, mu=2,eps_dbscan=0.3,min_samples_dbscan=3,zeta=1.0):
        """
        DenStream - Density-Based Clustering over an Evolving Data Stream with
        Noise.

        Parameters
        ----------
        lambd: float, optional
            The forgetting factor. The higher the value of lambda, the lower
            importance of the historical data compared to more recent data.
        eps : float, optional
            The maximum distance between two samples for them to be considered
            as in the same neighborhood.

        Attributes
        ----------
        labels_ : array, shape = [n_samples]
            Cluster labels for each point in the dataset given to fit().
            Noisy samples are given the label -1.

        Notes
        -----


        References
        ----------
        Feng Cao, Martin Estert, Weining Qian, and Aoying Zhou. Density-Based
        Clustering over an Evolving Data Stream with Noise.
        """
        self.lambd = lambd
        self.eps = eps
        self.eps_dbscan = eps_dbscan
        self.min_samples_dbscan = min_samples_dbscan
        self.beta = beta
        self.mu = mu
        self.t = 0
        self.p_micro_clusters = []
        self.o_micro_clusters = []
        self.zeta = zeta
        self.newUsers = []
        self.estimacao_tempo_newUsers = []

        if lambd > 0:
            self.tp = ceil((1 / lambd) * np.log((beta * mu) / (beta * mu - 1)))
        else:
            self.tp = sys.maxsize


    def _addUsers(self, X, y=None,y_old=None,estimacao_tempo=[],novos_users = [],estimacao_tempo_novosUsers = [], sample_weight=None, ad_users=False,time_param=500):
            """
            Parameter
            ----------
            X : {array-like, sparse matrix}, shape (n_samples, n_features)
                Subset of training data

            """

            X = check_array(X, dtype=np.float64, order="C")
           
            n_samples, _ = X.shape

            sample_weight = self._validate_sample_weight(sample_weight, n_samples)

            # if not hasattr(self, "potential_micro_clusters"):

            # if n_features != :
            # raise ValueError("Number of features %d does not match previous "
            # "data %d." % (n_features, self.coef_.shape[-1]))
            estimacaoGanhoCanal = estimacao_tempo
            
            indx=0
            for sample, weight in zip(X, sample_weight):
                self._partial_fit(sample,estimacaoGanhoCanal[indx], weight)
                indx = indx+1
               
                
            ######## chamar função de manutenção aqui com flag dizendo que não vai mais add usuarios e colocar tudo isso de baixo na função
            y_old_tempo = []
            contador = 0
            while contador<time_param:
                if self.t % 10 == 0:  
                    self.manutencao()

                    p_micro_cluster_centers = np.array([p_micro_cluster.center() for
                                                            p_micro_cluster in
                                                            self.p_micro_clusters])



                    p_micro_cluster_weights = [p_micro_cluster.weight() for p_micro_cluster in
                                                self.p_micro_clusters]


                    dbscan = DBSCAN(eps=self.eps_dbscan, min_samples=self.min_samples_dbscan , algorithm='brute')
                    dbscan.fit(p_micro_cluster_centers,
                                sample_weight=p_micro_cluster_weights)

                    y_old = []
                    for sample in X:
                        index, _ = self._get_nearest_micro_cluster(sample,
                                                                    self.p_micro_clusters)  
                        y_old.append(dbscan.labels_[index])

                    y_old_tempo.append(y_old)
                contador = contador+1
                self.t += 1            


############################ add flag para habilitar esta etapa ##############################################################
            ### add novos usuários
            if ad_users==True:
            
                y=[]
                user_nlist = novos_users.to_numpy(dtype='float32')

                for i,users in enumerate(user_nlist):
                    self.newUsers.append(users)
                    self.estimacao_tempo_newUsers.append(estimacao_tempo_novosUsers[i])
                
                for i,users in enumerate(self.newUsers):
                    #### add estimacao_tempo_novosUsers junto a fila de novos usuários
                    nova_amostra = users
                    new_sample_weight = np.ones(1, dtype=np.float32, order='C')[0]
                    
                    self._partial_fit(nova_amostra,self.estimacao_tempo_newUsers[i], new_sample_weight)

                    p_micro_cluster_centers = np.array([p_micro_cluster.center() for
                                                        p_micro_cluster in
                                                        self.p_micro_clusters])

                    p_micro_cluster_weights = [p_micro_cluster.weight() for p_micro_cluster in
                                                self.p_micro_clusters]

                    dbscan = DBSCAN(eps=self.eps_dbscan, min_samples=self.min_samples_dbscan , algorithm='brute')
                    dbscan.fit(p_micro_cluster_centers,
                                sample_weight=p_micro_cluster_weights)
                    
                    index, _ = self._get_nearest_micro_cluster(nova_amostra,self.p_micro_clusters)

                    y.append(dbscan.labels_[index])

                return  y_old_tempo , y

            else:
                return y_old_tempo


            
    

    def _get_nearest_micro_cluster(self, sample, micro_clusters):
        smallest_distance = sys.float_info.max
        nearest_micro_cluster = None
        nearest_micro_cluster_index = -1
        for i, micro_cluster in enumerate(micro_clusters):
            current_distance = np.linalg.norm(micro_cluster.center() - sample )
            if current_distance < smallest_distance:
                smallest_distance = current_distance
                nearest_micro_cluster = micro_cluster
                nearest_micro_cluster_index = i

        return nearest_micro_cluster_index, nearest_micro_cluster

    def _try_merge(self, sample,estimacaoGanhoCanal, weight, micro_cluster):
        if micro_cluster is not None:
            micro_cluster_copy = copy(micro_cluster)
            micro_cluster_copy.insert_sample(sample,estimacaoGanhoCanal, weight)
            if micro_cluster_copy.radius() <= self.eps:
                micro_cluster.insert_sample(sample, estimacaoGanhoCanal,weight)
                return True
        return False

    def _merging(self, sample,estimacaoGanhoCanal, weight):
        # Try to merge the sample with its nearest p_micro_cluster
        _, nearest_p_micro_cluster = \
            self._get_nearest_micro_cluster(sample, self.p_micro_clusters)
        success = self._try_merge(sample,estimacaoGanhoCanal, weight, nearest_p_micro_cluster)
        if success==True:
            print("sample:",sample)
            print("centro_mc:",nearest_p_micro_cluster.center())
            print("radius_mc:",nearest_p_micro_cluster.radius())

            for mc in self.p_micro_clusters:
                print("centros_deTodos_pmc: ",mc.center())

        if not success:
            # Try to merge the sample into its nearest o_micro_cluster
            index, nearest_o_micro_cluster = \
                self._get_nearest_micro_cluster(sample, self.o_micro_clusters)
            success = self._try_merge(sample,estimacaoGanhoCanal, weight, nearest_o_micro_cluster)
            
            if success:
                print("sample_out:",sample)

                for omc in self.o_micro_clusters:
                    print("centros_deTodos_omc: ",omc.center())

                if nearest_o_micro_cluster.weight() > self.beta * self.mu:
                    print("virou pmc: ",nearest_o_micro_cluster.center())
                    del self.o_micro_clusters[index]
                    self.p_micro_clusters.append(nearest_o_micro_cluster)
            else:
                # Create new o_micro_cluster
                micro_cluster = microCluster.MicroCluster(self.lambd, self.t)
                micro_cluster.insert_sample(sample,estimacaoGanhoCanal, weight)
                self.o_micro_clusters.append(micro_cluster)
                print("sample que criou omc: ", sample)
                print("criou omc: ", micro_cluster.center())

    def _decay_function(self, t):
        return 2 ** ((-self.lambd) * (t))



    def manutencao(self):
            for p_micro_cluster in self.p_micro_clusters:
                gainList = p_micro_cluster.getGainChannel()
                
                ganhoTempoList = p_micro_cluster.getGanhoTempo()

                sampleList = p_micro_cluster.getSample()
                
                #### mudar isso par aum while
                tam_init = len(gainList)
                idx=0
                while(tam_init>idx):
                    if (abs(abs(gainList[idx]) - abs(ganhoTempoList[idx][self.t])))> self.zeta:
                        self.newUsers.append(sampleList[idx])
                        self.estimacao_tempo_newUsers.append(ganhoTempoList[idx])
                        p_micro_cluster.delete_sample(idx)
                        tam_init = tam_init-1
                    idx = idx +1


            for o_micro_cluster in self.o_micro_clusters:
                gainList_outL = o_micro_cluster.getGainChannel()
                ganhoTempoList_out = o_micro_cluster.getGanhoTempo()
                sampleList = o_micro_cluster.getSample()
                tam_init = len(gainList_outL)
                idx=0
                while(tam_init>idx):
                    self.newUsers.append(sampleList[idx])
                    self.estimacao_tempo_newUsers.append(ganhoTempoList_out[idx])
                    o_micro_cluster.delete_sample(idx)
                    idx =idx +1
                    tam_init =tam_init-1




    def _partial_fit(self, sample,estimacaoGanhoCanal, weight):

        self._merging(sample, estimacaoGanhoCanal, weight)
        
        if self.t % self.tp == 0:  
            self.manutencao()

        self.t += 1

    def _validate_sample_weight(self, sample_weight, n_samples):
        """Set the sample weight array."""
        if sample_weight is None:
            # uniform sample weights
            sample_weight = np.ones(n_samples, dtype=np.float64, order='C')
        else:
            # user-provided array
            sample_weight = np.asarray(sample_weight, dtype=np.float64,
                                       order="C")
        if sample_weight.shape[0] != n_samples:
            raise ValueError("Shapes of X and sample_weight do not match.")
        return sample_weight

