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

        if lambd > 0:
            self.tp = ceil((1 / lambd) * np.log((beta * mu) / (beta * mu - 1)))
        else:
            self.tp = sys.maxsize


    def _addUsers(self, X, y=None,y_old=None,estimacao_tempo=[],novos_users = [],estimacao_tempo_novosUsers = [], sample_weight=None):
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
                print("sample: ",sample)
                print("weight: ",weight)
                self._partial_fit(sample,estimacaoGanhoCanal[indx], weight)
                indx = indx+1
               
                
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


            ### add novos usuários
            y=[]
            for users in enumerate(novos_users):
                print("novos_users: ",novos_users.to_numpy(dtype='float32')[0])
                self.newUsers.append(users)
            
            for i,users in enumerate(self.newUsers):
                
                nova_amostra = users
                new_sample_weight = np.ones(1, dtype=np.float32, order='C')[0]
                
                print("nova_amostra: ",nova_amostra)
                print("new_sample_weight: ",new_sample_weight)
                self._partial_fit(nova_amostra,estimacao_tempo_novosUsers[i], new_sample_weight)

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

            
            return y,y_old
            
    

    def _get_nearest_micro_cluster(self, sample, micro_clusters,flag=0):
        smallest_distance = sys.float_info.max
        nearest_micro_cluster = None
        nearest_micro_cluster_index = -1
        for i, micro_cluster in enumerate(micro_clusters):     
            current_distance = np.linalg.norm(micro_cluster.center()-sample)
            if flag==1:
                print("sample", sample)
                print("centro: ", micro_cluster.center())
                print("current_distance: ", current_distance)
            if current_distance < smallest_distance:
                smallest_distance = current_distance
                nearest_micro_cluster = micro_cluster
                nearest_micro_cluster_index = i
        if flag==1:
            print("microCluster_escolhido: ", nearest_micro_cluster.center())
            print("#############################################")
        return nearest_micro_cluster_index, nearest_micro_cluster

    def _try_merge(self, sample,estimacaoGanhoCanal, weight, micro_cluster):
        if micro_cluster is not None:
            micro_cluster_copy = copy(micro_cluster)
            micro_cluster_copy.insert_sample(sample,estimacaoGanhoCanal, weight)
            if micro_cluster_copy.radius() <= self.eps:
                print("microCluster_maisPróximo: ", micro_cluster.center())
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

    def _partial_fit(self, sample,estimacaoGanhoCanal, weight):

        self._merging(sample, estimacaoGanhoCanal, weight)
        
        if self.t % self.tp == 0:  
            
            for p_micro_cluster in self.p_micro_clusters:
                gainList = p_micro_cluster.getGainChannel()
                
                ganhoTempoList = p_micro_cluster.getGanhoTempo()
                print("abs(gainList[idx])",gainList)

                for idx in range(len(gainList)):
                    if (abs(gainList[idx]) - abs(ganhoTempoList[idx][self.t]))> self.zeta:
                        p_micro_cluster.delete_sample(sample,idx,weight)
                        self.newUsers.append(sample)


            Xis = [((self._decay_function(self.t - o_micro_cluster.creation_time
                                          + self.tp) - 1) /
                    (self._decay_function(self.tp) - 1)) for o_micro_cluster in
                   self.o_micro_clusters]
            
            self.o_micro_clusters = [o_micro_cluster for Xi, o_micro_cluster in
                                     zip(Xis, self.o_micro_clusters) if
                                     o_micro_cluster.weight() >= Xi]

            
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


    
# data = np.random.random([1000, 5]) * 500
# clusterer = DenStream(lambd=0.1, eps=100, beta=0.5, mu=3)
# #for row in data[:100]:

# #    print(f"Number of p_micro_clusters is {len(clusterer.p_micro_clusters)}")
# #    print(f"Number of o_micro_clusters is {len(clusterer.o_micro_clusters)}")
