import sys
import numpy as np
from sklearn.utils import check_array
from copy import copy
from DenStreamLib import microCluster
from math import ceil
from sklearn.cluster import DBSCAN


class DenStream:

    def __init__(self, lambd=1, eps=1, beta=2, mu=2,eps_dbscan=0.3,min_samples_dbscan=3):
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
        if lambd > 0:
            self.tp = ceil((1 / lambd) * np.log((beta * mu) / (beta * mu - 1)))
        else:
            self.tp = sys.maxsize

    def partial_fit(self, X, y=None, sample_weight=None):
        """
        Online learning.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Subset of training data

        y : Ignored

        sample_weight : array-like, shape (n_samples,), optional
            Weights applied to individual samples.
            If not provided, uniform weights are assumed.

        Returns
        -------
        self : returns an instance of self.
        """

        X = check_array(X, dtype=np.float64, order="C")

        n_samples, _ = X.shape

        sample_weight = self._validate_sample_weight(sample_weight, n_samples)

        # if not hasattr(self, "potential_micro_clusters"):

        # if n_features != :
        # raise ValueError("Number of features %d does not match previous "
        # "data %d." % (n_features, self.coef_.shape[-1]))

        for sample, weight in zip(X, sample_weight):
            self._partial_fit(sample, weight)
        return self

    def _addUsers(self, X, y=None,y_old=None,usuarioFinal=50,qtd_users_Add=10, sample_weight=None):
            """
            Parameter
            ----------
            X : {array-like, sparse matrix}, shape (n_samples, n_features)
                Subset of training data

            """
            X_part = X

            X=X_part[0:usuarioFinal]

            X = check_array(X, dtype=np.float64, order="C")

            n_samples, _ = X.shape

            sample_weight = self._validate_sample_weight(sample_weight, n_samples)

            # if not hasattr(self, "potential_micro_clusters"):

            # if n_features != :
            # raise ValueError("Number of features %d does not match previous "
            # "data %d." % (n_features, self.coef_.shape[-1]))

            for sample, weight in zip(X, sample_weight):
                self._partial_fit(sample, weight)
               
                
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
            usuarios_add = 0
            while (usuarios_add<qtd_users_Add):
                nova_amostra = X_part[0+usuarioFinal:usuarioFinal+1].to_numpy(dtype='float32')[0]
                new_sample_weight = np.ones(1, dtype=np.float32, order='C')[0]
                

                self._partial_fit(nova_amostra, new_sample_weight)

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

                usuarioFinal=usuarioFinal+1
                usuarios_add=usuarios_add+1


            
            return y,y_old
            
    def fit_predict(self, X, y=None, sample_weight=None):
        """
        Lorem ipsum dolor sit amet

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Subset of training data

        y : Ignored

        sample_weight : array-like, shape (n_samples,), optional
            Weights applied to individual samples.
            If not provided, uniform weights are assumed.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Cluster labels
        """

        X = check_array(X, dtype=np.float64, order="C")

        n_samples, _ = X.shape

        sample_weight = self._validate_sample_weight(sample_weight, n_samples)

        # if not hasattr(self, "potential_micro_clusters"):

        # if n_features != :
        # raise ValueError("Number of features %d does not match previous "
        # "data %d." % (n_features, self.coef_.shape[-1]))

        for sample, weight in zip(X, sample_weight):
            self._partial_fit(sample, weight)
        
        p_micro_cluster_centers = np.array([p_micro_cluster.center() for
                                            p_micro_cluster in
                                            self.p_micro_clusters])
        p_micro_cluster_weights = [p_micro_cluster.weight() for p_micro_cluster in
                                   self.p_micro_clusters]
        dbscan = DBSCAN(eps=self.eps_dbscan, min_samples=self.min_samples_dbscan , algorithm='brute')
        dbscan.fit(p_micro_cluster_centers,
                   sample_weight=p_micro_cluster_weights)

        y = []
        for sample in X:
            index, _ = self._get_nearest_micro_cluster(sample,
                                                       self.p_micro_clusters)
            y.append(dbscan.labels_[index])

        return y

    def _get_nearest_micro_cluster(self, sample, micro_clusters):
        smallest_distance = sys.float_info.max
        nearest_micro_cluster = None
        nearest_micro_cluster_index = -1
        for i, micro_cluster in enumerate(micro_clusters):
            current_distance = np.linalg.norm(micro_cluster.center() - sample)
            if current_distance < smallest_distance:
                smallest_distance = current_distance
                nearest_micro_cluster = micro_cluster
                nearest_micro_cluster_index = i
        return nearest_micro_cluster_index, nearest_micro_cluster

    def _try_merge(self, sample, weight, micro_cluster):
        if micro_cluster is not None:
            micro_cluster_copy = copy(micro_cluster)
            micro_cluster_copy.insert_sample(sample, weight)
            if micro_cluster_copy.radius() <= self.eps:
                micro_cluster.insert_sample(sample, weight)
                return True
        return False

    def _merging(self, sample, weight):
        # Try to merge the sample with its nearest p_micro_cluster
        _, nearest_p_micro_cluster = \
            self._get_nearest_micro_cluster(sample, self.p_micro_clusters)
        success = self._try_merge(sample, weight, nearest_p_micro_cluster)
        print("success: ", success)
        if not success:
            # Try to merge the sample into its nearest o_micro_cluster
            index, nearest_o_micro_cluster = \
                self._get_nearest_micro_cluster(sample, self.o_micro_clusters)
            success = self._try_merge(sample, weight, nearest_o_micro_cluster)
            print("success_out: ", success)
            if success:
                if nearest_o_micro_cluster.weight() > self.beta * self.mu:
                    del self.o_micro_clusters[index]
                    self.p_micro_clusters.append(nearest_o_micro_cluster)
            else:
                # Create new o_micro_cluster
                micro_cluster = microCluster.MicroCluster(self.lambd, self.t)
                micro_cluster.insert_sample(sample, weight)
                self.o_micro_clusters.append(micro_cluster)

    def _decay_function(self, t):
        return 2 ** ((-self.lambd) * (t))

    def _partial_fit(self, sample, weight):
        self._merging(sample, weight)
        
        if self.t % self.tp == 0:
            print("entrou")
            print("Tp: ",self.tp)
            print("t: ",self.t)

            self.p_micro_clusters = [p_micro_cluster for p_micro_cluster
                                     in self.p_micro_clusters if
                                     p_micro_cluster.weight() >= self.beta *
                                     self.mu]
            
            for p_micro_cluster in self.p_micro_clusters:
                print('raio_pmc: ',p_micro_cluster.radius()) 
                print('centros_pmc: ',p_micro_cluster.center()) 
                print("loop")

            Xis = [((self._decay_function(self.t - o_micro_cluster.creation_time
                                          + self.tp) - 1) /
                    (self._decay_function(self.tp) - 1)) for o_micro_cluster in
                   self.o_micro_clusters]
            print("Xis: ",Xis)
            self.o_micro_clusters = [o_micro_cluster for Xi, o_micro_cluster in
                                     zip(Xis, self.o_micro_clusters) if
                                     o_micro_cluster.weight() >= Xi]

            for  o_micro_cluster in self.o_micro_clusters:
                print('raio_omc: ',o_micro_cluster.radius()) 
                print('centros_omc: ',o_micro_cluster.center())
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
# #    clusterer.partial_fit([row], 1)
# #    print(f"Number of p_micro_clusters is {len(clusterer.p_micro_clusters)}")
# #    print(f"Number of o_micro_clusters is {len(clusterer.o_micro_clusters)}")
# y = clusterer.fit_predict(data[100:])
# print("finished!", max(y))