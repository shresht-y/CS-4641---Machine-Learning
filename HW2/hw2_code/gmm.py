import numpy as np
from tqdm import tqdm
from kmeans import KMeans


SIGMA_CONST = 1e-6
LOG_CONST = 1e-32

FULL_MATRIX = False # Set False if the covariance matrix is a diagonal matrix

class GMM(object):
    def __init__(self, X, K, max_iters=100):  # No need to change
        """
        Args:
            X: the observations/datapoints, N x D numpy array
            K: number of clusters/components
            max_iters: maximum number of iterations (used in EM implementation)
        """
        self.points = X
        self.max_iters = max_iters

        self.N = self.points.shape[0]  # number of observations
        self.D = self.points.shape[1]  # number of features
        self.K = K  # number of components/clusters

    # Helper function for you to implement
    def softmax(self, logit):  # [5pts]
        """
        Args:
            logit: N x D numpy array
        Return:
            prob: N x D numpy array. See the above function.
        Hint:
            Add keepdims=True in your np.sum() function to avoid broadcast error. 
        """
        
        #print(logit)
        #max_row = np.max(logit, axis=1, keepdims=True)
        #print(max_row)
        #print(logit - max_row)\
        max_values = np.max(logit, axis=1, keepdims=True)
        logit_scaled = logit - max_values
        #logit -= np.max(logit, axis=1, keepdims=True)
        logit_scaled = np.exp(logit_scaled)
        total_sum = np.sum(logit_scaled, axis=1)[:,None]
        #print(logit/total_sum)
        #print(np.sum(logit/total_sum, axis=1, keepdims=True))
        #print("max values", max_values)
        #print("LOGIT", logit+max_values)
        return logit_scaled/total_sum

    def logsumexp(self, logit):  # [5pts]
        """
        Args:
            logit: N x D numpy array
        Return:
            s: N x 1 array where s[i,0] = logsumexp(logit[i,:]). See the above function
        Hint:
            The keepdims parameter could be handy
        """
        #print("prev", logit)
        #logit_scaled = logit + 
        #print("scaled logit", logit_scaled)
        max_values = np.max(logit, axis=1, keepdims=True)
        logit_scaled = logit - max_values
        #print("scaled", logit_scaled)
        logit_scaled = np.exp(logit_scaled)
        logit_sums = np.sum(logit_scaled, axis=1, keepdims=True)
        logit_logs = np.log(logit_sums)
        #print(max_values+logit_logs)
        return logit_logs+max_values
        

    # for undergraduate student
    def normalPDF(self, points, mu_i, sigma_i):  # [5pts]
        """
        Args:
            points: N x D numpy array
            mu_i: (D,) numpy array, the center for the ith gaussian.
            sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            pdf: (N,) numpy array, the probability density value of N data for the ith gaussian

        Hint:
            np.diagonal() should be handy.
        """
        sig_diag = sigma_i.diagonal()
        numerator = np.exp(((np.square(points - mu_i))/(-2*sig_diag)))
        denom = np.sqrt((2*np.pi*sig_diag))
        return np.prod(numerator/denom, axis=1)

    # for grad students
    def multinormalPDF(self, points, mu_i, sigma_i):  # [5pts]
        """
        Args:
            points: N x D numpy array
            mu_i: (D,) numpy array, the center for the ith gaussian.
            sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            normal_pdf: (N,) numpy array, the probability density value of N data for the ith gaussian

        Hint:
            1. np.linalg.det() and np.linalg.inv() should be handy.
            2. The value in self.D may be outdated and not correspond to the current dataset,
            try using another method involving the current arguments to get the value of D
        """

        raise NotImplementedError


    def create_pi(self):
        """
        Initialize the prior probabilities 
        Args:
        Return:
        pi: numpy array of length K, prior
        """
        pi = np.ones(self.K)
        pi *= 1/self.K
        #print(pi)
        return pi
        #return NotImplementedError

    def create_mu(self):
        """
        Intialize random centers for each gaussian
        Args:
        Return:
        mu: KxD numpy array, the center for each gaussian.
        """
        indexes = np.random.uniform(0, self.N, self.K)
        indexes = indexes.astype(int)
        output = self.points[indexes]
        return output
        #return NotImplementedError
    
    def create_sigma(self):
        """
        Initialize the covariance matrix with np.eye() for each k. For grads, you can also initialize the 
        by K diagonal matrices.
        Args:
        Return:
        sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
            You will have KxDxD numpy array for full covariance matrix case
        """
        sigma = [np.eye(self.D)] * self.K
        #print(sigma)
        return np.asarray(sigma)
    
    def _init_components(self, **kwargs):  # [5pts]

        """
        Args:
            kwargs: any other arguments you want
        Return:
            pi: numpy array of length K, prior
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
                You will have KxDxD numpy array for full covariance matrix case

            Hint: np.random.seed(5) may be used at the start of this function to ensure consistent outputs.
        """
        np.random.seed(5) #Do Not Remove Seed

        #print(self.create_mu)
        return self.create_pi(), self.create_mu(), self.create_sigma()

    def _ll_joint(self, pi, mu, sigma, full_matrix=FULL_MATRIX, **kwargs):  # [10 pts]
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.

        Return:
            ll(log-likelihood): NxK array, where ll(i, k) = log pi(k) + log NormalPDF(points_i | mu[k], sigma[k])
        """

        # === graduate implementation
        #if full_matrix is True:
            #...

        ll = []
        # === undergraduate implementation
        if full_matrix is False:
            for i in range(self.K):
                #calculate log(pi(k))
                log_pi = np.log(pi[i] + 1e-32)
                #calculate pdf 
                pdf = self.normalPDF(self.points, mu[i], sigma[i])
                log_pdf = np.log(pdf + 1e-32)
                ll.append(log_pdf + log_pi)
            ll = np.asarray(ll)
            ll = ll.transpose()
            #print("ll", np.shape(ll))
            return ll
                
                
        #raise NotImplementedError

    def _E_step(self, pi, mu, sigma, full_matrix = FULL_MATRIX , **kwargs):  # [5pts]
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.

        Hint:
            You should be able to do this with just a few lines of code by using _ll_joint() and softmax() defined above.
        """
        # === graduate implementation
        #if full_matrix is True:
            # ...

        # === undergraduate implementation
        if full_matrix is False:
            #first calcuclate the ll
            joint_ll = self._ll_joint(pi,mu,sigma)
            return self.softmax(joint_ll)
            

    def _M_step(self, gamma, full_matrix=FULL_MATRIX, **kwargs):  # [10pts]
        """
        Args:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case

        Hint:
            There are formulas in the slides and in the Jupyter Notebook.
            Undergrads: To simplify your calculation in sigma, make sure to only take the diagonal terms in your covariance matrix
        """
        # === graduate implementation
        #if full_matrix is True:
            # ...

        # === undergraduate implementation
        if full_matrix is False:
            #calculate mu 
            gamma_sum = np.sum(gamma, axis=0)
            mu = []
            for i in range(self.K):
                gamma_col = gamma[:,i]
                gamma_col = np.reshape(gamma_col, (self.N, 1))
                #code for mu 
                mu_numerator = gamma_col*self.points
                mu_numerator = np.sum(mu_numerator, axis=0)
                mu.append(mu_numerator/gamma_sum[i])
            
            mu = np.asarray(mu)

            pi = gamma_sum/self.N
            sigma = []
            for i in range(self.K):
                gamma_col = gamma[:,i]
                gamma_col = np.reshape(gamma_col, (self.N, 1))
                x_minus_mu = self.points - mu[i]
                x_minus_mu_t = np.transpose(x_minus_mu)
                total_prod = gamma_col * x_minus_mu
                total_prod = x_minus_mu_t @ total_prod
                total_prod = total_prod/np.sum(gamma_col, axis=0)
                #print(total_prod)
                #multiply by the identity
                diagonals_only = np.eye(self.D)*total_prod
                sigma.append(diagonals_only)
            sigma = np.asarray(sigma)
            return pi, mu, sigma

    def __call__(self, full_matrix=FULL_MATRIX, abs_tol=1e-16, rel_tol=1e-16, **kwargs):  # No need to change
        """
        Args:
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want

        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            (pi, mu, sigma): (1xK np array, KxD numpy array, KxDxD numpy array)

        Hint:
            You do not need to change it. For each iteration, we process E and M steps, then update the paramters.
        """
        pi, mu, sigma = self._init_components(**kwargs)
        pbar = tqdm(range(self.max_iters))

        for it in pbar:
            # E-step
            gamma = self._E_step(pi, mu, sigma, full_matrix)

            # M-step
            pi, mu, sigma = self._M_step(gamma, full_matrix)

            # calculate the negative log-likelihood of observation
            joint_ll = self._ll_joint(pi, mu, sigma, full_matrix)
            loss = -np.sum(self.logsumexp(joint_ll))
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            pbar.set_description('iter %d, loss: %.4f' % (it, loss))
        return gamma, (pi, mu, sigma)

