import numpy as np
from scipy import optimize, linalg, interpolate

class BaseFitter:
    """This is an abstract base class. ABSs are classes that are useless on thier own, but provide
    code that can be used by subclasses.
    """

    def data(self):
        raise NotImplementedError("Not implemented in base class.")

    def uncertainties(self):
        raise NotImplementedError("Not implemented in base class.")

    def model(self, parameters):
        """Returns model predictions for the expectation value of the data, as a function of the
        parameters.
        Parameters are packed into a 1D array.
        """
        raise NotImplementedError("Not implemented in base class.")

    def weighted_residuals(self, parameters):
        return (self.data() - self.model(parameters)) / self.uncertainties()

    def fit(self, parameters0):
        result = optimize.least_squares(self.weighted_residuals, parameters0)
        pars_est = result.x
        jacobian = result.jac
        covariance_inv = np.dot(jacobian.T, jacobian)
        covariance = linalg.inv(covariance_inv)

        return pars_est, covariance

    def chi_squared(self, parameters):
        return np.sum(self.weighted_residuals(parameters)**2)
    
class matter_power_fitter(BaseFitter):
       
        def __init__(self, P_g_lambdas, P_g_lambdas_err, P_k , ks, k_bin_centers, k0):
            '''
                P_g_lambdas: values for galaxy power spectrum estimated from data (at bin centers)
                P_g_lambdas_err: values for error on galaxy power spectrum estimate
                P_k: Our matter power spectrum theory given by Kiyo
                ks: k values for matter power spectrum theory given by Kiyo
                k_bin_centers: k values for center of bins used to calculate P_g_lambdas
                k0: pivot scale for bias parameters
            '''
            
            self._P_g_lambdas = P_g_lambdas
            self._P_g_lambdas_err = P_g_lambdas_err
            self._P_k = interpolate.interp1d(ks, P_k)
            self._k_bin_centers = k_bin_centers
            self._k0 = k0
            
        def data(self):
            return self._P_g_lambdas
            
        def uncertainties(self):
            return self._P_g_lambdas_err
        
        def fit(self, parameters0):
            bounds = ([-.1, -np.inf, -np.inf, -np.inf],[.1, np.inf, np.inf, np.inf])
            result = optimize.least_squares(self.weighted_residuals, parameters0, bounds = bounds)
            
            pars_est = result.x
            jacobian = result.jac
            covariance_inv = np.dot(jacobian.T, jacobian)
            covariance = linalg.inv(covariance_inv)
            
            self._pars_est = pars_est
            self._covariance = covariance
            return pars_est, covariance
         
         
            
           
        def model(self, parameters):
            '''
                parameters: [alpha, b0, b1, b2]
            
            '''
            alpha, b0, b1, b2 = parameters
            
            k0 = self._k0
            k_bin_centers = self._k_bin_centers  
            
             
            
            
            # get P_k at bin centers
            P_k_bin_centers = self._P_k(k_bin_centers)
            
            model_values = np.array([])
            
            for i,k in enumerate(k_bin_centers):
                P_g_k = ((b0 + ((k / k0) * b1) +  ((k / k0)**2) * b2)**2) * P_k_bin_centers[i]
                model_value = ((1 + alpha)**3) * P_g_k
                model_values = np.append(model_values, model_value)
                
            return model_values
        
        
            
                
            
            
