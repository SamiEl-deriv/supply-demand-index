import numpy as np
import matplotlib.pyplot as plt
import numba as nb

dt = 1 / 365 / 24 / 60 / 60

def init_P0(T0):
    P0 = np.diag(1-1/T0)
    for j in range(len(T0)):
        loc_j = [k for k in range(len(T0)) if k != j]
        v = np.random.rand(len(T0)-1)
        P0[loc_j,j] = v / np.sum(v) / T0[j]
    return P0

def EM_algo_general(log_ratio : np.ndarray, NIt, P0, mu0, sigma0, zeta0, verbose=False):

    # initialisation
    P = np.zeros((np.shape(P0)[0],np.shape(P0)[1], NIt)) # Transition matrix
    mu = np.zeros((np.shape(mu0)[0], NIt)) # drifts
    sigma = np.zeros((np.shape(sigma0)[0], NIt)) # volatility
    zeta = np.zeros((np.shape(zeta0)[0], NIt)) # forecast_probabilities [0,1,2]
    P[:,:,0] , mu[:,0], sigma[:,0], zeta[:,0] = P0, mu0, sigma0, zeta0

    for n in range(1,NIt):
        if verbose: print("NIt = "+str(n))

        # EXPECTATION STEP
        # Get observed probabilities on normal distribution prior
        obs = np.zeros((np.shape(mu)[0],len(log_ratio)))
        for i in range(len(mu)):
            obs[i] = np.exp(-(np.array(log_ratio - mu[i,n-1])) ** 2 / (2 * sigma[i,n-1] ** 2)) / (np.sqrt(2 * np.pi) * sigma[i,n-1])
        #print(obs)
        
        obs_shape = obs.shape
        ci = np.zeros(obs_shape[1])
        xi_inference = np.zeros((obs_shape[0], obs_shape[1] + 1))
        xi_inference[:,-1] = zeta[:,n-1]
        
        # forward step / xi inferencing
        for j in range(obs_shape[1]):
            
            
            xi_inference[:,j] = P[:,:,n-1].dot(xi_inference[:,j-1]) # inference to forecast

            # Steps to get inference from previous forecast
            xi_inference[:,j] = xi_inference[:,j] * obs[:,j] 
            ci[j] = np.linalg.norm(xi_inference[:,j], ord=1)
            xi_inference[:,j] = xi_inference[:,j] / ci[j]

        # forward step / xi forecasting (backwards order)
        xi_forecast = np.zeros((obs_shape[0], obs_shape[1] + 1))
        xi_forecast[:,0] = P[:,:,n-1].dot(xi_inference[:,-1])
        xi_forecast[:,1:] = P[:,:,n-1].dot(xi_inference[:,:-1])

        # backward step
        chi = np.zeros((obs_shape[0], obs_shape[1]))
        phi = np.zeros((obs_shape[0], obs_shape[0], obs_shape[1]-1))
        chi[:,-1] = xi_inference[:,-2]

        for k in range(obs_shape[1]-2,-1,-1):
            phi[:,:,k] = (np.diag(chi[:,k+1]/xi_forecast[:,k+1]) @ P[:,:,n-1]) * xi_inference[:,k]
            chi[:,k] = np.sum(phi[:,:,k], axis=0)

        score = np.sum(np.log(np.sum(xi_forecast[:,:-1] * obs[:,:], axis=0)))
        if verbose: print("score = "+str(score))

        


        # MAXIMIZATION STEP

        for l in range(len(mu)):
            mu[l,n] = np.average(log_ratio, weights=chi[l,:])
            sigma[l,n] = np.sqrt(np.average((log_ratio-mu[l,n])**2, weights=chi[l,:]))

        zeta[:,n] = chi[:,0].dot(P[:,:,n-1])
        zeta[:,n] /= np.sum(zeta[:,n])

        P[:,:,n] = np.mean(phi, axis=2)
        P[:,:,n] /= np.sum(P[:,:,n], axis=0)


        

    return P, mu, sigma, zeta, xi_inference, xi_forecast, chi, phi


def see_convergence(P, mu, sigma, zeta, feed):

    fig, axes = plt.subplots(2, 4, figsize=(30,12))
    # zeta
    for i in range(zeta.shape[0]):
        axes[0,0].plot(zeta[i,:], label=f'zeta_{i}')
    axes[0,0].set_xlabel('Number of Iterations')
    axes[0,0].set_ylabel('Probability')
    axes[0,0].set_title('Evolution of zeta')
    axes[0,0].legend(loc='upper left')

    # vol
    vol = sigma/np.sqrt(dt)
    for i in range(vol.shape[0]):
        axes[0,1].plot(vol[i,:], label=f'vol_{i}')
    axes[0,1].set_xlabel('Number of Iterations')
    axes[0,1].set_ylabel('Volatility')
    axes[0,1].set_title(f'Evolution of volatility')
    axes[0,1].legend(loc='upper left')

    # mu
    drift = mu / dt + 0.5 * vol**2
    for i in range(mu.shape[0]):
        axes[0,2].plot(drift[i,:], label=f'drift_{i}')
    axes[0,2].set_xlabel('Number of Iterations')
    axes[0,2].set_ylabel('Drift')
    axes[0,2].set_title(f'Evolution of drift')
    axes[0,2].legend(loc='upper left')

    # T
    T = np.zeros((P.shape[0],P.shape[2]))
    for j in range(P.shape[2]):
        T[:,j] = 1 / (1-np.diag(P[:,:,j]))
    for i in range(T.shape[0]):
        axes[0,3].plot(T[i,:], label=f'T_{i}')
    axes[0,3].set_xlabel('Number of Iterations')
    axes[0,3].set_ylabel('Characteristic Time of Regime')
    axes[0,3].set_title('Evolution of Characteristic Time')
    axes[0,3].legend(loc='upper left')

    # Transitions
    for j in range(P.shape[1]):
        loc_j = [i for i in range(P.shape[0]) if i != j]
        sumPj = np.sum(P[loc_j,j,:], axis=0)
        for i in loc_j:
            axes[1,j].plot(P[i,j,:]/sumPj, label=f'pi_{i}{j}')
        axes[1,j].set_xlabel('Number of Iterations')
        axes[1,j].set_ylabel('Transition Probability')
        axes[1,j].set_title(f'Evolution of pi[:,{j}]')
        axes[1,j].legend(loc='center left')
    axes[1,-1].remove()
    plt.suptitle(f'EM-Algorithm Iterations for {feed}')
    plt.tight_layout()

if __name__ == "__main__":
    from DSI import DSI_Engine

    DSI_TYPE = 'DSI30'
    DSI_engine = DSI_Engine(DSI_TYPE)
    specs = DSI_Engine.get_DSI_base_specs(DSI_TYPE)
    T = specs['T']
    mu = specs['mu']
    sigma = specs['sigma']

    spot, _, _ = DSI_engine.generate_DSI_index(num_steps=86400 * 7)
    log_returns = np.diff(spot, 1)
    T = 30 * 60
    T0 = np.ones(3)* T
    P0 = init_P0(T0)
    mu0 = np.array([1,0,-1])*dt * mu
    sigma0 = np.array([1,1,1])*np.sqrt(dt) * sigma
    zeta0 = np.ones(3) / 3
    NIt = 20

    P_EM, mu_EM, sigma_EM, zeta_EM, alpha_EM, beta_EM, chi_EM, phi_EM = EM_algo_general(log_returns,NIt,P0,mu0,sigma0,zeta0)
    