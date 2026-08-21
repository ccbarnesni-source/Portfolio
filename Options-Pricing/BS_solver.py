import numpy as np
import scipy as sp

def BS_solver_crank_nicholson(S_upper, T, K, r, sigma, q=None, put=False, American=False, m=1000, n=1000):

    # Time and stock increments
    dS = S_upper/m
    dt = T/n

    # Initialise matrix as 0s. This also takes care of the one of the boundary conditions.
    V = np.zeros((m+1,n+1))

    # Terminal and far-field conditions depending on call or put
    if put:
        V[:,n] = np.maximum([K-i*dS for i in np.arange(m+1)], 0)
        V[0, :] = [ K*np.exp(-r*(T-j*dt)) for j in range(n+1) ]
    else:
        V[:,n] = np.maximum([i*dS-K for i in np.arange(m+1)], 0)
        V[m, :] = [ S_upper - K*np.exp(-r*(T-j*dt)) for j in range(n+1) ]

    # Define our matrix A
    lower = np.array([ dt*i*(r - i*(sigma**2)) for i in range(2,m) ])
    main = np.array([ 4+2*dt*(r+(i**2)*sigma**2) for i in range(1,m) ])
    upper = np.array([ -dt*i*(r + i*(sigma**2)) for i in range(1,m-1) ])
    A = sp.sparse.diags([lower, main, upper], offsets=(-1,0,1), format='csc')

    # Define our matrix D
    lower = -lower
    main = np.array([ 4-2*dt*(r+(i**2)*sigma**2) for i in range(1,m) ])
    upper = -upper
    D = sp.sparse.diags([lower, main, upper], offsets=(-1,0,1), format='csc')

    # Start loop
    for j in range(n):
        # Compute the RHS of the matrix equation.
        d = D @ V[1:m, n-j]

        # In order to apply the boundary condition we must update the either the last entry of d for a call, or the first entry for a put
        if put:
            d[0] += dt*(sigma**2 - r)*( V[0,n-j] + V[0,n-j-1] )
        else:
            d[m-2] += dt*(m-1)*((m-1)*sigma**2 + r)*(V[m, n-j] + V[m, n-j-1])

        # V has dimensions m+1 x n+1 whereas A has dimensions m x m. We initialised V to be zeros so the first row does not need changing.
        V[1:m, n-j-1] = sp.sparse.linalg.spsolve(A, d)

    return V

def BS_solver_binomial_tree(S0, T, K, r, sigma, d=None, put=False, American=False, n=1000):

    # Initialise the matrix and other variables
    V = np.zeros((n+1,n+1))
    dt = T/n
    u = np.exp(sigma*np.sqrt(dt))
    d = np.exp(-sigma*np.sqrt(dt))
    p = (np.exp(r*dt)-d)/(u-d)
    q = 1-p


    # Set initial stock price
    V[0,0] = S0

    # Apply the terminal condition. The range of possible stock prices is the same regardless of whether it is a put or call.
    S = np.array([u**(n-i)*d**i*S0 for i in range(n+1)])
    if put:
        V[:,n] = np.maximum(K-S,0)
    else:
        V[:,n] = np.maximum(S-K,0)

    # Begin the loop, iterating backwards through time
    for j in range(n, 0, -1):
        V[0:j, j-1] = [ np.exp(-r*dt)*(p*V[i,j] + q*V[i+1,j]) for i in range(j) ]
        if American:
            S = np.array([u**(j-i-1)*d**i*S0 for i in range(j)])
            if put:
                V[0:j, j-1] = np.maximum(V[0:j, j-1], K-S)
            else:
                V[0:j, j-1] = np.maximum(V[0:j, j-1], S-K)

    return V
