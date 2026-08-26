import numpy as np
import scipy as sp

def modified_thomas_algorithm(lower, main, upper, rhs, K, dS, put=False):
    # Since we handle the boundaries separately, the length of the diagonal will be 2 less than the grid on which we compute V
    a = lower.copy()
    b = main.copy()
    c = upper.copy()
    d = rhs.copy()
    n = len(main)

    # Forward sweep eliminating lower diagonal
    c[0] /= b[0]
    d[0] /= b[0]

    for i in range(1, n-1):
        # In our mathematical notation, the first entry in the lower diagonal is denoted a_2 because it first appears on the 2nd row of the matrix.
        # We must therefore reduce our indexing for the lower diagonal by 1

        denominator = b[i]-a[i-1]*c[i-1]
        c[i] /= denominator
        d[i] = (d[i] - a[i-1]*d[i-1]) / denominator

     # d has one more dimension than the upper diagonal, so we handle it here.
    denominator = b[n-1]-a[n-2]*c[n-2]
    d[n-1] = (d[n-1] - a[n-2]*d[n-2]) / denominator

    b = np.ones(n)

    # Backward sweep to solve applying early-exercise constraint
    solution = np.zeros(n)
    solution[n-1] = d[n-1]

    for i in range(2, n+1):
        solution[n-i] = d[n-i] - c[n-i]*solution[n-i+1]
        if put:
            solution[n-i] = np.maximum(solution[n-i], K - dS*(n+1-i) )
        else:
            solution[n-i] = np.maximum(solution[n-i], dS*(n+1-i) - K )

    return solution

def BS_solver_crank_nicolson(S_upper, T, K, r, sigma, q=None, put=False, American=False, m=1000, n=1000):

    assert (S_upper > K)

    # Time and stock increments
    dS = S_upper/m
    dt = T/n

    # Initialise matrix as 0s. This also takes care of the one of the boundary conditions.
    V = np.zeros((m+1,n+1))

    # Terminal and far-field conditions depending on call or put
    if put:
        V[:,n] = np.maximum([K-i*dS for i in np.arange(m+1)], 0)
        # Because of early exercise, we do not need to discount the strike price to present value for American options
        if American:
            V[0, :] = K * np.ones(n+1)
        else:
            V[0, :] = [ K*np.exp(-r*(T-j*dt)) for j in range(n+1) ]
    else:
        V[:,n] = np.maximum([i*dS-K for i in np.arange(m+1)], 0)
        V[m, :] = [ S_upper - K*np.exp(-r*(T-j*dt)) for j in range(n+1) ]

    # Define our matrix A
    lower_A = np.array([ dt*i*(r - i*(sigma**2)) for i in range(2,m) ])
    main_A = np.array([ 4+2*dt*(r+(i**2)*sigma**2) for i in range(1,m) ])
    upper_A = np.array([ -dt*i*(r + i*(sigma**2)) for i in range(1,m-1) ])
    A = sp.sparse.diags([lower_A, main_A, upper_A], offsets=(-1,0,1), format='csc')

    # Define our matrix D
    lower_D = -lower_A
    main_D = np.array([ 4-2*dt*(r+(i**2)*sigma**2) for i in range(1,m) ])
    upper_D = -upper_A

    D = sp.sparse.diags([lower_D, main_D, upper_D], offsets=(-1,0,1), format='csc')
    # Start loop
    for j in range(n):
        # Compute the RHS of the matrix equation.
        d = D @ V[1:m, n-j]
        # In order to apply the boundary condition we must update the either the last entry of d for a call, or the first entry for a put
        if put:
            d[0] += dt*(sigma**2 - r)*( V[0,n-j] + V[0,n-j-1] )
        else:
            d[m-2] += dt*(m-1)*((m-1)*sigma**2 + r)*(V[m, n-j] + V[m, n-j-1])

        if American:
            V[1:m, n-j-1] = modified_thomas_algorithm(lower_A, main_A, upper_A, rhs=d, K=K, dS=dS, put=put)
        else:
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
