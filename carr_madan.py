import numpy as np
from scipy.stats import norm
from functools import partial

#Some generic functions
def double_diff(f, dt, x):
    return (f(x + dt) + f(x - dt) - 2 * f(x)) / (dt * dt)

def integral(f,a,b,n):
    dx = (b-a)/n
    integral = 0
    for i in range(n):
        integral += f(a+i*dx)*dx
    return integral

#Payoff function for an option, in this case a european call option.
def payoff(S,K):
    #K = 105 #strike
    return max(S-K,0)

#Black scholes formula for call options
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = (S * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))
    return call_price

#Black scholes formula for put options
def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = (K * np.exp(-r * T) * norm.cdf(-d2)) - (S * norm.cdf(-d1))
    return put_price

#Implimentation of carr madan formula for option pricing.
def Carr_madam(payoff, S, K, T, r, sigma, sim):
    #Choice of increments, note this means that we have to end the integral as 2*S.
    dt = S / sim
    #Using partial to partially define the integrant functions.
    Payoff = lambda S: payoff(S=S,K=K)
    BS_put = lambda K: black_scholes_put(S=S, K=K, T=T, r=r, sigma=sigma)
    BS_call = lambda K: black_scholes_put(S=S,K=K,T=T,r=r,sigma=sigma)
    dd_payoff = partial(double_diff,Payoff,dt)

    #Taking the product of our functions
    put_term = lambda K: BS_put(K)*dd_payoff(K)
    call_term = lambda K: BS_call(K)*dd_payoff(K)

    #Returns the payoff of S, in addition to the integral of the put and call term.
    return Payoff(S) + integral(put_term,0.01,S,sim)+integral(call_term,S,2*S,sim)

def monte_carlo(S, K, T, r, sigma, sim,payoff):

    payoff_sum = 0
    Payoff = lambda S: payoff(S=S, K=K)
    #Deterministic term for the stock simulation
    det_term = S * np.exp((r - 0.5 * sigma ** 2) * T)

    for i in range(sim):
        #Simulation of a stock
        s = det_term * np.exp(sigma * np.random.normal(0,1) * np.sqrt(T))
        payoff_sum += Payoff(s)

    #Return the discounted avarege of the payoffs.
    return np.exp(-r * T) * payoff_sum / sim

# Example
S = 100  # Current stock price
K = 105  # Strike price
T = 1    # Time to maturity in years
r = 0.05 # Risk-free interest rate
sigma = 0.2 # Volatility
sim = 10000  # Number of simulations

price_carr_madan = Carr_madam(payoff, S, K, T, r, sigma, sim)

price_monte_carlo = monte_carlo(S, K, T, r, sigma, sim, payoff)

print(f"The European call option price using black-scholes is: {black_scholes_call(S, K, T, r, sigma)}")
print(f"The European call option price using Monte Carlo is: {price_monte_carlo}")
print(f"Option Price using Carr-Madan Approach: {price_carr_madan}")

