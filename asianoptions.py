import numpy as np

def payoff(S, K):
    return max(S - K, 0)

def monte_carlo(S, K, T, r, sigma, sim, payoff, steps):
    dt = T / steps
    payoff_sum = 0
    Payoff = lambda S: payoff(S=S, K=K)

    for i in range(sim):
        # Initialize the stock price vector
        stock = np.zeros(steps + 1)
        stock[0] = S  # Set the initial stock price

        # Simulate the stock prices
        for t in range(1, steps + 1):
            stock[t] = stock[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * np.random.normal(0, 1))

        # Find the mean stock value
        avg_stock = np.mean(stock)
        payoff_sum += Payoff(avg_stock)

    # Return the discounted average of the payoffs
    return np.exp(-r * T) * payoff_sum / sim

S = 100  # Current stock price
K = 105  # Strike price
T = 1    # Time to maturity in years
r = 0.05 # Risk-free interest rate
sigma = 0.2 # Volatility
sim = 10000  # Number of simulations
steps = 20

price_asian_option = monte_carlo(S, K, T, r, sigma, sim, payoff, steps)

print(f"Price of an asian option using monte carlo is: {price_asian_option}")
