import numpy as np
import matplotlib.pyplot as plt

# Function to simulate optical communication channel with noise
def simulate_optical_channel(num_symbols, snr_db):
    # Generate random symbols (1 for 'on', 0 for 'off')
    symbols = np.random.randint(0, 2, num_symbols)

    # Modulate symbols to optical signal (1 for 'on', -1 for 'off')
    optical_signal = 2 * symbols - 1

    # Add Gaussian noise to the optical signal
    noise_power = 1 / (10 ** (snr_db / 10))  # Calculate noise power from SNR
    noise = np.random.normal(0, np.sqrt(noise_power), num_symbols)
    received_signal = optical_signal + noise

    return symbols, received_signal

# Function to decode received symbols using Bayes' theorem
def bayes_decode(received_signal, snr_db):
    decoded_symbols = np.zeros_like(received_signal)
    for i, signal in enumerate(received_signal):
        # Calculate likelihoods for each symbol being 'on' or 'off'
        likelihood_on = np.exp(-0.5 * ((signal - 1) ** 2) / (1 / (10 ** (snr_db / 10))))
        likelihood_off = np.exp(-0.5 * ((signal + 1) ** 2) / (1 / (10 ** (snr_db / 10))))

        # Calculate posterior probabilities using Bayes' theorem
        posterior_on = likelihood_on / (likelihood_on + likelihood_off)
        posterior_off = likelihood_off / (likelihood_on + likelihood_off)

        # Make decision based on posterior probabilities
        if posterior_on > posterior_off:
            decoded_symbols[i] = 1
        else:
            decoded_symbols[i] = 0

    return decoded_symbols

# Function to estimate bit error rate (BER)
def estimate_ber(sent_symbols, received_symbols):
    return np.sum(sent_symbols != received_symbols) / len(sent_symbols)

# Parameters
# List of symbol numbers to test
symbol_numbers = [100, 1000, 5000, 10000, 20000, 50000, 100000, 500000]
snr_db = 10
# Initialize arrays to store BER and symbol numbers
# List of symbol numbers to test
num_monte_carlo_simulations = 10  # Number of Monte Carlo simulations

# Initialize arrays to store BER and symbol numbers
bers_avg = []  # Store average BER
symbol_nums = []

# Iterate over each symbol number
for num_symbols in symbol_numbers:
    bers_monte_carlo = []  # Store BER of each Monte Carlo simulation
    for _ in range(num_monte_carlo_simulations):
        # Simulate optical channel
        sent_symbols, received_signal = simulate_optical_channel(num_symbols, snr_db)

        # Decode received symbols using Bayes' theorem
        decoded_symbols = bayes_decode(received_signal, snr_db)

        # Estimate Bit Error Rate (BER)
        ber = round(estimate_ber(sent_symbols, decoded_symbols), 10)
        bers_monte_carlo.append(ber)

    # Calculate average BER of Monte Carlo simulations
    avg_ber = np.mean(bers_monte_carlo)

    # Append average BER and symbol number to arrays
    bers_avg.append(avg_ber)
    symbol_nums.append(num_symbols)

# Plot BER-symbol number curve
plt.plot(symbol_nums, bers_avg, marker='o')
plt.xlabel('Number of Symbols')
plt.ylabel('Average BER')
plt.title('Average BER vs. Number of Symbols (Monte Carlo Simulations)')
plt.grid(True)
plt.show()
