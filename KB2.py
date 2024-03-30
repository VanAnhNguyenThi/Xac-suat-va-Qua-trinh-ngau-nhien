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
num_symbols = 10000  # Number of symbols to transmit
snr_range_db = np.arange(0, 20, 1)  # Range of SNR values to test
num_monte_carlo_simulations = 10  # Number of Monte Carlo simulations

# Initialize arrays to store BER and SNR values
bers_avg = []
snrs = []

# Iterate over each SNR value
for snr_db in snr_range_db:
    bers_monte_carlo = []
    for _ in range(num_monte_carlo_simulations):
    # Simulate optical channel
        sent_symbols, received_signal = simulate_optical_channel(num_symbols, snr_db)

    # Decode received symbols using Bayes' theorem
        decoded_symbols = bayes_decode(received_signal, snr_db)

    # Estimate Bit Error Rate (BER)
        ber = round(estimate_ber(sent_symbols, decoded_symbols), 10)
        bers_monte_carlo.append(ber)
    # Append BER and SNR to arrays
        avg_ber = np.mean(bers_monte_carlo)

        bers_avg.append(avg_ber)
        snrs.append(snr_db)

# Plot BER-SNR curve
plt.plot(snrs, bers_avg, marker='o')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.title('BER vs. SNR')
plt.grid(True)
plt.show()
