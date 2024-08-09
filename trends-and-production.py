import pandas as pd
import matplotlib.pyplot as plt

# Load the provided Excel files
production_df = pd.read_excel('inputs/annual-scientific-production.xlsx')
trends_df = pd.read_excel('inputs/word-frequency-over-time.xlsx')

# Define the trend names for proper labeling
trend_names = {
    "QUANTUM CRYPTOGRAPHY": "Quantum Cryptography",
    "QUANTUM COMMUNICATION": "Quantum Communication",
    "QUANTUM COMPUTERS": "Quantum Computers",
    "NETWORK SECURITY": "Network Security",
    "QUANTUM THEORY": "Quantum Theory",
    "QUANTUM ENTANGLEMENT": "Quantum Entanglement",
    "QUANTUM OPTICS": "Quantum Optics",
    "SECURE COMMUNICATION": "Secure Communication",
    "PHOTONS": "Photons",
    "QUANTUM COMPUTING": "Quantum Computing"
}

# Plot Annual Scientific Production
fig, ax1 = plt.subplots(figsize=(14, 8))
ax1.plot(production_df["Year"], production_df["Articles"], marker='o', linestyle='-', color='blue')
ax1.set_xlabel('Year', fontsize=14)
ax1.set_ylabel('Number of Articles', fontsize=14)
ax1.tick_params(axis='y')
ax1.grid(True)
plt.xticks(production_df["Year"], rotation=45)

# Annotate each point with the number of articles
for i in range(len(production_df)):
    ax1.text(production_df["Year"][i], production_df["Articles"][i] + 5, str(production_df["Articles"][i]), ha='center', fontsize=9, color='blue')

# Save the figure
plt.savefig('outputs/annual_scientific_production.png')
plt.show()

# Plot Trends in Quantum Technology Topics
fig, ax2 = plt.subplots(figsize=(14, 8))

# Plot each trend line with proper naming
for column in trends_df.columns[1:]:
    ax2.plot(trends_df["Year"], trends_df[column], marker='o', linestyle='-', label=trend_names[column])

# Add labels and title
ax2.set_xlabel('Year', fontsize=14)
ax2.set_ylabel('Number of Articles', fontsize=14)
ax2.grid(True)
plt.xticks(trends_df["Year"], rotation=45)

# Add legend
fig.tight_layout()
fig.legend(title='Trends', loc='upper left', bbox_to_anchor=(0.1, 0.9))

# Save the figure
plt.savefig('outputs/trends_in_quantum_technology.png')
plt.show()
