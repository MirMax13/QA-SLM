import matplotlib.pyplot as plt

# Data
epochs = list(range(1, 11))
training_loss = [1.0446, 0.6821, 0.4251, 0.3321, 0.2962, 0.2023, 0.2081, 0.1791, 0.1587, 0.1489] #OpenChat
validation_loss = [0.838761, 0.549958, 0.430502, 0.366793, 0.329805, 0.297480, 0.277527, 0.267254, 0.259236, 0.2555]
# training_loss = [0.965000, 0.566300, 0.364600, 0.270100, 0.198400, 0.127400, 0.129500, 0.109600, 0.097000, 0.111200] #ChatGPT
# validation_loss = [0.751405, 0.408572, 0.264694, 0.179222, 0.132990, 0.104054, 0.087095, 0.076048, 0.070303, 0.067710]


# Plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, training_loss, marker='o', label='Training Loss')
plt.plot(epochs, validation_loss, marker='o', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs (OpenChat)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
