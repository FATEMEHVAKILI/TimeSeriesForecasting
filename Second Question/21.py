import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, RepeatVector
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv(
    'Agriculture, forestry, and fishing, value added (% of GDP).csv')

# Convert 'Year' column to datetime and set it as index (keeping original data intact)
data['Year'] = pd.to_datetime(data['Year'])
data_original = data.set_index('Year')  # Keep original DataFrame

data_values = data_original.values.astype(
    'float32')  # Use values for scaling and modeling
scaler = MinMaxScaler(feature_range=(-1, 1))
data_scaled = scaler.fit_transform(data_values)

# Create sequences
timestep = 5
X, Y = [], []
for i in range(len(data_scaled) - timestep):
    X.append(data_scaled[i:i + timestep])
    Y.append(data_scaled[i + timestep])

X, Y = np.array(X), np.array(Y)

# Split into train and test sets
k = 40
X_train, X_test = X[:k], X[k:]
Y_train, Y_test = Y[:k], Y[k:]

# Reshape for CNN-LSTM model
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Define CNN-LSTM Model
model = Sequential()
model.add(Conv1D(256, kernel_size=2, activation='relu', input_shape=(timestep, 1)))
model.add(Conv1D(128, kernel_size=2, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(RepeatVector(timestep))
model.add(LSTM(128, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

# Train the model
history = model.fit(X_train, Y_train, epochs=1000, verbose=1)

# Evaluate the model
predictions = model.predict(X_test)
predictions_inverse = scaler.inverse_transform(predictions)
Y_test_inverse = scaler.inverse_transform(Y_test.reshape(-1, 1))

# Show actual vs predicted values
plt.figure(figsize=(10, 5))
plt.plot(Y_test_inverse, label='Actual')
plt.plot(predictions_inverse, label='Predicted', linestyle='--')
plt.legend()
plt.title("Actual vs Predicted Values")

# Save the plot as a JPG file
plt.savefig('ActualvsPredicted_CNNLSTM.jpg', format='jpg')
plt.show()

# Calculate mean squared error
mse = mean_squared_error(Y_test_inverse, predictions_inverse)
print(f'Mean Squared Error: {mse}')

# Forecasting for the next 5 years
forecast_steps = 5
forecasted_values = []

# Use the last 'timestep' records as the starting point for forecasting
current_input = data_scaled[-timestep:]

for _ in range(forecast_steps):
    current_input_reshaped = current_input.reshape(1, timestep, 1)
    next_value = model.predict(current_input_reshaped)
    forecasted_values.append(next_value[0, 0])
    current_input = np.append(current_input[1:], next_value)

# Inverse transform the forecasted values
forecasted_values_inverse = scaler.inverse_transform(
    np.array(forecasted_values).reshape(-1, 1))

# Generate future dates based on the original DataFrame's index
last_date = data_original.index[-1]
future_dates = [last_date +
                pd.DateOffset(years=i + 1) for i in range(forecast_steps)]

# Create a DataFrame for future dates and forecasted values
forecast_df = pd.DataFrame(forecasted_values_inverse,
                           index=future_dates, columns=['Forecasted Value'])

# Print forecast results
print(forecast_df)

# Plot the forecasted values
plt.figure(figsize=(10, 5))
plt.plot(data_original.index, data_original.values, label='Historical Data')
plt.plot(forecast_df.index, forecast_df['Forecasted Value'],
         label='Forecast', linestyle='--', color='orange')
plt.legend()
plt.title("Forecast for the Next 5 Years")
plt.xlabel("Year")
plt.ylabel("Value")

# Save the final forecast plot as a JPG file
plt.savefig('ForecastNextYears_CNNLSTM.jpg', format='jpg')
plt.show()
