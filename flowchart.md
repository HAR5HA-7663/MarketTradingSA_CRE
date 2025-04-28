# Enhanced Stock Prediction Flowchart

```mermaid
flowchart TD
    A1["Day 1 Features"] --> LSTM
    A2["Day 2 Features"] --> LSTM
    A3["Day 3 Features"] --> LSTM
    A4["Day 4 Features"] --> LSTM
    A5["Day 5 Features"] --> LSTM

    subgraph Rolling_Input_Window [Input Sequence: 5 Days of Features]
      A1
      A2
      A3
      A4
      A5
    end

    LSTM["LSTM Layer (hidden=32)"]
    LSTM --> OUT["Dense Output Layer"]
    OUT --> PRED["Predicted Next-Day Return"]

    TGT["True Next-Day Return"]
    PRED -- "Compare (Loss MSE)" --> LOSS["Loss"]
    TGT  -- "Compare (Loss MSE)" --> LOSS
    LOSS -. "Backpropagation" .-> LSTM
```

- **Start at the top**: Input features for the last 5 days.
- **Rolling Window**: Use a rolling input window of 5 days of features.
- **LSTM Layer**: Process the input sequence through an LSTM layer with 32 hidden units.
- **Dense Output**: Generate the predicted next-day return using a dense output layer.
- **Loss Calculation**: Compare the predicted return with the true return using Mean Squared Error (MSE) loss.
- **Backpropagation**: Optimize the model using backpropagation based on the calculated loss.
