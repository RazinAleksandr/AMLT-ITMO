import pandas as pd


def postprocess(forecasts, real_data, lag_input, scaler):
    # Generate column names
    num_cols = len(forecasts[0])
    column_names = [f"predicted_{i+1}" for i in range(num_cols)]

    # Convert list to dataframe
    df_forecasts = pd.DataFrame(forecasts, columns=column_names, index=real_data.index)

    final = pd.concat([real_data.iloc[:, :lag_input], df_forecasts], axis=1)
    unscaled = pd.DataFrame(scaler.inverse_transform(final), columns=final.columns, index=final.index)
    return unscaled

def arima_postprocess(forecasts, real_data, lag_input, scaler):
    # Generate column names
    num_cols = len(forecasts[0])
    column_names = [f"predicted_{i+1}" for i in range(num_cols)]

    # Convert list to dataframe
    df_forecasts = pd.DataFrame(forecasts, columns=column_names)
    
    # Select the desired row from the first DataFrame
    selected_row = real_data.iloc[0:1, :lag_input]

    # Reset the index of the second DataFrame to match the index of the selected row
    df_forecasts.index = selected_row.index

    # Concatenate the two rows
    final = pd.concat([selected_row, df_forecasts], axis=1)

    unscaled = pd.DataFrame(scaler.inverse_transform(final), columns=final.columns, index=final.index)
    return unscaled