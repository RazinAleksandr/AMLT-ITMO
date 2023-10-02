import pandas as pd
from pandas import concat


# trajectory matrix for solving supervized learning task
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # входящая последовательность (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # прогнозируемая последовательность (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    agg = concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# def inverse_difference(last_ob, forecast):
#     inverted = [forecast[0] + last_ob]
#     for i in range(1, len(forecast)):
#         inverted.append(forecast[i] + inverted[i-1])
#     return inverted

# def inverse_transform(series, forecasts, scaler, n_test):
#     inverted = []

#     for i, forecast in enumerate(forecasts):
#         # Convert forecast to numpy
#         forecast_np = np.array(forecast).reshape(1, -1)
        
#         # Invert scaling
#         inv_scale = scaler.inverse_transform(forecast_np)
        
#         # Invert differencing
#         index = len(series) - n_test + i - 1
#         last_ob = series.iloc[index]
#         inv_diff = inverse_difference(last_ob, inv_scale[0])
        
#         # Store
#         inverted.append(inv_diff)
    
#     return inverted
