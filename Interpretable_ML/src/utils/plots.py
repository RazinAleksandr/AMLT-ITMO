import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import numpy as np


def plot_series_with_dema(df, ax=None):
    series = df.columns.tolist()[-1]
    if ax is None:
        plt.figure(figsize=(15, 7))
        ax = plt.gca()
    ax.set_title(series)
    ax.plot(df[series], label='Actual Time Series', color='blue')
    ax.plot(df['DEMA Values'], label='DEMA', color='green', alpha=0.7)
    ax.fill_between(df.index, df['Upper Bound'], df['Lower Bound'], color='red', alpha=0.2)
    anomaly_indices = df[df['Anomalies'] == 1].index
    ax.scatter(anomaly_indices, df[series].iloc[anomaly_indices], color='red', label='Anomalies')
    ax.legend()


def plot_time_intervals_with_dema(processed_df, data_dict, intervals=['0-4', '4-9', '9-17', '17-24'], ax=None):
    # Use Seaborn's "Set2" color palette
    colors = sns.color_palette("Set2", n_colors=len(intervals))
    
    if ax is None:
        plt.figure(figsize=(15, 7))
        ax = plt.gca()

    bottom_values = np.zeros(len(processed_df))
    for interval, color in zip(intervals, colors):
        df = data_dict[interval]
        
        # with timestamp as x axis - work not correctly
        # ax.plot(df['date'], df[interval] + bottom_values, color=color,linewidth=3)
        # ax.fill_between(df['date'], df['Upper Bound'] + bottom_values, df['Lower Bound'] + bottom_values, color='red', alpha=0.1)
        # anomaly_indices = df[df['Anomalies'] == 1].index
        # ax.scatter(df['date'].iloc[anomaly_indices], df[interval].iloc[anomaly_indices] + bottom_values[anomaly_indices], color='red', s=50)
        # ax.bar(df['date'], df[interval], bottom=bottom_values, color=color, alpha=0.3)
        
        # with idx as x axis 
        ax.plot(df.index, df[interval] + bottom_values, color=color,linewidth=3)
        ax.fill_between(df.index, df['Upper Bound'] + bottom_values, df['Lower Bound'] + bottom_values, color='red', alpha=0.1)
        anomaly_indices = df[df['Anomalies'] == 1].index
        ax.scatter(anomaly_indices, df[interval].iloc[anomaly_indices] + bottom_values[anomaly_indices], color='red', s=50)
        ax.bar(df.index, df[interval], bottom=bottom_values, color=color, alpha=0.3)
        bottom_values += df[interval].values

    # Calculate the percentage of transactions for each interval for the legend
    percentages = (processed_df[intervals].sum() / processed_df[intervals].sum().sum()) * 100
    # Create custom legend
    legend_handles = [Patch(facecolor=color, label=f"{interval}: {percent:.2f}%") for interval, color, percent in zip(intervals, colors, percentages)]
    
    # Setting labels, title, legend, and showing the plot
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Number of Transactions")
    ax.set_title("Transactions per Time Interval")
    ax.legend(handles=legend_handles, loc="upper left")
    # ax.set_xticks(processed_df['date'].unique())  # Set x-ticks based on the 'date' column
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()