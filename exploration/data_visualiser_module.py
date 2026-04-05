import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

"""
A class for visualizing time series data using matplotlib, with functionalities 
to create dual-axis plots, identify outliers, and explore trends across 
different time granularities such as weekdays, hours, and months.

Attributes:
    data (pd.DataFrame): A pandas DataFrame containing the time series data.

Methods:
    dual_axis_plot(primary_columns, secondary_columns, ylabel1="Primary Y-axis", ylabel2="Secondary Y-axis"):
        Creates a dual-axis line plot for comparing two sets of columns with separate y-axes.

    outlier_plot(column_name, title, group_by=None):
        Identifies and plots outliers in the specified column using the interquartile range (IQR).

    plot_weekday_trends(columns=None, title="Average Trends by Day of the Week", ylabel="Value"):
        Plots average values for specified columns by day of the week, grouped by year.

    plot_hourly_trends(columns, title="Average Hourly Trends", ylabel="Value"):
        Plots average hourly trends for specified columns, grouped by year.

    plot_monthly_trends(columns, title="Average Monthly Trends", ylabel="Value"):
        Plots average monthly trends for specified columns, grouped by year.

    plot_month_with_weekends(column, year, month, title="Carbon Intensity for Month", ylabel="Value"):
        Plots data for a specific month, highlighting weekend values.
"""


class DataVisualiser:
    """
    A class for visualizing time series data using matplotlib, with functionalities
    to create dual-axis plots, identify outliers, and explore trends at various time granularities.
    """

    def __init__(self, data):
        """
        Initializes the DataVisualiser with a pandas DataFrame.

        Parameters:
            data (pd.DataFrame): A pandas DataFrame containing time series data with a DateTime index.
        """
        self.data = data

    def dual_axis_plot(
        self,
        primary_columns,
        secondary_columns,
        ylabel1="Primary Y-axis",
        ylabel2="Secondary Y-axis",
    ):
        """
        Creates a dual-axis line plot for comparing two sets of columns with separate y-axes.

        Parameters:
            primary_columns (list of str): Columns to be plotted on the primary y-axis.
            secondary_columns (list of str): Columns to be plotted on the secondary y-axis.
            ylabel1 (str): Label for the primary y-axis. Defaults to "Primary Y-axis".
            ylabel2 (str): Label for the secondary y-axis. Defaults to "Secondary Y-axis".
        """
        # Dual y-axes plot
        fig, ax1 = plt.subplots(figsize=(14, 6))
        primary_colors = ["blue", "green", "purple", "orange"]
        secondary_colors = ["red", "brown", "gray", "pink"]
        if primary_columns:
            for i, column in enumerate(primary_columns):
                # Plot Renewable Percentage on the first y-axis
                ax1.plot(
                    self.data.index,
                    self.data[column],
                    label=column,
                    color=primary_colors[i % len(primary_colors)],
                )
        ax1.set_ylabel(ylabel1)
        ax1.legend(loc="upper left")
        ax1.tick_params(axis="y")

        # Plot Carbon Intensity on the second y-axis
        if secondary_columns:
            ax2 = ax1.twinx()
            for i, column in enumerate(secondary_columns):
                ax2.plot(
                    self.data.index,
                    self.data[column],
                    label=column,
                    color=secondary_colors[i % len(secondary_colors)],
                )
        ax2.set_ylabel(ylabel2)
        ax2.legend(loc="upper right")
        ax2.tick_params(axis="y")

        # Add title and legend
        title = primary_columns if primary_columns else []
        title += secondary_columns if secondary_columns else []
        fig.suptitle(f"{','.join (title)} over time", fontsize=16)
        plt.tight_layout()
        plt.show()

    def outlier_plot(self, column_name, title, group_by=None):
        """
        Identifies and plots outliers in the specified column using the interquartile range (IQR).

        Parameters:
            column_name (str): The column to analyze for outliers.
            title (str): The title of the plot.
            group_by (str, optional): Time granularity for grouping ('month', 'week', 'day', 'hour').
        """
        if group_by:
            if group_by == "month":
                grouped_data = self.data.resample("ME").mean()
            elif group_by == "week":
                grouped_data = self.data.resample("W").mean()
            elif group_by == "day":
                grouped_data = self.data.resample("D").mean()
            elif group_by == "hour":
                grouped_data = self.data.resample("h").mean()
            else:
                raise ValueError(f"Unknown group_by option: {group_by}")
        else:
            grouped_data = self.data

        Q1 = grouped_data[column_name].quantile(0.25)  # First Quartile (25%)
        Q3 = grouped_data[column_name].quantile(0.75)  # Third Quartile (75%)
        IQR = Q3 - Q1  # Interquartile Range

        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Plot Renewable Percentage over time
        plt.figure(figsize=(14, 6))
        plt.plot(
            grouped_data.index, grouped_data[column_name], label=column_name, alpha=0.8
        )

        # Highlight outliers
        outliers = grouped_data[
            (grouped_data[column_name] < lower_bound)
            | (grouped_data[column_name] > upper_bound)
        ]
        plt.scatter(
            outliers.index,
            outliers[column_name],
            color="red",
            label="Outliers",
            alpha=0.6,
        )

        # Add upper and lower bounds
        plt.axhline(
            lower_bound,
            color="red",
            linestyle="--",
            label=f"Lower Bound ({lower_bound:.2f}%)",
        )
        plt.axhline(
            upper_bound,
            color="green",
            linestyle="--",
            label=f"Upper Bound ({upper_bound:.2f}%)",
        )

        # Add labels and title
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel(column_name)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_weekday_trends(
        self, columns=None, title="Average Trends by Day of the Week", ylabel="Value"
    ):
        """
        Plots average values for specified columns by day of the week, grouped by year.

        Parameters:
            columns (list of str, optional): Columns to plot. Defaults to all columns.
            title (str): Title of the plot. Defaults to "Average Trends by Day of the Week".
            ylabel (str): Label for the y-axis. Defaults to "Value".
        """

        if isinstance(columns, str):
            columns = [columns]
        elif columns is None:
            columns = self.data.columns

        # Extract Year and Weekday using .dt
        df_temp = self.data.copy()
        df_temp["Year"] = df_temp.index.year
        df_temp["Weekday"] = df_temp.index.day_name()

        # Order the weekdays
        weekday_order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        df_temp["Weekday"] = pd.Categorical(
            df_temp["Weekday"], categories=weekday_order, ordered=True
        )
        for column in columns:
            weekday_data = (
                df_temp.groupby(["Year", "Weekday"], observed=False)[column]
                .mean()
                .reset_index()
            )

            weekday_data["Weekday"] = pd.Categorical(
                weekday_data["Weekday"], categories=weekday_order, ordered=True
            )
            weekday_data = weekday_data.sort_values("Weekday")

            # Group by Year for plotting
            grouped = weekday_data.groupby("Year")

            # Plot the trends
            plt.figure(figsize=(14, 6))
            for year, group in grouped:
                plt.plot(group["Weekday"], group[column], label=str(year))

            # Add labels, title, and legend
            plt.title(f"{title} ({column})")
            plt.xlabel("Day of the Week")
            plt.ylabel(ylabel)
            plt.legend(title="Year")
            plt.tight_layout()
            plt.show()

    def plot_hourly_trends(
        self, columns, title="Average Hourly Trends", ylabel="Value"
    ):
        """
        Plots average hourly trends for specified columns, grouped by year.

        Parameters:
            columns (list of str): Columns to plot.
            title (str): Title of the plot. Defaults to "Average Hourly Trends".
            ylabel (str): Label for the y-axis. Defaults to "Value".
        """

        # Extract Hour and Year
        df_temp = self.data.copy()
        df_temp["Year"] = df_temp.index.year
        df_temp["Hour"] = df_temp.index.hour

        for column in columns:

            # Calculate averages for each hour grouped by year
            hourly_data = (
                df_temp.groupby(["Year", "Hour"], observed=False)[column]
                .mean()
                .reset_index()
            )

            # Group by Year for plotting
            grouped = hourly_data.groupby("Year")

            plt.figure(figsize=(14, 6))
            for year, group in grouped:
                plt.plot(group["Hour"], group[column], label=str(year))

            plt.title(title)
            plt.xlabel("Hour of the Day")
            plt.ylabel(ylabel)
            plt.xticks(range(24))
            plt.legend(title="Year")
            plt.tight_layout()
            plt.show()

    def plot_monthly_trends(
        self, columns, title="Average monthly trends", ylabel="Value"
    ):
        """
        Plots average monthly trends for specified columns, grouped by year.

        Parameters:
            columns (list of str): Columns to plot.
            title (str): Title of the plot. Defaults to "Average Monthly Trends".
            ylabel (str): Label for the y-axis. Defaults to "Value".
        """

        df_temp = self.data.copy()
        df_temp["Year"] = df_temp.index.year
        df_temp["Month"] = df_temp.index.month

        # Define the full range of months
        month_labels = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]

        for column in columns:
            # Calculate averages for each hour grouped by year
            monthly_data = (
                df_temp.groupby(["Year", "Month"], observed=False)[column]
                .mean()
                .reset_index()
            )

            # Group by Year for plotting
            grouped = monthly_data.groupby("Year")
            plt.figure(figsize=(14, 6))
            for year, group in grouped:
                plt.plot(group["Month"], group[column], label=str(year))

            plt.title(f"{title} ({column})")
            plt.xlabel("Month")
            plt.ylabel(ylabel)
            plt.xticks(range(1, 13), month_labels)
            plt.legend(title="Year")
            plt.tight_layout()
            plt.show()

    def plot_month_with_weekends(
        self, column, year, month, title="Carbon Intensity for Month", ylabel="Value"
    ):
        """
        Plot data for a specific month and highlight weekends.

        Args:
            column (str): The name of the column to plot.
            year (int): The year of the data to filter.
            month (int): The month of the data to filter.
            title (str): The title of the plot.
            ylabel (str): The label for the Y-axis.
        """

        # Filter data for the specified month
        month_str = f"{year}-{month:02d}"  # Format year and month as YYYY-MM
        data_month = self.data.loc[month_str]

        # Identify weekends (Saturday = 5, Sunday = 6)
        weekends = data_month[data_month.index.weekday >= 5]

        # Plot the main data for the month
        plt.figure(figsize=(12, 6))
        plt.plot(
            data_month.index,
            data_month[column],
            label="Weekday Data",
            color="blue",
            zorder=3,
        )

        # Highlight weekends
        plt.scatter(
            weekends.index, weekends[column], color="red", label="Weekend", zorder=5
        )

        # Format the x-axis for days
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d"))

        # Add labels, title, and legend
        plt.title(f"{title} ({year}-{month:02d})")
        plt.xlabel("Day of the Month")
        plt.ylabel(ylabel)
        plt.legend()
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        plt.show()
