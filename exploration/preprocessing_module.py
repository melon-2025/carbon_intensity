import pandas as pd
import matplotlib.pyplot as plt

"""
A class for preprocessing data in a pandas DataFrame, with methods for dropping columns and rows, 
renaming columns, handling missing values, converting columns to datetime, and visualizing outliers.

Attributes:
    data (pd.DataFrame): The pandas DataFrame to be preprocessed.

Methods:
    drop(column_names=None, rows=None):
        Drops specified columns or rows from the DataFrame.

    rename_column(old_name, new_name):
        Renames a column in the DataFrame from an old name to a new name.

    check_missing_values():
        Checks and returns the count of missing values in each column of the DataFrame.

    data_overview():
        Prints an overview of the DataFrame, including its info, summary statistics, and missing values.

    convert_to_datetime(column_name):
        Converts a specified column to datetime format and sets it as the DataFrame index.

    box_plot_outlier(rows, columns):
        Creates box plots for all numeric columns in the DataFrame to visualize outliers.
"""


class Preprocess:
    """
    A class for preprocessing data in a pandas DataFrame, with multiple methods for data manipulation
    and visualization.
    """

    def __init__(self, data):
        """Initializes the Preprocess class.

        Parameters:
            data (pd.DataFrame): The pandas DataFrame to be preprocessed.

        """
        self.data = data

    def drop(self, column_names=None, rows=None):
        """
        Drops specified columns or rows from the DataFrame.

        Parameters:
            column_names (list of str, optional): List of column names to drop. Defaults to None.
            rows (int or list of int, optional): Index or list of row indices to drop. Defaults to None.

        Returns:
            None
        """
        if column_names is not None:
            self.data = self.data.drop(columns=column_names)
            print(f"{column_names} have been dropped")
        if rows is not None:
            if isinstance(rows, int):
                rows = [rows]
                self.data = self.data.drop(index=rows)
                print(f"{rows} have been dropped")

    def rename_column(self, old_name, new_name):
        """
        Renames a column in the DataFrame from an old name to a new name.

        Parameters:
            old_name (str): The current name of the column.
            new_name (str): The new name for the column.

        Returns:
            None
        """
        if old_name in self.data.columns:
            self.data.rename(columns={old_name: new_name}, inplace=True)
            print(f"{old_name} renamed to {new_name}")
        else:
            print(f"{old_name} not found")

    def check_missing_values(self):
        """
        Checks and returns the count of missing values in each column of the DataFrame.

        Returns:
            pd.Series: A Series where the index represents column names and values represent missing value counts.
        """
        return self.data.isnull().sum()

    def data_overview(self):
        """
        Prints an overview of the DataFrame, including:
        - Basic information about the DataFrame.
        - Summary statistics for numeric columns.
        - Missing values in each column.

        Returns:
            None
        """

        print("Basic data overview")
        self.data.info()
        print("\n")
        print("Summary Statistics")
        print(self.data.describe())
        print("\n")
        print("Missing values")
        print(self.check_missing_values())

    def convert_to_datetime(self, column_name):
        """
        Converts a specified column to datetime format and sets it as the DataFrame index.

        Parameters:
            column_name (str): The name of the column to convert to datetime.

        Returns:
            None
        """
        self.data[column_name] = pd.to_datetime(
            self.data[column_name], errors="coerce", dayfirst=True
        )
        self.data.set_index(column_name, inplace=True)

    def box_plot_outlier(self, rows, columns):
        """
        Creates box plots for all numeric columns in the DataFrame to visualize outliers.

        Parameters:
            rows (int): The number of rows in the subplot grid.
            columns (int): The number of columns in the subplot grid.

        Returns:
            None
        """
        # subploats
        fig, axes = plt.subplots(rows, columns, figsize=(12, rows * columns))
        axes = axes.flatten()  # Flatten for easy iteration
        for i, col in enumerate(self.data.columns):
            axes[i].boxplot(self.data[col], vert=False)
            axes[i].set_title(f"{col}")
            axes[i].set_xlabel(col)
        plt.tight_layout()
        plt.show()
