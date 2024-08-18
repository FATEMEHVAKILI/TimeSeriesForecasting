'''The code is developed based on SOLID principles. 
Its purpose is to convert the data file to CSV 
and extract the necessary data for the first case.'''

import pandas as pd
from abc import ABC, abstractmethod


class IDataProcessor(ABC):
    @abstractmethod
    def save_as_csv(self, csv_file):
        pass


class IIndicatorFilter(ABC):
    @abstractmethod
    def filter_by_indicator(self, indicator_name):
        pass


class ExcelProcessor(IDataProcessor):
    def __init__(self, excel_file):
        self.excel_file = excel_file
        self.df = pd.read_excel(excel_file)

    def save_as_csv(self, csv_file):
        self.df.to_csv(csv_file, index=False)
        print(
            f"Excel file '{self.excel_file}' has been converted to CSV file '{csv_file}'.")


class IndicatorFilter(IIndicatorFilter):
    def __init__(self, df):
        self.df = df

    def filter_by_indicator(self, indicator_name):
        filtered_row = self.df[self.df['Indicator Name'] == indicator_name]
        years = filtered_row.columns[2:]
        values = filtered_row.iloc[0, 2:]
        return pd.DataFrame({
            'Year': years,
            indicator_name: values
        })

    def save_filtered_data(self, filtered_df, csv_file):
        filtered_df.to_csv(csv_file, index=False)
        print(f"Filtered data has been saved to CSV file '{csv_file}'.")


# Using the classes
excel_processor = ExcelProcessor('data.xlsx')
excel_processor.save_as_csv('data.csv')

indicator_filter = IndicatorFilter(excel_processor.df)

# Extract indicators and save them
for indicator in [
    'Exports of goods and services (constant 2015 US$)',
    'Agriculture, forestry, and fishing, value added (% of GDP)',
    'GDP per capita (current US$)',
    'Gross capital formation (constant LCU)',
    'Inflation, consumer prices (annual %)'
]:
    filtered_df = indicator_filter.filter_by_indicator(indicator)
    indicator_filter.save_filtered_data(filtered_df, f"{indicator}.csv")
