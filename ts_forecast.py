import pandas
import matplotlib.pyplot as plt
from fbprophet import Prophet
import numpy as np

# TODO: Write a test for the forecast method, data visualization, etc
# NOTE: Work in progress, NOT FULLY OPERATIONAL YET! 

class DemandForecaster:
    """ A simple tool to load, clean, and forecast timeseries datasets.
    """

    def __init__(self, filename):
        self.filename = filename

    def loadData(self):

        # Convert Timestamp column to index col and transform to datetime
        df = pandas.read_csv(self.filename)

        # to_datetime is way faster if we specify the date format of the string
        # column instead of letting it "guess"
        date_format = '%m/%d/%Y %H:%M:%S'
        df['Timestamp'] = pandas.to_datetime(df['Timestamp'],
                                             format=date_format)

        df.set_index('Timestamp', inplace=True)

        # Re-index data to every 5 minutes (remove repeated values), impute
        # the nearest value
        df = df.reindex(pandas.date_range(start=df.index[0],
                                          end=df.index[-1],
                                          freq='5T'),
                        method='nearest')
        # Fill original missing data with the previous value
        df = df.fillna(method='ffill')
        self.data = df

        df['y'] = df['Load']
        df['ds'] = df.index

        df.drop(['Load'], axis='columns', inplace=True)


        return df

    def getForecast(self, timestamp, N, H):
        # Compute the forecast H time steps ahead of the value at timestamp
        ts = self.data.loc[:timestamp]
        m = Prophet()
        train = self.data.loc[:timestamp]
        future = self.data.loc[self.data.loc[timestamp:].index[:H]]

        m.fit(train)

        fcast = m.predict(future)['yhat'].values[-1]


        return fcast

    def forecast(self, N, H):
        """Forecast the value of a time series
        INPUT:
            data: Pandas dataframe containing a Datetime index and a data column
            N: Number of training data points
            H: Forecast horizon in terms of time steps

        Returns:
            df_forecast: Dataframe containing the forecast value H steps ahead
                         each given data pointgiven data
        """

        df = self.data


        fcast_output = np.array([self.getForecast(tstamp, N, H) for
                                 tstamp in df.index[N:]])

        dummies = np.array(['' for k in range(N)])

        self.fcastdata = np.concatenate([dummies, fcast_output])

        return fcast_output


    def writeOutput(self, dataoutfile):

        tstamps = self.data.index.astype(str)
        x = self.data['y'].values.astype(str)
        xpred = self.fcastdata.astype(str)

        with open(dataoutfile, 'w') as f:

            lines = [','.join(row) + '\n' for row in zip(tstamps, x, xpred)]

            f.writelines(lines)
