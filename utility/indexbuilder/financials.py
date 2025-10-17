from data import Data
import psycopg2
import pandas as pd
import tqdm
from datetime import datetime


class Feed(Data):

    # Credentials (private)
    __host = 'feeddb10-qa.regentmarkets.com'
    __port = '5433'
    __dbname = 'feed'
    __user = 'quants'
    __password = 'c2Ju%qnwgSPmEs!g'

    __symbol = {"EURUSD": "frxEURUSD"}

    # From start date including end date
    def __init__(self, start_date: str, end_date: str, symbol: str) -> None:
        super().__init__(start_date, end_date)
        self.start_date = start_date
        self.end_date = end_date
        # if start_date and end_date are equal,
        self.symbol = symbol

    def get_table_list(self):
        # Calculate the no. of months and hence no. of tables
        n_tables = (int(self.end_date.year) - int(self.start_date.year)) * \
            12 + int(self.end_date.month) - int(self.start_date.month) + 1

        table_list = []
        year = int(self.start_date.year)
        month = int(self.start_date.month)
        for _ in range(n_tables):
            if (month > 12):
                year = year + 1
                month = 1
            table_list.append("tick_" + str(year) + "_" + str(month))
            month = month + 1
        print("No. of tables: ", n_tables)
        print("Table query list: ", table_list)
        return table_list

    def get_data(self, save_csv=False):
        table_list = self.get_table_list()
        x = pd.DataFrame([])
        for table in table_list:
            # Establish connection
            conn = psycopg2.connect(
                host=self.__host,
                user=self.__user,
                password=self.__password,
                dbname=self.__dbname,
                port=self.__port,
                sslmode='disable'
            )
            print("Connected to Database!")

            with conn.cursor(name='custom_cursor') as cur:
                cur.itersize = 100000  # chunk size
                sql_command = "SELECT * FROM feed.{} WHERE underlying = '{}' AND ts >= '{}'::TIMESTAMP AND ts < '{}'::TIMESTAMP;"\
                    .format(str(table), self.symbol, self.start_date, self.end_date + pd.Timedelta(1, unit="d"))
                data = pd.read_sql(sql_command, conn)
                print(data)

                x = pd.concat([x, pd.DataFrame(data)],
                              ignore_index=True, axis=0)

                print('Data for {}_{} Loaded Successfully'.format(
                    str(table), self.symbol))

                if save_csv:
                    print("Saving as csv ....")
                    data.to_csv('Data_{}_{}.csv'.format(
                        str(table), self.symbol), index=None, header=True)
                    print("Saved as csv!")

                cur.close()
                conn.close()
                print('Connection Closed!')
        x = x.reset_index(drop=True)
        return x


start_date = datetime.strptime("2022-10-01", "%Y-%m-%d")
end_date = datetime.strptime("2022-10-25", "%Y-%m-%d")

feed = Feed(start_date=start_date, end_date=end_date,
            symbol="frxUSDJPY").get_data(save_csv=True)

print(feed)
