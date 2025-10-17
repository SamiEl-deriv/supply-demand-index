import psycopg2
import pandas as pd

# Put user_credentials below here (check your last pass "Database Credential")


conn = psycopg2.connect(
    sslmode = 'disable',
    **(user_credentials["chronicle"])
)


SQL = \
f"""SELECT * from chronicle limit 2;"""
print(SQL)
mydb = pd.read_sql(SQL, conn)
filename = f"chronicle.csv"
mydb.to_csv(filename)




