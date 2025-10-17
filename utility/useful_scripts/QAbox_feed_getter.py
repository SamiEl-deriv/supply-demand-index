import psycopg2
import pandas as pd

# Put user_credentials below here
user_credentials = {
"feed":
	{
	"host":"feeddb10-qa.regentmarkets.com",
	"port":"5432",
	"dbname":"feed",
	"user":"",
	"password":'',
    },
"chronicle":
	{"host":"chronicledb-qa.regentmarkets.com",
	"port":"5432",
	"dbname":"chronicle",
	"user":"",
	"password":'',
    }
}

underlyings = ['CRASH300N','CRASH500','CRASH1000','BOOM300N','BOOM500','BOOM1000']

conn = psycopg2.connect(
    sslmode = 'disable',
    **(user_credentials["feed"])
)

for underlying in underlyings:
	SQL = \
f"""SELECT underlying, ts, spot
FROM feed.tick
WHERE underlying IN ('{underlying}') 
AND ts >= '2024-08-01' 
AND ts < '2024-09-01' 
ORDER BY ts, underlying;"""
	print(SQL)
	mydb = pd.read_sql(SQL, conn)
	filename = f"{underlying}_cb.csv"
	mydb.to_csv(filename)





