import pandas as pd

from db_connection import engine
from queries import WELL_INTELLIGENCE_QUERY


print("Running Industrial Well Intelligence query...")


try:

    df = pd.read_sql(
        WELL_INTELLIGENCE_QUERY,
        engine
    )

    print()
    print("Query executed successfully!")
    print()

    print(df.to_string(index=False))

    print()
    print(f"Number of wells: {len(df)}")

except Exception as e:

    print("Query failed.")
    print(e)