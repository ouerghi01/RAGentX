import pandas as pd
from cassandra_service import CassandraManager

session = CassandraManager().get_session()
query = "SELECT table_name FROM system_schema.tables WHERE keyspace_name='shop'"
rows = session.execute(query)

excel_writer = pd.ExcelWriter('cassandra_data.xlsx', engine='openpyxl')

for row in rows:
    table_name = row.table_name
    query = f"SELECT * FROM shop.{table_name}"
    rows = session.execute(query)
    columns = rows.column_names
    data = [list(row) for row in rows]
    df = pd.DataFrame(data, columns=columns)
    df.to_excel(excel_writer, sheet_name=table_name, index=False)

excel_writer.close()

print("Data saved to cassandra_data.xlsx")
