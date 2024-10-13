import sqlite3
import json

# 连接到 SQLite 数据库
conn = sqlite3.connect('test.db')
cursor = conn.cursor()

# 定义一个函数来打印表中的所有数据
def print_table_data(table_name):
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()
    print(f"Data in table '{table_name}':")
    for row in rows:
        print(row)
    print()  # 打印空行以分隔不同的表数据

# 定义一个字典，将表名映射到包含 JSON 数据的列名
json_columns = {
    "study_user_attributes": "value_json",
    "study_system_attributes": "value_json",
    "trial_user_attributes": "value_json",
    "trial_system_attributes": "value_json",
    "trial_params": "distribution_json"
}

# 遍历所有表并打印数据
tables = [
    "studies", "version_info", "study_directions",
    "study_user_attributes", "study_system_attributes",
    "trials", "trial_user_attributes", "trial_system_attributes",
    "trial_params", "trial_values", "trial_intermediate_values",
    "trial_heartbeats", "alembic_version"
]

for table in tables:
    print_table_data(table)
    if table in json_columns:
        column_name = json_columns[table]
        print(f"JSON data in column '{column_name}' of table '{table}':")
        cursor.execute(f"SELECT {column_name} FROM {table}")
        rows = cursor.fetchall()
        for row in rows:
            try:
                print(json.loads(row[0]))
            except json.JSONDecodeError:
                print("Not a valid JSON:", row[0])
        print()  # 打印空行以分隔不同的表数据

# 关闭 Cursor 和 Connection
cursor.close()
conn.close()