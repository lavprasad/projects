import sqlite3

# Connect to the database
conn = sqlite3.connect("predictions2.db")
c = conn.cursor()

# Create the table if it does not exist
c.execute('''CREATE TABLE IF NOT EXISTS predictions2
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              fruit TEXT,
              count INTEGER,
              expiry_date DATE)''')

# Example data
fruit = "Apple"
count = 10
expiry_date = "2024-05-05"
fruit = "Banana"
count = 6
expiry_date = "2024-05-12"

fruit = "Peas"
count = 4
expiry_date = "2024-05-20"

fruit = "Tomato"
count = 11
expiry_date = "2024-05-20"
# Format: "YYYY-MM-DD"

# Insert data into the table
c.execute("INSERT INTO predictions2 (fruit, count, expiry_date) VALUES (?, ?, ?)", (fruit, count, expiry_date))

# Commit changes and close the connection
conn.commit()
conn.close()
