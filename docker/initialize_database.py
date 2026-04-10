#!/usr/bin/env python3
"""
Initialize TypeDB database with SEKOSA schema and data
"""
from typedb.driver import TypeDB, SessionType, TransactionType
import time
import sys

DATABASE_NAME = "sekosa"
SERVER_ADDRESS = "localhost:1729"

def wait_for_typedb(max_retries=30):
    """Wait for TypeDB server to be ready"""
    for i in range(max_retries):
        try:
            driver = TypeDB.core_driver(SERVER_ADDRESS)
            driver.close()
            print("TypeDB server is ready!")
            return True
        except Exception as e:
            print(f"Waiting for TypeDB... ({i+1}/{max_retries})")
            time.sleep(1)
    return False

def load_file(filepath):
    """Load TypeQL content from file"""
    with open(filepath, 'r') as f:
        return f.read()

def initialize_database():
    """Create database and load schema and data"""
    try:
        # Connect to TypeDB
        driver = TypeDB.core_driver(SERVER_ADDRESS)
        print(f"Connected to TypeDB at {SERVER_ADDRESS}")
        
        # Create database if it doesn't exist
        if driver.databases.contains(DATABASE_NAME):
            print(f"Database '{DATABASE_NAME}' already exists")
            driver.close()
            return
        
        print(f"Creating database '{DATABASE_NAME}'...")
        driver.databases.create(DATABASE_NAME)
        print("Database created successfully")
        
        # Load schema
        print("Loading schema...")
        schema_content = load_file('/app/SEKOSA_schema.tql')
        with driver.session(DATABASE_NAME, SessionType.SCHEMA) as session:
            with session.transaction(TransactionType.WRITE) as transaction:
                transaction.query.define(schema_content)
                transaction.commit()
        print("Schema loaded successfully")
        
        # Load data
        print("Loading data...")
        data_content = load_file('/app/SEKOSA_data.tql')
        with driver.session(DATABASE_NAME, SessionType.DATA) as session:
            with session.transaction(TransactionType.WRITE) as transaction:
                transaction.query.insert(data_content)
                transaction.commit()
        print("Data loaded successfully")
        
        driver.close()
        print("\nDatabase initialization complete!")
        
    except Exception as e:
        print(f"Error initializing database: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    if not wait_for_typedb():
        print("Failed to connect to TypeDB server")
        sys.exit(1)
    
    initialize_database()
