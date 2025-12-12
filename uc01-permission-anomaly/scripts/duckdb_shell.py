"""
Interactive DuckDB Shell for Parquet Data

Auto-discovers parquet files in data/ directory and registers them as views.
Usage:
    uv run python scripts/duckdb_shell.py
    uv run python scripts/duckdb_shell.py --query "SELECT * FROM anomaly_scores LIMIT 5"
"""

import duckdb
import glob
import os
import argparse
import sys
from pathlib import Path
import pandas as pd

# Project root setup
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

def get_parquet_files():
    """Find all parquet files in data directory."""
    patterns = [
        str(DATA_DIR / "**" / "*.parquet"),
    ]
    files = []
    for pattern in patterns:
        # Recursive glob requires ** and recursive=True
        files.extend(glob.glob(pattern, recursive=True))
    return files

def register_views(con, files):
    """Register parquet files as views in DuckDB."""
    print(f"Scanning {DATA_DIR} for parquet files...")
    registered = []
    
    for file_path in files:
        path = Path(file_path)
        # Create table name from filename (minus extension)
        table_name = path.stem
        
        # Handle duplicates by appending folder name if needed
        if table_name in registered:
            table_name = f"{path.parent.name}_{table_name}"
            
        try:
            rel_path = os.path.relpath(path, os.getcwd())
            con.execute(f"CREATE OR REPLACE VIEW {table_name} AS SELECT * FROM read_parquet('{rel_path}')")
            print(f"  ✓ {table_name} -> {rel_path}")
            registered.append(table_name)
        except Exception as e:
            print(f"  ✗ Failed to register {path.name}: {e}")
            
    return registered

def main():
    parser = argparse.ArgumentParser(description="DuckDB Shell for Parquet Data")
    parser.add_argument("--query", "-q", help="Run specific SQL query and exit")
    args = parser.parse_args()
    
    # Initialize in-memory database
    con = duckdb.connect(database=':memory:')
    
    # Register views
    files = get_parquet_files()
    if not files:
        print(f"No parquet files found in {DATA_DIR}")
        return
        
    tables = register_views(con, files)
    print(f"\n{len(tables)} views registered. Ready for queries!\n")
    
    # Execute query if provided
    if args.query:
        print(f"Running: {args.query}")
        print("-" * 50)
        try:
            result = con.execute(args.query).df()
            with pd.option_context('display.max_columns', None, 'display.width', 1000):
                print(result)
        except Exception as e:
            print(f"Error: {e}")
        return

    # Interactive shell
    print("Entering interactive mode. Type 'exit' to quit.")
    print("Example: SELECT * FROM anomaly_scores LIMIT 5;")
    
    while True:
        try:
            # Simple REPL
            sys.stdout.write("duckdb> ")
            sys.stdout.flush()
            query = sys.stdin.readline().strip()
            
            if not query:
                continue
                
            if query.lower() in ('exit', 'quit', 'q'):
                break
                
            result = con.execute(query).df()
            with pd.option_context('display.max_columns', None, 'display.width', 1000):
                print(result)
            print()
            
        except KeyboardInterrupt:
            print("\nType 'exit' to quit")
            continue
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
