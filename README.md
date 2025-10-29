# DBF to SQL Server Import Tool

Import your DBF (dBase) files into SQL Server.

```bash
$ python dbf_to_sqlserver.py -h
usage: dbf_to_sqlserver.py [-h] [--credentials CREDENTIALS] [--target-db-id TARGET_DB_ID] [--dbf-dir DBF_DIR] [--if-exists {fail,replace,append}] [--chunk-size CHUNK_SIZE] [--schema SCHEMA]
                           [--driver {auto,pyodbc,pymssql}] [--encoding ENCODING]

Import DBF files to SQL Server

options:
  -h, --help            show this help message and exit
  --credentials, -c CREDENTIALS
                        Path to data source credentials JSON file (default: db_credentials.json)
  --target-db-id, --tdb-id TARGET_DB_ID
                        Target database identifier in credentials file (default: db_01_ms)
  --dbf-dir, -d DBF_DIR
                        Directory containing DBF files
  --if-exists, -i {fail,replace,append}
                        Action when table exists (default: append)
  --chunk-size CHUNK_SIZE
                        Chunk size for data loading (default: 10000)
  --schema, -s SCHEMA   Database schema for tables (default: dbo)
  --driver {auto,pyodbc,pymssql}
                        SQL Server driver to use (default: pymssql)
  --encoding, -e ENCODING
                        Character encoding for DBF files (e.g., cp1252, cp850, iso-8859-15, iso-8859-1, latin1). If not specified, auto-detection will try common encodings. Use this to override when
                        auto-detection produces garbled text.
```

## Quick Setup

1. **Create a virtual env**:

On Linux:

```bash
python3.11 -m venv dbf311
source ./dbf311/bin/activate
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Create your credentials file** (`db_credentials.json`):

```json
{
   "db_01_ms": {
       "db_type": "mssql",
       "auth_mode": "classic",
       "info": {
           "username": "your_username",
           "password": "your_password",
           "server": "localhost",
           "port": 1433,
           "database": "your_database"
       }
   }
}
```

   You can add multiple databases to the same file - just use different identifiers like `db_02_ms`, `db_03_ms`, etc.

4. **Run your first import**:

```bash
python dbf_to_sqlserver.py \
   --credentials. ./path/to/your/db_credentials.json \
   --target-db-id db_01_ms \
   --dbf-dir /path/to/your/dbf/files \
   --schema toto
```

The tool will create one table per DBF file and track everything in a `dbf_ingestion_log` table.

## CLI Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--credentials` | `-c` | Path to credentials JSON file | `db_credentials.json` |
| `--target-db-id` | `--tdb-id` | Target database ID from credentials | `db_01_ms` |
| `--dbf-dir` | `-d` | Directory with DBF files | Required |
| `--if-exists` | `-i` | What to do if table exists: `fail`, `replace`, or `append` | `append` |
| `--schema` | `-s` | Database schema for tables | `dbo` |
| `--chunk-size` | | Rows to load at once | `10000` |
| `--driver` | | SQL Server driver: `auto`, `pyodbc`, or `pymssql` | `pymssql` |
| `--encoding` | `-e` | Force specific encoding (e.g., `cp850`, `cp1252`) | Auto-detect |

## Import Modes

- **append** (default): Adds data to existing tables. Smart enough to handle column differences.
- **replace**: Drops and recreates tables. Use when you want a fresh start.
- **fail**: Stops if tables already exist. Use when you want to be cautious.

## What Gets Created

### Your Data Tables
- One table per DBF file (named after the file, lowercase)
- Each row includes a `run_id` column tracking which import run loaded it
- Each row includes a `meta_source` column with the source DBF file path

### Tracking Table (`dbf_ingestion_log`)
Every import is logged here with:
- Which files were processed
- How many rows loaded
- Success/failure status
- Any error messages
- Timestamp of the import

**Status values:**
- `SUCCESS`: Everything worked perfectly
- `INCOMPLETE`: Processed but with warnings (e.g., missing memo file)
- `FAILED`: Something went wrong

## Find Your Data Later

Each import run gets a unique ID. Use it to query your data:

```sql
-- See all your import runs
SELECT
    run_id,
    COUNT(*) as files_processed,
    SUM(rows_loaded) as total_rows,
    MIN(ingestion_timestamp) as started_at
FROM dbo.dbf_ingestion_log
GROUP BY run_id
ORDER BY started_at DESC;

-- Get data from a specific import
SELECT * FROM dbo.your_table
WHERE run_id = 'your-uuid-here';

-- Find failed imports
SELECT * FROM dbo.dbf_ingestion_log
WHERE success = 'FAILED';
```

## Logging

The tool keeps you informed with:
- **Console output**: Color-coded messages showing progress
- **Daily log files**: `dbf_import_YYYY-MM-DD.log` with full details


## Troubleshooting

**Garbled text/strange characters?**
Try specifying the encoding: `-e cp850` or `-e cp1252`

**Using ODBC drivers?**
Use pyodbc driver: `--driver pyodbc` (ODBC needed)


## Build a binary with nuitka

### Linux

```bash
./build-linux-pip.sh
```
### Windows

```bash
.\build-windows-pip.bat
```