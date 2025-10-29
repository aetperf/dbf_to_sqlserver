#!/usr/bin/env python
"""
DBF to SQL Server import script
Loads DBF files into SQL Server tables with comprehensive logging and error handling
"""
# pylint: disable=too-many-lines

import argparse
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from urllib.parse import quote_plus

import pandas as pd
import sqlalchemy as sa
from dbfread import DBF
from loguru import logger
from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    text,
)
from sqlalchemy.types import (
    BIGINT,
    DECIMAL,
    FLOAT,
    INTEGER,
    NVARCHAR,
    VARCHAR,
    DATE,
    DATETIME,
)
from sqlalchemy.dialects.mssql import BIT


# DBF field type to SQL Server type mapping
DBF_TO_SQLSERVER_TYPE_MAP = {
    "C": VARCHAR,  # Character/String field
    "N": DECIMAL,  # Numeric field (use DECIMAL for precision)
    "F": FLOAT,  # Float field
    "L": BIT,  # Logical/Boolean field
    "D": DATE,  # Date field
    "T": DATETIME,  # DateTime field (FoxPro extension)
    "I": INTEGER,  # Integer field (FoxPro extension)
    "B": BIGINT,  # BigInt field (FoxPro extension)
    "M": NVARCHAR,  # Memo field (use NVARCHAR for Unicode support)
    "G": NVARCHAR,  # General/OLE field (treat as text)
    "P": NVARCHAR,  # Picture field (treat as text)
    "Y": DECIMAL,  # Currency field (use DECIMAL)
}


def get_dbf_field_info(dbf_file_path: Path) -> Dict[str, Dict[str, Any]]:
    """Extract field information from DBF file header

    Returns:
        Dictionary mapping field names to their type information
        Format: {field_name: {'type': 'C', 'length': 50, 'decimal': 0}}
    """
    try:
        dbf_reader = DBF(dbf_file_path, load=False)
        field_info = {}

        for field in dbf_reader.fields:
            field_info[field.name] = {
                "type": field.type,
                "length": field.length,
                "decimal": field.decimal_count,
            }

        logger.debug(f"Extracted field info from {dbf_file_path}: {field_info}")
        return field_info

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(f"Failed to extract field info from {dbf_file_path}: {e}")
        return {}


def create_pandas_dtype_map(field_info: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Create pandas dtype mapping from DBF field information

    Args:
        field_info: Field information from get_dbf_field_info()

    Returns:
        Dictionary suitable for pandas to_sql dtype parameter
    """
    dtype_map = {}

    for field_name, info in field_info.items():
        field_type = info["type"]
        length = info["length"]
        decimal = info["decimal"]

        # Use the mapping dictionary as base
        if field_type in DBF_TO_SQLSERVER_TYPE_MAP:
            base_type = DBF_TO_SQLSERVER_TYPE_MAP[field_type]

            # Apply length/precision customization for specific types
            if field_type == "C":  # Character/String
                dtype_map[field_name] = base_type(length if length > 0 else 255)
            elif field_type == "N":  # Numeric
                if decimal > 0:
                    # Decimal field with precision
                    dtype_map[field_name] = DECIMAL(precision=length, scale=decimal)
                else:
                    # Integer field
                    if length <= 9:
                        dtype_map[field_name] = INTEGER()
                    else:
                        dtype_map[field_name] = BIGINT()
            elif field_type == "M":  # Memo
                dtype_map[field_name] = base_type(length="max")
            elif field_type == "Y":  # Currency
                dtype_map[field_name] = DECIMAL(precision=19, scale=4)
            else:
                # Use base type as-is for other types
                dtype_map[field_name] = base_type()
        else:
            # Default fallback for unknown types
            logger.warning(
                f"Unknown DBF field type '{field_type}' for field "
                f"'{field_name}', using VARCHAR(255)"
            )
            dtype_map[field_name] = VARCHAR(255)

    # Add run_id column type
    dtype_map["run_id"] = VARCHAR(36)

    # Add meta_source column type for tracking DBF file path
    dtype_map["meta_source"] = VARCHAR(500)

    logger.debug(f"Created pandas dtype map: {dtype_map}")
    return dtype_map


def setup_logging():
    """Configure loguru logger"""
    log_format = (
        "[<g>{time:YYYY-MM-DD HH:mm:ss.SSSZ}</g> :: <c>{level}</c> ::"
        + " <e>{process.id}</e> :: <y>{process.name}</y>] {message}"
    )

    logger.remove()
    logger.add(sys.stdout, format=log_format, level="INFO")
    logger.add(
        "dbf_import_{time:YYYY-MM-DD}.log",
        format=(
            "[{time:YYYY-MM-DD HH:mm:ss.SSSZ} :: {level} :: "
            "{process.id} :: {process.name}] {message}"
        ),
        level="DEBUG"
    )


def load_credentials(credentials_path: str) -> Dict[str, Any]:
    """Load database credentials from JSON file"""
    try:
        with open(credentials_path, "r", encoding="utf-8") as f:
            credentials = json.load(f)
        logger.info(f"Loaded credentials from {credentials_path}")
        return credentials
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(f"Failed to load credentials from {credentials_path}: {e}")
        raise


def create_mssql_engine(  # pylint: disable=too-many-locals
    creds: Dict[str, Any], driver: str = "auto", target_db_id: str = "db_01_ms"
) -> sa.Engine:
    """Create SQL Server engine with enhanced connection handling"""
    start_time = time.perf_counter()

    target_config = creds.get(target_db_id, {})
    if not target_config:
        available_db = list(creds.keys())
        raise ValueError(
            f"Target database '{target_db_id}' not found in credentials. Available: {available_db}"
        )
    assert target_config["db_type"] == "mssql"
    assert target_config["auth_mode"] == "classic"
    info = target_config.get("info", {})

    username = info.get("username")
    password = info.get("password")
    server = info.get("server")
    port = info.get("port", 1433)
    database = info.get("database")

    logger.info(f"ms_username = '{username}'")
    logger.info(f"ms_server = '{server}'")
    logger.info(f"ms_port = '{port}'")
    logger.info(f"ms_database = '{database}'")

    if driver == "auto":
        # Default to pymssql (more reliable on Linux), fall back to pyodbc
        try:
            import pymssql  # pylint: disable=import-outside-toplevel,unused-import

            driver = "pymssql"
            logger.info("Auto-detected driver: pymssql (default)")
        except ImportError:
            try:
                import pyodbc  # pylint: disable=import-outside-toplevel,unused-import

                driver = "pyodbc"
                logger.info("Auto-detected driver: pyodbc (fallback)")
            except ImportError as exc:
                raise ImportError("Neither pymssql nor pyodbc is available") from exc

    if driver == "pyodbc":
        connection_string = (
            f"mssql+pyodbc://{quote_plus(username)}:{quote_plus(password)}"
            f"@{server}:{port}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
        )
        engine = create_engine(
            connection_string,
            pool_size=10,
            max_overflow=5,
            pool_recycle=3600,
            pool_pre_ping=True,
        )
    elif driver == "pymssql":
        connection_string = (
            f"mssql+pymssql://{quote_plus(username)}:{quote_plus(password)}"
            f"@{server}:{port}/{database}"
        )
        engine = create_engine(
            connection_string,
            isolation_level="AUTOCOMMIT",
            pool_size=10,
            max_overflow=5,
            pool_recycle=3600,
            pool_pre_ping=True,
        )
    else:
        raise ValueError(f"Unsupported driver: {driver}")

    # Test the connection
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            data = result.fetchone()[0]
            assert data == 1
            logger.info(f"{driver} sqlalchemy engine verified")
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(f"Failed to verify connection with {driver}: {e}")
        raise

    elapsed_time = time.perf_counter() - start_time
    logger.info(f"create_mssql_engine - Elapsed time (s): {elapsed_time:.3f}")

    return engine


def create_ingestion_table(engine: sa.Engine, schema: str = "dbo"):
    """Create ingestion tracking table if it doesn't exist"""
    metadata = MetaData()
    Table(
        "dbf_ingestion_log",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("run_id", String(36), nullable=False),
        Column("file_path", String(500), nullable=False),
        Column("file_name", String(255), nullable=False),
        Column("target_table", String(255), nullable=False),
        Column("rows_loaded", Integer),
        Column("ingestion_timestamp", DateTime, default=datetime.now),
        Column("success", String(10)),
        Column("error_message", String(1000)),
        schema=schema,
    )

    try:
        metadata.create_all(engine)
        logger.info("Ingestion tracking table created/verified")
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(f"Failed to create ingestion table: {e}")


def _create_target_schema_if_not_exist(engine: sa.Engine, schema_name: str) -> None:
    """
    Create target schema if it doesn't exist.

    Creates the specified schema in the target database if it doesn't already exist.

    Parameters
    ----------
    engine : sa.Engine
        SQLAlchemy engine for database connection
    schema_name : str
        Schema name to create if it doesn't exist
    """
    if (
        schema_name and schema_name.lower() != "dbo"
    ):  # Skip empty strings and default dbo schema
        sql = f"""IF (NOT EXISTS (SELECT * FROM sys.schemas WHERE name = '{schema_name}'))
        BEGIN
            EXEC ('CREATE SCHEMA [{schema_name}] AUTHORIZATION [dbo]')
        END"""
        logger.debug(sql)
        logger.info(f"Creating target schema '{schema_name}' if not exists")

        try:
            with engine.connect() as conn:
                conn.execute(text(sql))
                conn.commit()
            logger.info(
                f"Schema '{schema_name}' created successfully or already exists"
            )
        except Exception as e:
            logger.error(f"Failed to create schema '{schema_name}': {e}")
            raise


def log_ingestion(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    engine: sa.Engine,
    file_path: str,
    file_name: str,
    target_table: str,
    rows_loaded: int,
    success: bool,
    run_id: str,
    error_message: str = None,
    schema: str = "dbo",
):
    """Log ingestion attempt to tracking table"""
    try:
        with engine.connect() as conn:
            conn.execute(
                text(
                    f"""
                INSERT INTO {schema}.dbf_ingestion_log
                (run_id, file_path, file_name, target_table, rows_loaded, ingestion_timestamp, success, error_message)
                VALUES (:run_id, :file_path, :file_name, :target_table, :rows_loaded, :timestamp, :success, :error_message)
                """
                ),
                {
                    "run_id": run_id,
                    "file_path": str(file_path),
                    "file_name": file_name,
                    "target_table": target_table,
                    "rows_loaded": rows_loaded,
                    "timestamp": datetime.now(),
                    "success": (
                        success
                        if isinstance(success, str)
                        else ("SUCCESS" if success else "FAILED")
                    ),
                    "error_message": error_message[:1000] if error_message else None,
                },
            )
            conn.commit()
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(f"Failed to log ingestion for {file_name}: {e}")


def validate_and_prepare_schema(
    engine: sa.Engine, table_name: str, df: pd.DataFrame, schema: str = "dbo"
) -> pd.DataFrame:
    """Validate schema and prepare DataFrame for flexible appending

    Returns:
        Modified DataFrame that matches the existing table schema
        - Missing columns in DataFrame are added with NULL values
        - Extra columns in DataFrame are kept (table will be extended)
    """
    try:
        with engine.connect() as conn:
            # Get existing table columns
            result = conn.execute(
                text(
                    f"""
                SELECT COLUMN_NAME, DATA_TYPE 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_NAME = '{table_name}' AND TABLE_SCHEMA = '{schema}'
                ORDER BY ORDINAL_POSITION
            """
                )
            )
            existing_columns = {row[0].lower(): row[1] for row in result}

            # Get column sets
            df_columns = set(df.columns.str.lower())
            table_columns = set(existing_columns.keys())

            # Find missing and extra columns
            missing_in_df = table_columns - df_columns
            extra_in_df = df_columns - table_columns

            # Log schema differences
            if missing_in_df:
                logger.warning(
                    f"Table {table_name}: Missing columns in new data "
                    f"will be filled with NULL: {missing_in_df}"
                )
            if extra_in_df:
                logger.warning(
                    f"Table {table_name}: Extra columns in new data "
                    f"will extend the table: {extra_in_df}"
                )

            # Create a copy of the DataFrame to modify
            df_modified = df.copy()

            # Add missing columns with NULL values
            for col in missing_in_df:
                # Find original case column name from existing table
                original_col = next(
                    k for k in existing_columns.keys() if k.lower() == col
                )
                df_modified[original_col] = None
                logger.debug(f"Added missing column '{original_col}' with NULL values")

            logger.info(f"Schema preparation completed for table {table_name}")
            return df_modified

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(f"Failed to prepare schema for table {table_name}: {e}")
        raise


def table_exists(engine: sa.Engine, table_name: str, schema: str = "dbo") -> bool:
    """Check if table exists in database"""
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    f"""
                SELECT COUNT(*) 
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_NAME = '{table_name}' AND TABLE_SCHEMA = '{schema}'
            """
                )
            )
            return result.scalar() > 0
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(f"Failed to check if table {table_name} exists: {e}")
        return False


def process_dbf_file(  # pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments,too-many-return-statements,too-many-branches,too-many-statements
    engine: sa.Engine,
    dbf_file: Path,
    if_exists: str,
    run_id: str,
    chunk_size: int = 10000,
    schema: str = "dbo",
    encoding: str = None,
) -> bool:
    """Process a single DBF file and load it into SQL Server"""
    table_name = dbf_file.stem.lower()
    logger.info(f"Processing {dbf_file} -> table {table_name}")

    try:
        # Read DBF file with encoding handling
        logger.info(f"Reading DBF file: {dbf_file}")

        df = None
        memo_file_missing = False
        detected_encoding = None

        # If user specified an encoding, use it directly
        if encoding is not None:
            logger.info(f"Using user-specified encoding: {encoding}")
            try:
                dbf_reader = DBF(
                    dbf_file,
                    encoding=encoding,
                    load=True,
                    ignore_missing_memofile=True,
                    lowernames=False,  # Preserve field name case
                    char_decode_errors="replace",  # Replace invalid chars instead of failing
                    raw=True,  # Read raw values without parsing to avoid type conversion errors
                )
                df = pd.DataFrame(iter(dbf_reader))
                detected_encoding = encoding
                logger.info(f"Successfully read {dbf_file} with encoding: {encoding}")
                if hasattr(dbf_reader, "_memo_file_missing") or not hasattr(
                    dbf_reader, "memo"
                ):
                    logger.warning(
                        f"Memo file missing for {dbf_file} - memo fields may be incomplete"
                    )
                    memo_file_missing = True
            except Exception as exc:
                raise ValueError(
                    f"Could not read {dbf_file} with encoding '{encoding}'. "
                    f"Try running without --encoding to use auto-detection."
                ) from exc
        else:
            # Auto-detect encoding: try multiple encodings for DBF files
            # (French text often uses cp850 or cp1252)
            logger.info("Auto-detecting encoding...")
            encodings_to_try = [
                "cp1252",
                "cp850",
                "iso-8859-15",
                "iso-8859-1",
                "latin1",
            ]

            for enc in encodings_to_try:
                try:
                    dbf_reader = DBF(
                        dbf_file,
                        encoding=enc,
                        load=True,
                        ignore_missing_memofile=True,
                        lowernames=False,  # Preserve field name case
                        char_decode_errors="replace",  # Replace invalid chars instead of failing
                        raw=True,  # Read raw values without parsing to avoid type conversion errors
                    )
                    df = pd.DataFrame(iter(dbf_reader))
                    detected_encoding = enc  # Save the encoding that worked
                    logger.info(f"Successfully read {dbf_file} with encoding: {enc}")
                    if hasattr(dbf_reader, "_memo_file_missing") or not hasattr(
                        dbf_reader, "memo"
                    ):
                        logger.warning(
                            f"Memo file missing for {dbf_file} - memo fields may be incomplete"
                        )
                        memo_file_missing = True
                    break
                except (UnicodeDecodeError, UnicodeError):
                    logger.debug(f"Failed to read {dbf_file} with encoding: {enc}")
                    continue

            if df is None:
                # Fallback: try with errors='ignore' and raw mode
                try:
                    df = pd.DataFrame(
                        iter(
                            DBF(
                                dbf_file,
                                encoding="latin1",
                                load=True,
                                ignore_missing_memofile=True,
                                char_decode_errors="replace",
                                raw=True,  # Read raw values without parsing
                            )
                        )
                    )
                    detected_encoding = "latin1"
                    logger.warning(
                        f"Read {dbf_file} with latin1 encoding in raw mode "
                        f"(some characters may be lost)"
                    )
                except Exception as exc:
                    raise ValueError(
                        f"Could not read {dbf_file} with any encoding"
                    ) from exc

        if df.empty:
            logger.warning(
                f"DBF file {dbf_file} is empty - creating empty table with schema"
            )

            # Get DBF field information even for empty files
            field_info = get_dbf_field_info(dbf_file)
            if not field_info:
                logger.error(
                    f"Could not extract field info from empty DBF file {dbf_file}"
                )
                log_ingestion(
                    engine,
                    dbf_file,
                    dbf_file.name,
                    table_name,
                    0,
                    False,
                    run_id,
                    "Could not extract schema from empty file",
                    schema,
                )
                return False

            # Create dtype mapping for table schema
            dtype_map = create_pandas_dtype_map(field_info)
            logger.info(
                f"Created type mapping for empty table with {len(field_info)} fields"
            )

            # Create empty DataFrame with proper schema
            empty_df = pd.DataFrame(columns=list(field_info.keys()))

            # Add run_id and meta_source columns
            empty_df["run_id"] = pd.Series(dtype=str)
            empty_df["meta_source"] = pd.Series(dtype=str)

            # Handle existing table logic for empty files
            exists = table_exists(engine, table_name, schema)

            if exists and if_exists == "fail":
                error_msg = f"Table {table_name} already exists and if_exists='fail'"
                logger.error(error_msg)
                log_ingestion(
                    engine,
                    dbf_file,
                    dbf_file.name,
                    table_name,
                    0,
                    False,
                    run_id,
                    error_msg,
                    schema,
                )
                return False

            if exists and if_exists == "append":
                logger.info(
                    f"Empty file {dbf_file} - table {table_name} already exists, "
                    f"skipping schema creation"
                )
                log_ingestion(
                    engine,
                    dbf_file,
                    dbf_file.name,
                    table_name,
                    0,
                    True,
                    run_id,
                    "Empty file - table already exists",
                    schema,
                )
                return True

            # Create the empty table with proper schema
            try:
                empty_df.to_sql(
                    table_name,
                    con=engine,
                    if_exists=if_exists,
                    index=False,
                    schema=schema,
                    dtype=dtype_map,
                )
                logger.success(
                    f"Created empty table {table_name} with proper schema from {dbf_file}"
                )
                log_ingestion(
                    engine,
                    dbf_file,
                    dbf_file.name,
                    table_name,
                    0,
                    True,
                    run_id,
                    "Empty file - table created with schema",
                    schema,
                )
                return True

            except Exception as e:  # pylint: disable=broad-exception-caught
                error_msg = f"Failed to create empty table {table_name}: {e}"
                logger.error(error_msg)
                log_ingestion(
                    engine,
                    dbf_file,
                    dbf_file.name,
                    table_name,
                    0,
                    False,
                    run_id,
                    error_msg,
                    schema,
                )
                return False

        logger.info(f"Loaded {len(df)} rows from {dbf_file}")

        # Get DBF field information for proper type mapping
        field_info = get_dbf_field_info(dbf_file)
        dtype_map = create_pandas_dtype_map(field_info)
        logger.info(f"Created type mapping for {len(field_info)} fields")

        # Add run_id column to track which run loaded each row
        df["run_id"] = run_id
        logger.debug(f"Added run_id column with value: {run_id}")

        # Add meta_source column to track source DBF file path
        df["meta_source"] = str(dbf_file)
        logger.debug(f"Added meta_source column with value: {str(dbf_file)}")

        # Handle existing table logic
        exists = table_exists(engine, table_name, schema)

        if exists and if_exists == "fail":
            error_msg = f"Table {table_name} already exists and if_exists='fail'"
            logger.error(error_msg)
            log_ingestion(
                engine,
                dbf_file,
                dbf_file.name,
                table_name,
                0,
                False,
                run_id,
                error_msg,
                schema,
            )
            return False

        if exists and if_exists == "append":
            try:
                df = validate_and_prepare_schema(engine, table_name, df, schema)
                logger.info(
                    f"DataFrame prepared for flexible append to table {table_name}"
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                error_msg = f"Failed to prepare schema for table {table_name}: {e}"
                logger.error(error_msg)
                log_ingestion(
                    engine,
                    dbf_file,
                    dbf_file.name,
                    table_name,
                    0,
                    False,
                    run_id,
                    error_msg,
                    schema,
                )
                return False

        # Clean data before loading to prevent SQL errors
        # Replace NaN with None for proper NULL handling
        df = df.where(pd.notna(df), None)

        # Use detected encoding for decoding bytes (fallback to latin1 if not detected)
        decode_encoding = detected_encoding if detected_encoding else "latin1"
        logger.debug(f"Using encoding '{decode_encoding}' for decoding raw bytes")

        # Process columns based on DBF field types
        for field_name, info in field_info.items():
            if field_name not in df.columns:
                continue
            field_type = info["type"]

            # Handle numeric fields - coerce errors to None
            if field_type in ["N", "F", "I", "B"]:  # Numeric types
                # First decode bytes if needed
                if df[field_name].dtype == object:

                    def decode_and_convert_numeric(val):
                        if val is None:
                            return None
                        if isinstance(val, bytes):
                            try:
                                decoded = val.decode(
                                    decode_encoding, errors="replace"
                                ).strip()
                                if not decoded:
                                    return None
                                # Try to convert to numeric
                                return pd.to_numeric(decoded, errors="coerce")
                            except Exception:  # pylint: disable=broad-exception-caught
                                return None
                        return val

                    df.loc[:, field_name] = df[field_name].apply(
                        decode_and_convert_numeric
                    )

                # Convert to numeric, invalid values become NaN (then None)
                df.loc[:, field_name] = pd.to_numeric(df[field_name], errors="coerce")

            # Handle boolean/logical fields
            elif field_type == "L":  # Logical/Boolean field

                def decode_and_convert_bool(
                    val,
                ):  # pylint: disable=too-many-return-statements
                    if val is None:
                        return None
                    if isinstance(val, bytes):
                        try:
                            decoded = (
                                val.decode(decode_encoding, errors="replace")
                                .strip()
                                .upper()
                            )
                            if decoded in ("T", "Y", "1"):
                                return True
                            if decoded in ("F", "N", "0"):
                                return False
                            if decoded in ("", " ", "?"):
                                return None
                            return None  # Invalid value
                        except Exception:  # pylint: disable=broad-exception-caught
                            return None
                    # Already a boolean or convertible
                    if isinstance(val, bool):
                        return val
                    if isinstance(val, (int, float)):
                        return bool(val)
                    return None

                df.loc[:, field_name] = df[field_name].apply(decode_and_convert_bool)

            # Handle date/datetime fields
            elif field_type in ["D", "T"]:  # Date/DateTime fields
                # Dates should already be datetime objects from DBF, but verify
                def ensure_datetime(val):
                    if val is None or pd.isna(val):
                        return None
                    if isinstance(val, (pd.Timestamp, datetime)):
                        return val
                    if isinstance(val, bytes):
                        # Try to decode and parse date
                        try:
                            decoded = val.decode(
                                decode_encoding, errors="replace"
                            ).strip()
                            if not decoded or decoded == "00000000":
                                return None
                            return pd.to_datetime(decoded, errors="coerce")
                        except Exception:  # pylint: disable=broad-exception-caught
                            return None
                    return val

                df.loc[:, field_name] = df[field_name].apply(ensure_datetime)

            # Handle text fields - clean control characters and invalid strings
            elif field_type in ["C", "M", "G", "P"]:  # Character/Memo/General fields
                if df[field_name].dtype == object:  # Only process if it's string-like

                    def clean_text(val):  # pylint: disable=too-many-return-statements
                        if val is None:
                            return None
                        if isinstance(val, bytes):
                            # Decode bytes to string
                            try:
                                decoded = val.decode(
                                    decode_encoding, errors="replace"
                                ).strip()
                                if not decoded:
                                    return None
                                # Check if it has too many unprintable characters (likely binary)
                                unprintable = sum(
                                    1
                                    for c in decoded
                                    if ord(c) < 32 and c not in "\t\n\r"
                                )
                                if (
                                    unprintable > len(decoded) * 0.3
                                ):  # More than 30% unprintable
                                    return None  # Store NULL for binary data
                                # Clean the string
                                cleaned = decoded.replace("\x00", "").replace(
                                    "\x03", ""
                                )
                                cleaned = "".join(
                                    c if ord(c) >= 32 or c in "\t\n\r" else " "
                                    for c in cleaned
                                )
                                return cleaned.strip() if cleaned.strip() else None
                            except Exception:  # pylint: disable=broad-exception-caught
                                return None
                        if isinstance(val, str):
                            # Remove or replace problematic characters
                            # Replace null bytes and other control characters
                            cleaned = val.replace("\x00", "").replace("\x03", "")
                            # Remove other control characters except tabs and newlines
                            cleaned = "".join(
                                c if ord(c) >= 32 or c in "\t\n\r" else " "
                                for c in cleaned
                            )
                            return cleaned.strip() if cleaned.strip() else None
                        return val

                    df.loc[:, field_name] = df[field_name].apply(clean_text)

        # Load data in chunks
        # Adjust chunk size for wide tables to avoid SQL statement size limits
        num_columns = len(df.columns)
        if num_columns > 100:
            adjusted_chunk_size = min(
                chunk_size, 1000
            )  # Smaller chunks for wide tables
            logger.info(
                f"Table has {num_columns} columns, using smaller chunk size: {adjusted_chunk_size}"
            )
        else:
            adjusted_chunk_size = chunk_size

        total_rows = len(df)
        rows_loaded = 0

        for i in range(0, total_rows, adjusted_chunk_size):
            chunk = df.iloc[i : i + adjusted_chunk_size]

            try:
                # Use smaller chunks for large tables to avoid SQL statement size limits
                chunk.to_sql(
                    table_name,
                    con=engine,
                    if_exists=if_exists if i == 0 else "append",
                    index=False,
                    schema=schema,
                    dtype=dtype_map,
                    # Use default method for better compatibility with problematic data
                    method=None,
                )
                rows_loaded += len(chunk)
                logger.debug(
                    f"Loaded chunk {i//adjusted_chunk_size + 1}, "
                    f"rows {i+1}-{min(i+adjusted_chunk_size, total_rows)}"
                )

            except Exception as e:  # pylint: disable=broad-exception-caught
                error_msg = f"Failed to load chunk {i//adjusted_chunk_size + 1}: {e}"
                logger.error(error_msg)
                log_ingestion(
                    engine,
                    dbf_file,
                    dbf_file.name,
                    table_name,
                    rows_loaded,
                    False,
                    run_id,
                    error_msg,
                    schema,
                )
                return False

        logger.success(
            f"Successfully loaded {rows_loaded} rows into table {table_name}"
        )
        # Check if memo file was missing and add to success message
        memo_warning = None
        success_status = True
        if memo_file_missing:
            memo_warning = "Warning: Missing memo file - memo fields may be incomplete"
            success_status = "INCOMPLETE"

        log_ingestion(
            engine,
            dbf_file,
            dbf_file.name,
            table_name,
            rows_loaded,
            success_status,
            run_id,
            memo_warning,
            schema,
        )
        return True

    except Exception as e:  # pylint: disable=broad-exception-caught
        error_msg = f"Failed to process {dbf_file}: {e}"
        logger.error(error_msg)
        log_ingestion(
            engine,
            dbf_file,
            dbf_file.name,
            table_name,
            0,
            False,
            run_id,
            error_msg,
            schema,
        )
        return False


def main():
    """Main function to run the DBF import process"""
    parser = argparse.ArgumentParser(description="Import DBF files to SQL Server")
    parser.add_argument(
        "--credentials",
        "-c",
        default="db_credentials.json",
        help="Path to data source credentials JSON file (default: db_credentials.json)",
    )
    parser.add_argument(
        "--target-db-id",
        "--tdb-id",
        default="db_01_ms",
        help="Target database identifier in credentials file (default: db_01_ms)",
    )
    parser.add_argument(
        "--dbf-dir",
        "-d",
        required=True,
        help="Directory containing DBF files",
    )
    parser.add_argument(
        "--if-exists",
        "-i",
        choices=["fail", "replace", "append"],
        default="append",
        help="Action when table exists (default: append)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Chunk size for data loading (default: 10000)",
    )
    parser.add_argument(
        "--schema",
        "-s",
        default="dbo",
        help="Database schema for tables (default: dbo)",
    )
    parser.add_argument(
        "--driver",
        choices=["auto", "pyodbc", "pymssql"],
        default="pymssql",
        help="SQL Server driver to use (default: pymssql)",
    )
    parser.add_argument(
        "--encoding",
        "-e",
        default=None,
        help=(
            "Character encoding for DBF files "
            "(e.g., cp1252, cp850, iso-8859-15, iso-8859-1, latin1). "
            "If not specified, auto-detection will try common encodings. "
            "Use this to override when auto-detection produces garbled text."
        ),
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()
    logger.info("Starting DBF to SQL Server import process")
    logger.info(
        f"Arguments: credentials={args.credentials}, dbf_dir={args.dbf_dir}, "
        f"if_exists={args.if_exists}, schema={args.schema}, driver={args.driver}"
    )

    try:
        # Load credentials and create connection
        credentials = load_credentials(args.credentials)
        engine = create_mssql_engine(credentials, args.driver, args.target_db_id)

        # Create target schema if it doesn't exist
        _create_target_schema_if_not_exist(engine, args.schema)

        # Create ingestion tracking table
        create_ingestion_table(engine, args.schema)

        # Find DBF files
        dbf_dir = Path(args.dbf_dir)
        if not dbf_dir.exists():
            logger.error(f"DBF directory does not exist: {dbf_dir}")
            sys.exit(1)

        dbf_files = [
            f for f in dbf_dir.iterdir() if f.is_file() and f.suffix.upper() == ".DBF"
        ]

        if not dbf_files:
            logger.warning(f"No DBF files found in directory: {dbf_dir}")
            sys.exit(0)

        logger.info(f"Found {len(dbf_files)} DBF files to process")

        # Generate unique run ID for this execution
        run_id = str(uuid.uuid4())
        logger.info(f"Generated run ID: {run_id}")

        # Process each DBF file
        successful_files = 0
        failed_files = 0

        for dbf_file in dbf_files:
            logger.info(
                f"Processing file {successful_files + failed_files + 1}/"
                f"{len(dbf_files)}: {dbf_file.name}"
            )

            if process_dbf_file(
                engine,
                dbf_file,
                args.if_exists,
                run_id,
                args.chunk_size,
                args.schema,
                args.encoding,
            ):
                successful_files += 1
            else:
                failed_files += 1

        # Final summary
        logger.info("Import process completed")
        logger.info(f"Successfully processed: {successful_files} files")
        logger.info(f"Failed to process: {failed_files} files")

        if failed_files > 0:
            logger.warning(
                f"{failed_files} files failed to process - check logs for details"
            )
            sys.exit(1)
        else:
            logger.success("All files processed successfully!")

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(f"Fatal error in main process: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
