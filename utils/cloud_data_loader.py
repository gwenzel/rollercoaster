"""
Cloud data loader for RFDB CSV files.
Supports both local development and cloud deployment (S3/GCS).
"""

import os
import pandas as pd
import streamlit as st
from io import StringIO
from typing import Optional

def load_rfdb_csv(park: str, coaster: str, csv_name: str, use_cloud: bool = True) -> Optional[pd.DataFrame]:
    """
    Load RFDB CSV file from cloud storage or local filesystem.
    
    Args:
        park: Park name (folder name)
        coaster: Coaster name (folder name)
        csv_name: CSV filename
        use_cloud: If True, try cloud storage first; if False, only use local
        
    Returns:
        DataFrame if successful, None otherwise
    """
    # Try cloud storage first (if enabled and credentials available)
    if use_cloud:
        df = _load_from_s3(park, coaster, csv_name)
        if df is not None:
            return df
        
        df = _load_from_gcs(park, coaster, csv_name)
        if df is not None:
            return df
    
    # Fallback to local filesystem
    return _load_from_local(park, coaster, csv_name)


def _load_from_s3(park: str, coaster: str, csv_name: str) -> Optional[pd.DataFrame]:
    """Load CSV from AWS S3."""
    try:
        import boto3
        
        # Get credentials from Streamlit secrets or environment
        access_key = st.secrets.get('AWS_ACCESS_KEY_ID') if hasattr(st, 'secrets') else os.getenv('AWS_ACCESS_KEY_ID')
        secret_key = st.secrets.get('AWS_SECRET_ACCESS_KEY') if hasattr(st, 'secrets') else os.getenv('AWS_SECRET_ACCESS_KEY')
        bucket = st.secrets.get('S3_BUCKET') if hasattr(st, 'secrets') else os.getenv('S3_BUCKET', 'rfdb-data')
        
        if not access_key or not secret_key:
            return None
        
        s3_client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )
        
        key = f'rfdb_csvs/{park}/{coaster}/{csv_name}'
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
        return df
        
    except Exception as e:
        # Silently fail - will try other methods
        return None


def _load_from_gcs(park: str, coaster: str, csv_name: str) -> Optional[pd.DataFrame]:
    """Load CSV from Google Cloud Storage."""
    try:
        from google.cloud import storage
        
        # Get credentials from Streamlit secrets or environment
        bucket_name = st.secrets.get('GCS_BUCKET') if hasattr(st, 'secrets') else os.getenv('GCS_BUCKET')
        if not bucket_name:
            return None
        
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob_name = f'rfdb_csvs/{park}/{coaster}/{csv_name}'
        blob = bucket.blob(blob_name)
        
        if not blob.exists():
            return None
        
        content = blob.download_as_text()
        df = pd.read_csv(StringIO(content))
        return df
        
    except Exception as e:
        # Silently fail - will try other methods
        return None


def _load_from_local(park: str, coaster: str, csv_name: str) -> Optional[pd.DataFrame]:
    """Load CSV from local filesystem."""
    try:
        # Try multiple possible locations
        possible_paths = [
            os.path.join('rfdb_csvs', park, coaster, csv_name),
            os.path.join(os.getcwd(), 'rfdb_csvs', park, coaster, csv_name),
            os.path.join(os.path.dirname(__file__), '..', 'rfdb_csvs', park, coaster, csv_name),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return pd.read_csv(path)
        
        return None
        
    except Exception as e:
        return None


def list_rfdb_parks(use_cloud: bool = True) -> list:
    """List available parks from cloud or local storage."""
    if use_cloud:
        parks = _list_parks_from_s3()
        if parks:
            return parks
        
        parks = _list_parks_from_gcs()
        if parks:
            return parks
    
    return _list_parks_from_local()


def _list_parks_from_s3() -> list:
    """List parks from S3 bucket."""
    try:
        import boto3
        
        access_key = st.secrets.get('AWS_ACCESS_KEY_ID') if hasattr(st, 'secrets') else os.getenv('AWS_ACCESS_KEY_ID')
        secret_key = st.secrets.get('AWS_SECRET_ACCESS_KEY') if hasattr(st, 'secrets') else os.getenv('AWS_SECRET_ACCESS_KEY')
        bucket = st.secrets.get('S3_BUCKET') if hasattr(st, 'secrets') else os.getenv('S3_BUCKET', 'rfdb-data')
        
        if not access_key or not secret_key:
            return []
        
        s3_client = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)
        
        # List all prefixes under rfdb_csvs/
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix='rfdb_csvs/', Delimiter='/')
        parks = [prefix['Prefix'].split('/')[1] for prefix in response.get('CommonPrefixes', [])]
        return parks
        
    except Exception:
        return []


def _list_parks_from_gcs() -> list:
    """List parks from GCS bucket."""
    try:
        from google.cloud import storage
        
        bucket_name = st.secrets.get('GCS_BUCKET') if hasattr(st, 'secrets') else os.getenv('GCS_BUCKET')
        if not bucket_name:
            return []
        
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        parks = set()
        for blob in bucket.list_blobs(prefix='rfdb_csvs/'):
            parts = blob.name.split('/')
            if len(parts) >= 2:
                parks.add(parts[1])
        
        return sorted(list(parks))
        
    except Exception:
        return []


def _list_parks_from_local() -> list:
    """List parks from local filesystem."""
    try:
        possible_roots = [
            'rfdb_csvs',
            os.path.join(os.getcwd(), 'rfdb_csvs'),
            os.path.join(os.path.dirname(__file__), '..', 'rfdb_csvs'),
        ]
        
        for root in possible_roots:
            if os.path.exists(root) and os.path.isdir(root):
                parks = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
                if parks:
                    return sorted(parks)
        
        return []
        
    except Exception:
        return []


def list_rfdb_coasters(park: str, use_cloud: bool = True) -> list:
    """List available coasters for a park from cloud or local storage."""
    if use_cloud:
        coasters = _list_coasters_from_s3(park)
        if coasters:
            return coasters
        
        coasters = _list_coasters_from_gcs(park)
        if coasters:
            return coasters
    
    return _list_coasters_from_local(park)


def _list_coasters_from_s3(park: str) -> list:
    """List coasters from S3 bucket for a given park."""
    try:
        import boto3
        
        access_key = st.secrets.get('AWS_ACCESS_KEY_ID') if hasattr(st, 'secrets') else os.getenv('AWS_ACCESS_KEY_ID')
        secret_key = st.secrets.get('AWS_SECRET_ACCESS_KEY') if hasattr(st, 'secrets') else os.getenv('AWS_SECRET_ACCESS_KEY')
        bucket = st.secrets.get('S3_BUCKET') if hasattr(st, 'secrets') else os.getenv('S3_BUCKET', 'rfdb-data')
        
        if not access_key or not secret_key:
            return []
        
        s3_client = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)
        
        # List all prefixes under rfdb_csvs/{park}/
        prefix = f'rfdb_csvs/{park}/'
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter='/')
        coasters = [prefix['Prefix'].split('/')[2] for prefix in response.get('CommonPrefixes', [])]
        return coasters
        
    except Exception:
        return []


def _list_coasters_from_gcs(park: str) -> list:
    """List coasters from GCS bucket for a given park."""
    try:
        from google.cloud import storage
        
        bucket_name = st.secrets.get('GCS_BUCKET') if hasattr(st, 'secrets') else os.getenv('GCS_BUCKET')
        if not bucket_name:
            return []
        
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        coasters = set()
        prefix = f'rfdb_csvs/{park}/'
        for blob in bucket.list_blobs(prefix=prefix):
            parts = blob.name.split('/')
            if len(parts) >= 3:
                coasters.add(parts[2])
        
        return sorted(list(coasters))
        
    except Exception:
        return []


def _list_coasters_from_local(park: str) -> list:
    """List coasters from local filesystem for a given park."""
    try:
        possible_roots = [
            'rfdb_csvs',
            os.path.join(os.getcwd(), 'rfdb_csvs'),
            os.path.join(os.path.dirname(__file__), '..', 'rfdb_csvs'),
        ]
        
        for root in possible_roots:
            park_path = os.path.join(root, park)
            if os.path.exists(park_path) and os.path.isdir(park_path):
                coasters = [d for d in os.listdir(park_path) if os.path.isdir(os.path.join(park_path, d))]
                if coasters:
                    return sorted(coasters)
        
        return []
        
    except Exception:
        return []


def list_rfdb_csvs(park: str, coaster: str, use_cloud: bool = True) -> list:
    """List available CSV files for a coaster from cloud or local storage."""
    if use_cloud:
        csvs = _list_csvs_from_s3(park, coaster)
        if csvs:
            return csvs
        
        csvs = _list_csvs_from_gcs(park, coaster)
        if csvs:
            return csvs
    
    return _list_csvs_from_local(park, coaster)


def _list_csvs_from_s3(park: str, coaster: str) -> list:
    """List CSV files from S3 bucket for a given park/coaster."""
    try:
        import boto3
        
        access_key = st.secrets.get('AWS_ACCESS_KEY_ID') if hasattr(st, 'secrets') else os.getenv('AWS_ACCESS_KEY_ID')
        secret_key = st.secrets.get('AWS_SECRET_ACCESS_KEY') if hasattr(st, 'secrets') else os.getenv('AWS_SECRET_ACCESS_KEY')
        bucket = st.secrets.get('S3_BUCKET') if hasattr(st, 'secrets') else os.getenv('S3_BUCKET', 'rfdb-data')
        
        if not access_key or not secret_key:
            return []
        
        s3_client = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)
        
        # List all objects under rfdb_csvs/{park}/{coaster}/
        prefix = f'rfdb_csvs/{park}/{coaster}/'
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        
        csv_files = []
        if 'Contents' in response:
            for obj in response['Contents']:
                key = obj['Key']
                filename = os.path.basename(key)
                if filename.endswith('.csv'):
                    csv_files.append(filename)
        
        return sorted(csv_files)
        
    except Exception:
        return []


def _list_csvs_from_gcs(park: str, coaster: str) -> list:
    """List CSV files from GCS bucket for a given park/coaster."""
    try:
        from google.cloud import storage
        
        bucket_name = st.secrets.get('GCS_BUCKET') if hasattr(st, 'secrets') else os.getenv('GCS_BUCKET')
        if not bucket_name:
            return []
        
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        csv_files = []
        prefix = f'rfdb_csvs/{park}/{coaster}/'
        for blob in bucket.list_blobs(prefix=prefix):
            if blob.name.endswith('.csv'):
                filename = os.path.basename(blob.name)
                csv_files.append(filename)
        
        return sorted(csv_files)
        
    except Exception:
        return []


def _list_csvs_from_local(park: str, coaster: str) -> list:
    """List CSV files from local filesystem for a given park/coaster."""
    try:
        possible_roots = [
            'rfdb_csvs',
            os.path.join(os.getcwd(), 'rfdb_csvs'),
            os.path.join(os.path.dirname(__file__), '..', 'rfdb_csvs'),
        ]
        
        for root in possible_roots:
            coaster_path = os.path.join(root, park, coaster)
            if os.path.exists(coaster_path) and os.path.isdir(coaster_path):
                csv_files = [f for f in os.listdir(coaster_path) if f.endswith('.csv')]
                if csv_files:
                    return sorted(csv_files)
        
        return []
        
    except Exception:
        return []

