# RFDB Data Deployment - Private Data Solutions

## Problem
The `rfdb_csvs/` directory contains data that you don't want to expose in a public GitHub repository, but you need it accessible in your deployed Streamlit app.

## Solutions (Ranked by Ease)

### 1. **Private GitHub Repository** (Easiest)
- Make your repository private on GitHub
- Streamlit Cloud supports private repos (paid plan required)
- All data stays in the repo, no code changes needed
- **Cost**: Streamlit Cloud Team plan ($20/month per user)

### 2. **Cloud Storage (S3/GCS/Azure)** (Recommended for Public Repos)
Store data in cloud storage and load at runtime.

#### AWS S3 Example
```python
# In pages/02_RFDB_Data.py
import boto3
from io import StringIO

# Load credentials from Streamlit secrets
s3_client = boto3.client(
    's3',
    aws_access_key_id=st.secrets['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=st.secrets['AWS_SECRET_ACCESS_KEY']
)

def load_csv_from_s3(bucket, key):
    """Load CSV from S3 bucket"""
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))

# Use instead of local file reading
df = load_csv_from_s3('your-bucket', f'rfdb_csvs/{park}/{coaster}/{csv_name}')
```

#### Google Cloud Storage Example
```python
from google.cloud import storage

def load_csv_from_gcs(bucket_name, blob_name):
    """Load CSV from GCS bucket"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return pd.read_csv(StringIO(blob.download_as_text()))
```

**Setup**:
1. Create S3/GCS bucket
2. Upload `rfdb_csvs/` to bucket
3. Add credentials to Streamlit Cloud secrets
4. Update code to load from cloud storage

### 3. **Streamlit Secrets + Environment Variables**
Store data access credentials securely.

**In Streamlit Cloud**:
1. Go to app settings → Secrets
2. Add:
```toml
AWS_ACCESS_KEY_ID = "your-key"
AWS_SECRET_ACCESS_KEY = "your-secret"
S3_BUCKET = "your-bucket-name"
```

**In code**:
```python
import os
AWS_KEY = st.secrets.get('AWS_ACCESS_KEY_ID', os.getenv('AWS_ACCESS_KEY_ID'))
```

### 4. **Hybrid Approach: Sample Data + User Upload**
Include a small sample dataset in the repo, allow users to upload their own.

```python
# In pages/02_RFDB_Data.py
st.sidebar.file_uploader("Upload RFDB CSV", type=['csv'])

# Check if user uploaded file
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    # Use sample data from repo
    df = pd.read_csv('rfdb_csvs/sample/coaster.csv')
```

### 5. **Git LFS (Large File Storage)**
For large files, but still public if repo is public.

```bash
git lfs install
git lfs track "rfdb_csvs/**/*.csv"
git add .gitattributes
git add rfdb_csvs/
git commit -m "Add RFDB data via LFS"
```

**Note**: Still visible in public repos, just stored separately.

### 6. **Separate Data Repository + API**
Host data in a private repo, serve via API.

```python
# Load data from private API endpoint
import requests

def load_csv_from_api(coaster_id):
    response = requests.get(
        f'https://your-api.com/rfdb/{coaster_id}',
        headers={'Authorization': f'Bearer {st.secrets["API_KEY"]}'}
    )
    return pd.read_csv(StringIO(response.text))
```

---

## Recommended Solution for Your Case

**For a public repo**: Use **AWS S3 or Google Cloud Storage**

### Quick Setup Steps:

1. **Upload data to cloud storage**:
   ```bash
   # AWS S3
   aws s3 sync rfdb_csvs/ s3://your-bucket/rfdb_csvs/
   
   # Or use web console to upload
   ```

2. **Create IAM user with read-only access** (for security)

3. **Add to Streamlit Cloud secrets**:
   ```toml
   AWS_ACCESS_KEY_ID = "AKIA..."
   AWS_SECRET_ACCESS_KEY = "secret..."
   S3_BUCKET = "your-bucket-name"
   ```

4. **Update `pages/02_RFDB_Data.py`** to load from S3 instead of local files

---

## Cost Estimates

- **AWS S3**: ~$0.023/GB/month storage + $0.005/1000 requests
- **GCS**: Similar pricing
- **Streamlit Cloud Private Repo**: $20/month
- **Git LFS**: Free for <1GB, then $5/month per 50GB

For a few GB of CSV data, cloud storage costs are minimal (~$0.10-0.50/month).

---

## Security Best Practices

1. **Never commit credentials** - Use Streamlit secrets
2. **Use IAM roles with minimal permissions** - Read-only for data bucket
3. **Enable bucket versioning** - For data recovery
4. **Use bucket policies** - Restrict access by IP if needed
5. **Rotate credentials regularly**

---

## Implementation Helper

I can help you implement the S3/GCS solution by:
1. Creating a helper module for cloud data loading
2. Updating `pages/02_RFDB_Data.py` to use cloud storage
3. Adding fallback to local files for development
4. Creating setup instructions

Would you like me to implement one of these solutions?

