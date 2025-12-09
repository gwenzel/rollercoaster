# S3 Bucket Setup Guide for RFDB Data

## What You Need

### 1. AWS Account & S3 Bucket
- AWS account (free tier is fine)
- S3 bucket created (e.g., `rfdb-data` or `rollercoaster-rfdb`)
- Bucket region (e.g., `us-east-1`)

### 2. IAM User with Access
- IAM user with programmatic access
- Access key ID and secret access key
- Permissions: Read-only access to the bucket (for security)

### 3. Data Structure in S3
Upload your `rfdb_csvs/` folder maintaining the same structure:
```
s3://your-bucket-name/
└── rfdb_csvs/
    ├── cedarpoint/
    │   ├── steelvengeance/
    │   │   ├── run1.csv
    │   │   ├── run2.csv
    │   │   └── ...
    │   └── ...
    ├── sixflagsmagicmountain/
    └── ...
```

### 4. Streamlit Secrets Configuration
Add these to Streamlit Cloud secrets (or `.streamlit/secrets.toml` for local):

```toml
AWS_ACCESS_KEY_ID = "AKIA..."
AWS_SECRET_ACCESS_KEY = "your-secret-key"
S3_BUCKET = "your-bucket-name"
```

---

## Step-by-Step Setup

### Step 1: Create S3 Bucket

1. Go to AWS Console → S3
2. Click "Create bucket"
3. Bucket name: `rfdb-data` (or your choice, must be globally unique)
4. Region: Choose closest to your users (e.g., `us-east-1`)
5. **Uncheck "Block all public access"** (or configure bucket policy for private access)
6. Click "Create bucket"

### Step 2: Upload Data

**Option A: AWS Console (Web UI)**
1. Go to your bucket
2. Click "Upload"
3. Upload entire `rfdb_csvs/` folder
4. Maintain folder structure: `rfdb_csvs/park/coaster/file.csv`

**Option B: AWS CLI (Faster for large datasets)**
```bash
# Install AWS CLI if needed
pip install awscli

# Configure credentials
aws configure
# Enter: Access Key ID, Secret Key, Region

# Upload folder
aws s3 sync rfdb_csvs/ s3://your-bucket-name/rfdb_csvs/
```

**Option C: Python Script**
```python
import boto3
import os

s3 = boto3.client('s3', 
    aws_access_key_id='YOUR_KEY',
    aws_secret_access_key='YOUR_SECRET'
)

bucket = 'your-bucket-name'
local_dir = 'rfdb_csvs'

for root, dirs, files in os.walk(local_dir):
    for file in files:
        local_path = os.path.join(root, file)
        s3_path = os.path.relpath(local_path, local_dir)
        s3_key = f'rfdb_csvs/{s3_path}'
        s3.upload_file(local_path, bucket, s3_key)
        print(f"Uploaded: {s3_key}")
```

### Step 3: Create IAM User (Read-Only Access)

1. Go to AWS Console → IAM → Users
2. Click "Add users"
3. Username: `streamlit-rfdb-reader`
4. Access type: **Programmatic access**
5. Click "Next: Permissions"
6. Click "Attach existing policies directly"
7. Search for "S3" and select: **AmazonS3ReadOnlyAccess**
   - Or create custom policy (see below)
8. Click through to create user
9. **Save the Access Key ID and Secret Access Key** (shown only once!)

**Custom Policy (More Restrictive - Recommended)**
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::your-bucket-name",
                "arn:aws:s3:::your-bucket-name/*"
            ]
        }
    ]
}
```

### Step 4: Configure Streamlit Secrets

**For Streamlit Cloud:**
1. Go to your app on share.streamlit.io
2. Click "Settings" → "Secrets"
3. Add:
```toml
AWS_ACCESS_KEY_ID = "AKIA..."
AWS_SECRET_ACCESS_KEY = "your-secret-key-here"
S3_BUCKET = "your-bucket-name"
```

**For Local Development:**
Create `.streamlit/secrets.toml`:
```toml
AWS_ACCESS_KEY_ID = "AKIA..."
AWS_SECRET_ACCESS_KEY = "your-secret-key-here"
S3_BUCKET = "your-bucket-name"
```

**⚠️ Important:** Add `.streamlit/secrets.toml` to `.gitignore`!

### Step 5: Update RFDB Page to Use Cloud Loader

The `utils/cloud_data_loader.py` is ready. We need to update `pages/02_RFDB_Data.py` to use it.

---

## Cost Estimate

**Storage:**
- 1 GB = ~$0.023/month
- 10 GB = ~$0.23/month
- 100 GB = ~$2.30/month

**Requests:**
- GET requests: $0.0004 per 1,000 requests
- LIST requests: $0.0005 per 1,000 requests

**Example:** 10 GB data, 1,000 page views/month
- Storage: $0.23
- Requests: ~$0.01
- **Total: ~$0.24/month**

---

## Security Best Practices

1. ✅ Use IAM user (not root account)
2. ✅ Grant only read-only permissions
3. ✅ Never commit credentials to git
4. ✅ Use Streamlit secrets (encrypted)
5. ✅ Rotate keys periodically
6. ✅ Enable bucket versioning (optional)
7. ✅ Set up CloudTrail logging (optional)

---

## Testing

After setup, test locally:
```python
from utils.cloud_data_loader import load_rfdb_csv, list_rfdb_parks

# Test listing parks
parks = list_rfdb_parks(use_cloud=True)
print(f"Found {len(parks)} parks")

# Test loading a CSV
df = load_rfdb_csv('cedarpoint', 'steelvengeance', 'run1.csv', use_cloud=True)
if df is not None:
    print(f"Loaded {len(df)} rows")
else:
    print("Failed to load from S3")
```

---

## Troubleshooting

**"Access Denied"**
- Check IAM user has `s3:GetObject` and `s3:ListBucket` permissions
- Verify bucket name in secrets matches actual bucket

**"Bucket not found"**
- Check bucket name spelling
- Verify bucket exists in the correct region
- Check AWS credentials are correct

**"No parks found"**
- Verify data structure: `s3://bucket/rfdb_csvs/park/coaster/file.csv`
- Check bucket permissions allow listing

**Slow loading**
- Consider using CloudFront CDN (advanced)
- Or use regional bucket closer to users

---

## Next Steps

Once your S3 bucket is ready, I can:
1. Update `pages/02_RFDB_Data.py` to use the cloud loader
2. Add fallback to local files for development
3. Test the integration

Let me know when your bucket is set up and I'll help integrate it!

