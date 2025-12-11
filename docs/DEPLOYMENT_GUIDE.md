# Streamlit Deployment Guide

## Quick Deploy: Streamlit Cloud (Recommended)

### Prerequisites
1. GitHub account
2. Your code pushed to a GitHub repository

### Steps

1. **Push your code to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/rollercoaster.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to https://share.streamlit.io
   - Sign in with GitHub
   - Click "New app"
   - Select your repository and branch
   - **Main file path**: `app_builder.py`
   - Click "Deploy"

3. **Your app will be live at**: `https://your-app-name.streamlit.app`

### Important Notes
- Make sure `requirements.txt` includes all dependencies
- Streamlit Cloud automatically installs from `requirements.txt`
- Free tier includes unlimited public apps
- Private repos require paid plan

---

## Alternative: Docker Deployment

### Create Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run Streamlit
ENTRYPOINT ["streamlit", "run", "app_builder.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run
```bash
docker build -t rollercoaster-app .
docker run -p 8501:8501 rollercoaster-app
```

### Deploy to Cloud Platforms
- **Heroku**: Use Heroku Container Registry
- **AWS ECS/Fargate**: Push to ECR, create task definition
- **Google Cloud Run**: `gcloud run deploy`
- **Azure Container Instances**: Use Azure CLI

---

## Alternative: Traditional VPS

### On Ubuntu/Debian Server

```bash
# Install Python and pip
sudo apt update
sudo apt install python3 python3-pip

# Clone your repo
git clone https://github.com/yourusername/rollercoaster.git
cd rollercoaster

# Install dependencies
pip3 install -r requirements.txt

# Run with systemd service (create /etc/systemd/system/streamlit.service)
```

### systemd Service File
```ini
[Unit]
Description=Streamlit Rollercoaster App
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/rollercoaster
ExecStart=/usr/local/bin/streamlit run app_builder.py --server.port=8501
Restart=always

[Install]
WantedBy=multi-user.target
```

### With Nginx Reverse Proxy
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## Pre-Deployment Checklist

- [ ] `requirements.txt` is up to date
- [ ] All file paths are relative (no absolute paths)
- [ ] Model files are included or accessible (`models/*.pth`, `models/*.pkl`)
- [ ] **RFDB data directory included** (`rfdb_csvs/` folder must be in repository)
- [ ] **Ratings data included** (`ratings_data/` folder must be in repository)
- [ ] Environment variables configured (if needed)
- [ ] Tested locally with `streamlit run app_builder.py`
- [ ] Removed any hardcoded local paths
- [ ] Checked for sensitive data (API keys, etc.)
- [ ] Checked `.gitignore` doesn't exclude data directories

---

## Common Issues

### Missing Dependencies
- Ensure all imports are in `requirements.txt`
- Check for system dependencies (e.g., `gcc` for some packages)

### File Path Issues
- Use relative paths: `./models/model.pth` not `/Users/...`
- Use `os.path.join()` for cross-platform compatibility

### Memory Issues
- Streamlit Cloud has memory limits on free tier
- Consider optimizing large files or using cloud storage

### Model Files
- Ensure model files (`.pth`, `.pkl`) are in the repository
- Or use cloud storage (S3, GCS) and load at runtime

### RFDB Data Access
- **Critical**: The `rfdb_csvs/` directory must be included in your repository
- Check `.gitignore` to ensure `rfdb_csvs/` is NOT ignored
- The RFDB page uses improved path resolution with fallbacks
- If data is missing, you'll see helpful error messages
- For large datasets, consider:
  - Using Git LFS (Large File Storage)
  - Or hosting data separately and loading via API/cloud storage

---

## Environment Variables (if needed)

Create `.streamlit/config.toml`:
```toml
[server]
port = 8501
enableCORS = false
enableXsrfProtection = false
```

For secrets, use Streamlit Cloud's secrets management or environment variables.

