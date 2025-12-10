# Submissions Storage - Deployment Guide

## Current Setup

The leaderboard uses **local JSON files** stored in the `submissions/` directory. This works great for local development but has limitations in deployment.

## How It Works

### Local Development ✅
- Submissions are saved to `submissions/` folder
- Files persist between runs
- Works perfectly for testing

### Streamlit Cloud Deployment ⚠️
- **Read-only**: If you commit `submissions/` to git, existing submissions will be available
- **Write limitations**: New submissions saved during runtime are **ephemeral** (lost when app restarts)
- Filesystem is reset on each deployment/restart

## Solutions

### Option 1: Commit Submissions to Git (Simplest)
**Pros**: Free, simple, works immediately  
**Cons**: New submissions in deployment won't persist

1. Commit your local `submissions/` folder to git:
   ```bash
   git add submissions/
   git commit -m "Add initial submissions"
   git push
   ```

2. Existing submissions will be available in deployment
3. New submissions made in deployment will be lost on restart (but visible during session)

### Option 2: Use Cloud Storage (Recommended for Production)
For persistent storage in deployment, you can use:

- **S3** (what we had before - simple but requires AWS setup)
- **GitHub Gist API** (free, simple, no auth needed for public repos)
- **Firebase Realtime Database** (free tier available)
- **Supabase** (free tier, PostgreSQL-based)

### Option 3: Accept Ephemeral Storage
If you're okay with submissions only lasting during the session:
- Current setup works fine
- Users can see leaderboard during their session
- Data resets on app restart

## Current Behavior

- ✅ **Local**: Full read/write, persists
- ✅ **Deployment (read)**: Works if `submissions/` is in git
- ⚠️ **Deployment (write)**: Saves during session, lost on restart

## Recommendation

For a simple, no-setup solution:
1. Commit your existing `submissions/` folder to git
2. Accept that new submissions in deployment are session-only
3. If you need persistence, we can add a simple cloud storage solution

The code will automatically detect deployment environments and show warnings if needed.

