# PowerShell Conda Environment Configuration

## ✅ Configuration Complete!

Your PowerShell has been configured to **automatically activate the `rc` conda environment** every time you open a new PowerShell window.

## 📁 What Was Created

### PowerShell Profile
**Location**: `C:\Users\Lenovo\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1`

This profile script:
1. Initializes conda for PowerShell
2. Automatically activates the `rc` environment
3. Sets a custom prompt showing the active environment
4. Displays a welcome message with Python version

### Execution Policy
Set to `RemoteSigned` for CurrentUser to allow the profile script to run.

## 🚀 How to Use

### Option 1: Open New PowerShell Window
Simply open a new PowerShell window and the `rc` environment will be automatically activated!

You should see:
```
🐍 Conda environment "rc" activated!
📍 Python: Python 3.x.x
(rc) PS C:\Users\Lenovo\rollercoaster>
```

### Option 2: Reload Current Session
To apply changes to your current PowerShell session:
```powershell
. $PROFILE
```

## 🎯 Quick Test

After opening a new PowerShell or reloading the profile:

```powershell
# Check Python is from rc environment
python --version

# Check if streamlit is available
python -c "import streamlit; print('Streamlit version:', streamlit.__version__)"

# Run the drawing app
streamlit run app_drawing.py --server.port 8502

# Or run the main app
streamlit run app.py
```

## 🛠️ Customization

### To Change Auto-Activated Environment

Edit the profile:
```powershell
notepad $PROFILE
```

Change this line:
```powershell
conda activate rc
```

To:
```powershell
conda activate your_environment_name
```

### To Disable Auto-Activation

Comment out the activation line:
```powershell
# conda activate rc
```

### To Add More Startup Commands

Add them to the profile after the conda activation:
```powershell
# Example: Set default directory
Set-Location C:\Users\Lenovo\rollercoaster

# Example: Display project info
Write-Host "🎢 Roller Coaster Project" -ForegroundColor Magenta
```

## 📋 Profile Contents

Your current profile includes:

```powershell
# Conda initialization for PowerShell
# Auto-activate 'rc' environment

# Initialize conda for PowerShell
(& 'C:\Users\Lenovo\anaconda3\Scripts\conda.exe' 'shell.powershell' 'hook') | Out-String | Invoke-Expression

# Auto-activate rc environment
conda activate rc

# Set friendly prompt
function prompt {
    "$(if ($env:CONDA_DEFAULT_ENV) {'('+$env:CONDA_DEFAULT_ENV+') '})PS $($executionContext.SessionState.Path.CurrentLocation)$('>' * ($nestedPromptLevel + 1)) "
}

Write-Host '🐍 Conda environment "rc" activated!' -ForegroundColor Green
Write-Host '📍 Python:' (python --version) -ForegroundColor Cyan
```

## 🔧 Troubleshooting

### Profile Not Loading

**Issue**: PowerShell doesn't run the profile
**Solution**: Check execution policy
```powershell
Get-ExecutionPolicy -Scope CurrentUser
```
Should be `RemoteSigned` or `Unrestricted`

If not:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Conda Command Not Found

**Issue**: Error about conda not being recognized
**Solution**: Verify conda path in profile matches your installation
```powershell
# Check if conda.exe exists
Test-Path 'C:\Users\Lenovo\anaconda3\Scripts\conda.exe'
```

If False, update the path in your profile.

### Wrong Environment Activates

**Issue**: Different environment activates
**Solution**: Check `.condarc` for default environment settings
```powershell
conda config --show-sources
```

### Slow PowerShell Startup

**Issue**: PowerShell takes long to start
**Solution**: Conda initialization can be slow. This is normal. Typical startup time: 2-3 seconds.

## 📚 Additional Commands

### Manage Conda Environments

```powershell
# List all environments
conda env list

# Create new environment
conda create -n myenv python=3.11

# Activate different environment (overrides auto-activation)
conda activate myenv

# Deactivate environment
conda deactivate

# Return to rc environment
conda activate rc
```

### Check What's Installed

```powershell
# List packages in current environment
conda list

# Check if specific package exists
conda list streamlit

# Install new package
conda install package_name

# Or use pip
pip install package_name
```

## 🎨 VS Code Integration

VS Code terminals should also use this configuration automatically. If not:

1. **Open VS Code Settings** (Ctrl+,)
2. Search for: `terminal.integrated.defaultProfile.windows`
3. Set to: `PowerShell`
4. Restart VS Code

Or in `settings.json`:
```json
{
    "terminal.integrated.defaultProfile.windows": "PowerShell",
    "python.defaultInterpreterPath": "C:\\Users\\Lenovo\\anaconda3\\envs\\rc\\python.exe"
}
```

## 🚀 Now You Can:

✅ **Open PowerShell** → `rc` environment auto-activates  
✅ **Run `streamlit run app.py`** → Works immediately  
✅ **Run `streamlit run app_drawing.py`** → Works immediately  
✅ **Use `python`, `pip`, `conda`** → All point to `rc` environment  
✅ **No more "command not found"** errors for streamlit  

## 🔄 To Apply Changes Now

Close this PowerShell window and open a new one, OR run:

```powershell
. $PROFILE
```

---

**Profile Location**: `$PROFILE` = `C:\Users\Lenovo\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1`

**Edit Profile**: `notepad $PROFILE`

**Reload Profile**: `. $PROFILE`
