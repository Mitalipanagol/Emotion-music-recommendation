# 🚀 GitHub Repository Setup Guide

Complete step-by-step guide to create and upload your Emotion-Based Music Recommendation System to GitHub.

---

## 📋 Prerequisites

- ✅ Git installed on your computer
- ✅ GitHub account created
- ✅ Project files ready

---

## 🔧 Step 1: Install Git (If Not Already Installed)

### Windows
1. Download Git from: https://git-scm.com/download/win
2. Run the installer
3. Use default settings

### Verify Installation
```bash
git --version
```

---

## 🌐 Step 2: Create GitHub Repository

### Option A: Using GitHub Website

1. **Go to GitHub**: https://github.com
2. **Sign in** to your account
3. **Click** the "+" icon (top right) → "New repository"
4. **Fill in details**:
   - **Repository name**: `emotion-music-recommendation`
   - **Description**: `A deep learning system that detects emotions and recommends music`
   - **Visibility**: Public (or Private if you prefer)
   - **DO NOT** initialize with README (we already have one)
   - **DO NOT** add .gitignore (we already have one)
   - **DO NOT** choose a license (we already have one)
5. **Click** "Create repository"

### Option B: Using GitHub CLI (Advanced)
```bash
gh repo create emotion-music-recommendation --public --source=. --remote=origin
```

---

## 💻 Step 3: Initialize Git in Your Project

Open **Command Prompt** or **PowerShell** in your project folder:

```bash
cd "C:\Users\vpana\OneDrive\Desktop\major project"
```

### Initialize Git Repository
```bash
git init
```

### Configure Git (First Time Only)
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

---

## 📝 Step 4: Prepare Files for Upload

### Replace README with GitHub Version
```bash
# Backup current README
copy README.md README_OLD.md

# Use GitHub README
copy README_GITHUB.md README.md
```

### Verify .gitignore
Make sure `.gitignore` is present and contains:
```
__pycache__/
*.pyc
model/*.h5
data/train/
data/test/
data/*.csv
data/*.zip
```

---

## 📦 Step 5: Add Files to Git

### Add All Files
```bash
git add .
```

### Check What Will Be Committed
```bash
git status
```

You should see:
- ✅ Green files = Will be committed
- ❌ Red files = Ignored (large files like model, dataset)

### Create First Commit
```bash
git commit -m "Initial commit: Emotion-based music recommendation system"
```

---

## 🔗 Step 6: Connect to GitHub

### Add Remote Repository
Replace `YOUR_USERNAME` with your actual GitHub username:

```bash
git remote add origin https://github.com/YOUR_USERNAME/emotion-music-recommendation.git
```

### Verify Remote
```bash
git remote -v
```

---

## 🚀 Step 7: Push to GitHub

### Push to Main Branch
```bash
git branch -M main
git push -u origin main
```

### Enter Credentials
- **Username**: Your GitHub username
- **Password**: Use **Personal Access Token** (not your password)

#### How to Create Personal Access Token:
1. Go to: https://github.com/settings/tokens
2. Click "Generate new token" → "Generate new token (classic)"
3. Give it a name: "Emotion Music Project"
4. Select scopes: ✅ `repo` (all)
5. Click "Generate token"
6. **Copy the token** (you won't see it again!)
7. Use this token as password when pushing

---

## ✅ Step 8: Verify Upload

1. Go to: `https://github.com/YOUR_USERNAME/emotion-music-recommendation`
2. You should see all your files uploaded
3. README.md should be displayed on the main page

---

## 📁 What Gets Uploaded vs Ignored

### ✅ Uploaded to GitHub:
- Source code (`.py` files)
- Documentation (`.md` files)
- Requirements (`requirements.txt`)
- Sample images (`test_images/`)
- Screenshots (`screenshots/`)
- Configuration files

### ❌ NOT Uploaded (Too Large):
- Trained model (`model/emotion_model.h5`) - 4.87 MB
- Dataset (`data/train/`, `data/test/`) - ~300 MB
- Cache files (`__pycache__/`)

---

## 🔄 Step 9: Future Updates

### After Making Changes

```bash
# Check what changed
git status

# Add changed files
git add .

# Commit with message
git commit -m "Description of changes"

# Push to GitHub
git push
```

---

## 📊 Step 10: Add Model File (Optional)

If you want to upload the trained model (4.87 MB):

### Option A: Regular Git (if < 100 MB)
```bash
# Remove model from .gitignore
# Edit .gitignore and remove line: model/*.h5

git add model/emotion_model.h5
git commit -m "Add trained model"
git push
```

### Option B: Git LFS (for large files)
```bash
# Install Git LFS
git lfs install

# Track .h5 files
git lfs track "*.h5"

# Add and commit
git add .gitattributes
git add model/emotion_model.h5
git commit -m "Add trained model with LFS"
git push
```

---

## 🎨 Step 11: Customize Repository

### Add Topics/Tags
1. Go to your repository on GitHub
2. Click "⚙️ Settings" → "About" (gear icon)
3. Add topics: `deep-learning`, `emotion-recognition`, `music-recommendation`, `tensorflow`, `streamlit`, `computer-vision`, `python`

### Add Repository Description
- Short description: "Deep learning system for emotion detection and music recommendation"

### Add Website (Optional)
- If you deploy to Heroku/Streamlit Cloud, add the URL here

---

## 📸 Step 12: Add Screenshots to README

1. Upload screenshots to `screenshots/` folder
2. Commit and push
3. Update README.md with image links:

```markdown
## Demo

### Webcam Mode
![Webcam Demo](screenshots/screenshot_webcam.png)

### Happy Emotion Detection
![Happy Detection](screenshots/screenshot_happy.png)
```

---

## 🌟 Step 13: Make Repository Stand Out

### Add Badges
Already included in README_GITHUB.md:
- Python version
- TensorFlow version
- Streamlit version
- License
- Status

### Create GitHub Pages (Optional)
1. Go to Settings → Pages
2. Source: Deploy from branch `main`
3. Folder: `/docs` or `/` (root)
4. Save

### Add Social Preview Image
1. Go to Settings
2. Scroll to "Social preview"
3. Upload an image (1280×640 px recommended)

---

## 🐛 Troubleshooting

### Problem: "Permission denied"
**Solution**: Use Personal Access Token instead of password

### Problem: "Large files detected"
**Solution**: Make sure `.gitignore` is working. Check with `git status`

### Problem: "Repository not found"
**Solution**: Check remote URL with `git remote -v`

### Problem: "Failed to push"
**Solution**: Pull first with `git pull origin main --rebase`, then push

---

## 📋 Quick Command Reference

```bash
# Initialize
git init
git add .
git commit -m "Initial commit"

# Connect to GitHub
git remote add origin https://github.com/YOUR_USERNAME/emotion-music-recommendation.git
git branch -M main
git push -u origin main

# Future updates
git add .
git commit -m "Update message"
git push
```

---

## ✅ Checklist

Before pushing to GitHub, make sure:

- [ ] `.gitignore` is present and configured
- [ ] README_GITHUB.md is renamed to README.md
- [ ] LICENSE file is present
- [ ] requirements.txt is up to date
- [ ] Large files (model, dataset) are ignored
- [ ] Personal information is removed
- [ ] Code is clean and commented
- [ ] Documentation is complete

---

## 🎉 Success!

Your repository is now live on GitHub! 🚀

**Share your repository**:
```
https://github.com/YOUR_USERNAME/emotion-music-recommendation
```

---

**Need Help?** Check GitHub documentation: https://docs.github.com

