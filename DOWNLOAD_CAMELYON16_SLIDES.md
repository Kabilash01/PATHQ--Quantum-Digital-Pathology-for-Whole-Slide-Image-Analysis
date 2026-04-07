# CAMELYON16 Dataset Download Guide

## Current Status
✅ You have: 50 XML annotation files
✅ You have: stage_labels.csv
❌ Missing: Actual whole slide images (.tif files)

## What You Need to Download

CAMELYON16 has 400 whole slide images split into:
- **Training set**: 270 slides (160 normal + 110 tumor)
- **Test set**: 130 slides (80 normal + 50 tumor)

Each slide is 1-3 GB, so:
- Full dataset = ~700 GB ❌ (Don't download all!)
- Start with 10-20 slides = ~20-30 GB ✅ (Recommended)

---

## Method 1: Download via GigaScience (Recommended)

### Step 1: Access the Data
1. Go to: https://doi.org/10.5524/100439
2. Look for "CAMELYON16" folder
3. Navigate to: `CAMELYON16/training/`

### Step 2: Download Specific Slides

**Option A: Start with 4 slides (~5-10 GB)**
Download these to test your pipeline:
```
tumor/tumor_001.tif
tumor/tumor_009.tif
normal/normal_001.tif
normal/normal_002.tif
```

**Option B: Small training set (20 slides ~30 GB)**
```
Tumor slides (10):
tumor_001.tif through tumor_010.tif

Normal slides (10):
normal_001.tif through normal_010.tif
```

### Step 3: Place Files in Correct Location
After downloading, move them:
```bash
cd ~/Downloads
mv tumor_*.tif ~/PATHQ--Quantum-Digital-Pathology-for-Whole-Slide-Image-Analysis-DMI-/notebooks/data/camelyon16/
mv normal_*.tif ~/PATHQ--Quantum-Digital-Pathology-for-Whole-Slide-Image-Analysis-DMI-/notebooks/data/camelyon16/
```

---

## Method 2: Download via AWS (Faster)

If you have AWS CLI installed:

```bash
# Install AWS CLI first (if not installed)
sudo apt install awscli -y

# Navigate to your data directory
cd ~/PATHQ--Quantum-Digital-Pathology-for-Whole-Slide-Image-Analysis-DMI-/notebooks/data/camelyon16/

# Download specific slides (example)
aws s3 cp s3://camelyon-dataset/CAMELYON16/training/tumor/tumor_001.tif . --no-sign-request
aws s3 cp s3://camelyon-dataset/CAMELYON16/training/tumor/tumor_009.tif . --no-sign-request
aws s3 cp s3://camelyon-dataset/CAMELYON16/training/normal/normal_001.tif . --no-sign-request
aws s3 cp s3://camelyon-dataset/CAMELYON16/training/normal/normal_002.tif . --no-sign-request
```

**Note:** Check the actual AWS path at https://registry.opendata.aws/camelyon/

---

## Method 3: Download via wget (If Direct Links Available)

If GigaScience provides direct download links, you can use:

```bash
cd ~/PATHQ--Quantum-Digital-Pathology-for-Whole-Slide-Image-Analysis-DMI-/notebooks/data/camelyon16/

# Example (replace with actual URLs from GigaScience)
wget "https://[gigascience-url]/tumor_001.tif"
wget "https://[gigascience-url]/tumor_009.tif"
wget "https://[gigascience-url]/normal_001.tif"
wget "https://[gigascience-url]/normal_002.tif"
```

---

## Recommended Download Strategy

### Phase 1: Test (Start Here)
Download **4 slides** (~5-10 GB):
- 2 tumor slides
- 2 normal slides

This lets you test your entire pipeline without waiting days.

### Phase 2: Development (Week 2-3)
Download **20 slides** (~30 GB):
- 10 tumor slides
- 10 normal slides

Enough for real training and validation.

### Phase 3: Full Training (Week 7-8)
Download **100-200 slides** (~150-300 GB):
Only if your pipeline works and you need more data for final results.

---

## Verify Your Download

After downloading, run:
```bash
cd ~/PATHQ--Quantum-Digital-Pathology-for-Whole-Slide-Image-Analysis-DMI-/notebooks/data/camelyon16/
ls -lh *.tif
```

You should see files like:
```
tumor_001.tif    2.1G
tumor_009.tif    1.8G
normal_001.tif   1.5G
normal_002.tif   1.6G
```

---

## Matching Slides with Annotations

Your XML annotations correspond to slides like this:
- `patient_004_node_4.xml` → `tumor_004.tif`
- `patient_009_node_1.xml` → `tumor_009.tif`

Not all slides have annotations (normal slides don't).

---

## Troubleshooting

**Q: Download too slow?**
- Try AWS method (usually faster)
- Use download manager with resume capability
- Download during off-peak hours

**Q: Not enough disk space?**
- Start with just 4 slides
- Delete after feature extraction (save .pt features only)
- Use external hard drive

**Q: Can't find download links?**
- Post on CAMELYON16 forum
- Email challenge organizers
- Use alternative pathology datasets (BRACS, TCGAlung)

---

## Alternative: Start Without CAMELYON16

While downloading, you can work with:
1. **PatchCamelyon** (already available via HuggingFace)
2. **PathMNIST** (small, quick download)
3. **CMU-1-Small-Region.svs** (already in your folder)

These let you build and test your pipeline immediately!
