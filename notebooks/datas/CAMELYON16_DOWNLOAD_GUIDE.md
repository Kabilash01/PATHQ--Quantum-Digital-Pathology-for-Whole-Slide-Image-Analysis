# CAMELYON16 Download Guide

## Step 1: Download Small Files First (from current page)

Download these 3 files from GigaScience:

1. **reference.csv** (3.56 KB)
   - Lists which slides are normal vs tumor
   - Essential for training

2. **lesion_annotations.zip** (329.43 KB - CAMELYON16 version)
   - Tumor region annotations
   - Needed for patch extraction

3. **evaluation_python.zip** (3.08 KB)
   - Evaluation scripts
   - Optional but useful

**Place these in:** `notebooks/data/camelyon16/`

---

## Step 2: Find CAMELYON16 Training Slides

The page you're on shows mostly CAMELYON17 data (patient_199.zip, etc.).

You need to navigate to find CAMELYON16 slides. Look for:

- **Folder:** CAMELYON16/training/
- **Subfolders:** 
  - `normal/` (contains normal_001.tif, normal_002.tif, etc.)
  - `tumor/` (contains tumor_001.tif, tumor_002.tif, etc.)

---

## Step 3: Download Only What You Need

**For your project, download:**

### Option A: Minimal Test (Start Here - ~5 GB)
```
normal/normal_001.tif
normal/normal_002.tif
tumor/tumor_001.tif
tumor/tumor_002.tif
```

### Option B: Small Training Set (~20 GB)
```
normal/normal_001.tif through normal_010.tif (10 slides)
tumor/tumor_001.tif through tumor_010.tif (10 slides)
```

### Option C: Full Training Set (NOT recommended - ~300 GB)
```
ALL 400 slides - Don't do this unless you have:
- 500 GB free space
- Fast internet
- Serious cloud GPU budget
```

---

## Step 4: After Downloading Slides

Place all .tif files directly in:
```
~/PATHQ.../notebooks/data/camelyon16/
```

Your structure should look like:
```
data/camelyon16/
├── reference.csv
├── lesion_annotations.zip
├── normal_001.tif
├── normal_002.tif
├── tumor_001.tif
└── tumor_002.tif
```

---

## Alternative: Quick Test with Sample Slides

If you can't find CAMELYON16 slides yet, download OpenSlide test slides:

```bash
cd ~/PATHQ.../notebooks/data/camelyon16/
wget https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1-Small-Region.svs
wget https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1.svs
```

These will let you test your pipeline while you wait for real CAMELYON16 data.

---

## Summary

**Right now, download:**
1. ✅ reference.csv (3.56 KB)
2. ✅ lesion_annotations.zip for CAMELYON16 (329 KB)
3. ❌ Skip the large patient_*.zip files (those are CAMELYON17)

**Then find the CAMELYON16 folder** and download 2-4 slides to start testing.
