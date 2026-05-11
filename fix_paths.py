#!/usr/bin/env python3
"""
Quick fix script: Update notebook config to point to actual CAMELYON16 location
Run this BEFORE starting Week 1 notebook
"""
import json
from pathlib import Path

notebook_path = Path('./notebooks/week1_setup_and_data.ipynb')

with open(notebook_path, 'r') as f:
    nb = json.load(f)

# Find and fix the Config cell (Cell 2)
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_text = ''.join(cell['source'])
        if 'CAMELYON16_DIR' in source_text and 'class Config' in source_text:
            # Replace the incorrect path
            old_line = "    CAMELYON16_DIR  = Path('./data/camelyon16')"
            new_line = "    CAMELYON16_DIR  = Path('../camelyon16/images')"

            new_source = source_text.replace(old_line, new_line)
            new_source = new_source.replace(
                "print(f'  Data dir:    {cfg.DATA_DIR.resolve()}')",
                "print(f'  Data dir:    {cfg.DATA_DIR.resolve()}')\n"
                "print(f'  CAMELYON16:  {cfg.CAMELYON16_DIR.resolve()} ')\n"
                "print(f'  Found slides: {len(list(cfg.CAMELYON16_DIR.glob(\"*/*.tif\")))} total')"
            )

            cell['source'] = new_source.split('\n')
            # Re-add newlines
            cell['source'] = [line + '\n' if i < len(cell['source'])-1 else line
                             for i, line in enumerate(cell['source'])]

            print(f"✅ Fixed Config cell")
            break

# Save updated notebook
with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"\n✅ Notebook updated: {notebook_path}")
print(f"\n🎉 Ready to run Week 1!")
print(f"\nNext steps:")
print(f"  1. jupyter notebook notebooks/week1_setup_and_data.ipynb")
print(f"  2. Run all cells (Cells 1-12)")
print(f"  3. Processing will take 2-4 hours for patch extraction")
