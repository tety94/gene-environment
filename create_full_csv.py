import os
from glob import glob
from config import OUTPUT_FOLDER

merged_folder = OUTPUT_FOLDER
output_file = os.path.join(merged_folder, "full_chr.csv")

# elimina se esiste
if os.path.exists(output_file):
    os.remove(output_file)

csv_files = sorted([
    f for f in glob(os.path.join(merged_folder, "chr*_merged.csv"))
    if not f.endswith("full_chr.csv")
])


#check per evitare problemi
def get_all_ids(fpath):
    with open(fpath) as f:
        next(f)  # salta header
        return [line.split(",")[0] for line in f]

ref_ids = get_all_ids(csv_files[0])

for f in csv_files[1:]:
    ids = get_all_ids(f)
    if ids != ref_ids:
        raise RuntimeError(f"❌ ID mismatch in {f}")




print(f"🔹 Unione testuale di {len(csv_files)} file")

# ---- HEADER ----
with open(csv_files[0], "r") as f:
    header = f.readline().strip()

for fpath in csv_files[1:]:
    with open(fpath, "r") as f:
        cols = f.readline().strip().split(",")[1:]  # salta ID
        header += "," + ",".join(cols)

with open(output_file, "w") as out:
    out.write(header + "\n")

# ---- RIGHE ----
files = [open(f, "r") for f in csv_files]

# salta header
for f in files:
    f.readline()

with open(output_file, "a") as out:
    for rows in zip(*files):
        base = rows[0].strip()
        rest = []
        for r in rows[1:]:
            rest.append(",".join(r.strip().split(",")[1:]))
        out.write(base + "," + ",".join(rest) + "\n")

for f in files:
    f.close()

print(f"✅ File finale creato: {output_file}")
