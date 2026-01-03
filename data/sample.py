from ultralytics import YOLO

MODEL_PATH = "capmodelnew.pt"
OUTPUT_FILE = "data.yaml"

# Load YOLO model
model = YOLO(MODEL_PATH)

# Extract class names in correct order
names_dict = model.names
names_list = [names_dict[i] for i in range(len(names_dict))]
nc = len(names_list)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    # ---- KEEP THIS PART EXACT ----
    f.write("train: ../train/images\n")
    f.write("val: ../valid/images\n")
    f.write("test: ../test/images\n\n")

    # ---- AUTO-GENERATED PART ----
    f.write(f"nc: {nc}\n")
    f.write("names: [\n")
    for name in names_list:
        f.write(f"'{name}',\n")
    f.write("]\n\n")

    # ---- KEEP THIS PART EXACT ----
    f.write("roboflow:\n")
    f.write("  workspace: imageinsight-wwdyd\n")
    f.write("  project: orgi-imageinsight\n")
    f.write("  version: 5\n")
    f.write("  license: Private\n")
    f.write("  url: https://app.roboflow.com/imageinsight-wwdyd/orgi-imageinsight/5\n")

print(f"✅ data.yaml created successfully with {nc} classes")
