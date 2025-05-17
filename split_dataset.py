import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def split_dataset(
    input_dir,
    output_dir,
    test_size=0.2,
    random_state=42,
    valid_dir_name="valid",
    train_dir_name="train"
):
    class_names = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    for class_name in class_names:
        class_path = os.path.join(input_dir, class_name)
        image_files = [
            f for f in os.listdir(class_path)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]

        if len(image_files) < 2:
            print(f"[WARNING] Skipping class '{class_name}' (not enough images to split).")
            continue

        train_files, valid_files = train_test_split(
            image_files,
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )

        train_class_dir = os.path.join(output_dir, train_dir_name, class_name)
        valid_class_dir = os.path.join(output_dir, valid_dir_name, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(valid_class_dir, exist_ok=True)

        for fname in tqdm(train_files, desc=f"Train {class_name:>5}"):
            src = os.path.join(class_path, fname)
            dst = os.path.join(train_class_dir, fname)
            shutil.copy2(src, dst)

        for fname in tqdm(valid_files, desc=f"Valid {class_name:>5}"):
            src = os.path.join(class_path, fname)
            dst = os.path.join(valid_class_dir, fname)
            shutil.copy2(src, dst)

    print("\nDataset split completed successfully.")

if __name__ == "__main__":
    INPUT_DIR = ""           
    OUTPUT_DIR = ""         
    TEST_SIZE = 0.2                          
    RANDOM_STATE = 42                        

    split_dataset(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
