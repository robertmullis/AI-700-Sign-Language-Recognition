import os
import shutil

print("Downloading datasets from Kaggle...")

# Install kaggle if not already installed
if os.system("pip show kaggle") != 0:
    os.system("pip install kaggle")

# Download Sign MNIST
if not os.path.exists("data/sign_mnist_train.csv") or not os.path.exists("data/sign_mnist_test.csv"):
    os.system("kaggle datasets download -d datamunge/sign-language-mnist --unzip -p data/")
    print("Sign MNIST downloaded.")
else:
    print("Sign MNIST already exists, skipping.")

# Download ASL Alphabet
if not os.path.exists("data/asl_alphabet_train/asl_alphabet_train"):
    os.system("kaggle datasets download -d grassknoted/asl-alphabet --unzip -p data/")
    print("ASL Alphabet downloaded.")
else:
    print("ASL Alphabet already exists, skipping.")

# Remove extra ASL folders that don't correspond to letters
print("Cleaning up extra ASL folders...")
for folder in ['del', 'nothing', 'space', 'J', 'Z']:
    path = os.path.join("data/asl_alphabet_train/asl_alphabet_train", folder)
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"  Deleted {folder}")

# Verify
folders = os.listdir("data/asl_alphabet_train/asl_alphabet_train")
print(f"ASL folders remaining: {len(folders)} — {sorted(folders)}")
print("Finished: Data is ready.")