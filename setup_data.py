import os

print("Downloading dataset from Kaggle...")
os.system("pip install kaggle")
os.system("kaggle datasets download -d datamunge/sign-language-mnist --unzip -p data/")
print("Finished: Data is in the /data folder.")
