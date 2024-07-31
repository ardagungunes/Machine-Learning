import os
import shutil

path_legitimate = "./Legitimate"
path_phishing = "./Phishing"


if not os.path.exists(path_legitimate):
    os.mkdir(path_legitimate)

else:
    pass

if not os.path.exists(path_phishing):
    os.mkdir(path_phishing)

else:
    pass

a = 0
file_counter = 0

for root, subfolders, filenames in os.walk("benign_25k"):
    for filename in filenames:
        filepath = os.path.join(root, filename)

        if filename == "html.txt":

            shutil.move(filepath, os.path.join("Legitimate", f"{file_counter}_{filename}"))
            file_counter += 1

file_counter = 0

for root, subfolders, filenames in os.walk("phish_sample_30k"):
    for filename in filenames:
        filepath = os.path.join(root, filename)

        if filename == "html.txt":

            shutil.move(filepath, os.path.join("Phishing", f"{file_counter}_{filename}"))
            file_counter += 1



