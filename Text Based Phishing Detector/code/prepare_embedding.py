from trafilatura import fetch_url, extract
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import time
import torch
import pickle
import sys
from googletrans import Translator




device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)


translator = Translator()

modelRoberta = SentenceTransformer('aditeyabaral/sentencetransformer-xlm-roberta-base')
modelRoberta.to(device)


modelBert = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
modelBert.to(device)

chosen_embedding = sys.argv[1]

# Embeddings and class labels(1 for legitimate, 0 for phishing)
X = np.empty((0, 768))
Y = []





def fill_legitimate_embeddings(x, y):
    calc = 1
    stop = 0
    start = time.time()
    arda = 0
    for root, subfolders, filenames in os.walk(r"../Legitimate"):
        for filename in filenames:
            filepath = os.path.normpath(os.path.join(root, filename))

            if stop == 1000:
                end = time.time()
                print(f"{calc} = {int(end - start)}")
                start = time.time()
                stop = 0
                calc += 1
            #try:
            with open(filepath, "r", encoding="utf-8") as file:
                result = extract(file.read())

                if result is not None and len(result) > 0:
                    if chosen_embedding == "xlm-roberta":
                        embeddings = modelRoberta.encode(result)

                    elif chosen_embedding == "sbert":

                        try:
                            result = translator.translate(result, dest="en").text

                        except Exception as e:

                            result = None

                        if result is not None:
                            embeddings = modelBert.encode(result)
                        else:
                            # Handle the case when translation fails
                            embeddings = modelBert.encode("empty")

                else:
                    if chosen_embedding == "xlm-roberta":
                        embeddings = modelRoberta.encode("empty")

                    elif chosen_embedding == "sbert":
                        embeddings = modelBert.encode("empty")

                x = np.vstack([x, embeddings])
                y.append(1)
            #except Exception as e:
                #print(f"Error processing file '{filepath}': {e}")
                #print(stop)


            stop += 1

    return x, y


X, Y = fill_legitimate_embeddings(X, Y)
print("--------------------Legitimate is done----------------------")


def fill_phishing_embeddings(x, y):
    calc = 1
    start = time.time()
    stop = 0
    for root, subfolders, filenames in os.walk(r"../Phishing"):
        for filename in filenames:
            filepath = os.path.normpath(os.path.join(root, filename))

            if stop == 1000:
                end = time.time()
                print(f"{calc} = {int(end - start)}")
                start = time.time()
                stop = 0
                calc += 1
            try:
                with open(filepath, "r", encoding="latin-1") as file:

                    result = extract(file.read())

                    if result is not None and len(result) > 0:
                        if chosen_embedding == "xlm-roberta":
                            embeddings = modelRoberta.encode(result)


                        elif chosen_embedding == "sbert":

                            try:
                                result = translator.translate(result, dest="en").text

                            except:
                                result = None

                            if result is not None:

                                embeddings = modelBert.encode(result)

                            else:

                                # Handle the case when translation fails

                                embeddings = modelBert.encode("empty")


                    else:
                        if chosen_embedding == "xlm-roberta":
                            embeddings = modelRoberta.encode("empty")

                        elif chosen_embedding == "sbert":
                            embeddings = modelBert.encode("empty")

                    x = np.vstack([x, embeddings])
                    y.append(0)
            except Exception as e:
                print(f"Error processing file '{filepath}': {e}")
                print(stop)

            stop += 1

    return x, y


X, Y = fill_phishing_embeddings(X, Y)

Y = np.array(Y)

x_y = {"X": X, "Y": Y}

#print(x_y)
if chosen_embedding == "xlm-roberta":
    with open("./embeddings/embeddings-xlm-roberta.pkl", "wb") as filee:
        pickle.dump(x_y, filee)

elif chosen_embedding == "sbert":
    with open("./embeddings/embeddings-sbert.pkl", "wb") as filee:
        pickle.dump(x_y, filee)














