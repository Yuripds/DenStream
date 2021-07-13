from DenStreamLib import denStream
import pandas as pd


def test_denstream():
    train = pd.read_csv("/home/yuri_pedro/Documentos/DenStream/tests/train.csv", header=None)
    clusterer = denStream.DenStream(lambd=0.1, eps=1.5, beta=0.5, mu=3)
    y = clusterer.fit_predict(train)

   