import pandas as pd, numpy as np, matplotlib.pyplot as plt, joblib, json, yaml, os, argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, RocCurveDisplay, PrecisionRecallDisplay
from pyod.models.auto_encoder import AutoEncoder

cfg = yaml.safe_load(open("/content/manifest.yaml"))
if not os.path.exists(cfg["dataset_path"]):
    import urllib.request
    urllib.request.urlretrieve(cfg["dataset_url"], cfg["dataset_path"])

def train():
    df = pd.read_csv(cfg["dataset_path"])
    X = df.drop(columns=[cfg["target"]]+cfg["drop"], errors="ignore")
    y = df[cfg["target"]]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=cfg["test_size"], random_state=cfg["random_state"], stratify=y)
    scaler = StandardScaler().fit(Xtr)
    Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)

    model = AutoEncoder(contamination=cfg["contamination"], batch_size=cfg["batch_size"], random_state=cfg["random_state"])
    model.fit(Xtr)

    scores = model.decision_function(Xte)
    yhat = (scores >= model.threshold_).astype(int)

    roc = roc_auc_score(yte, scores)
    ap = average_precision_score(yte, scores)
    f1, prec, rec = f1_score(yte, yhat), precision_score(yte, yhat), recall_score(yte, yhat)
    cm = confusion_matrix(yte, yhat)

    joblib.dump(model, cfg["model_path"])
    joblib.dump(scaler, cfg["scaler_path"])
    json.dump({"roc_auc":roc,"ap":ap,"f1":f1,"precision":prec,"recall":rec,"confusion_matrix":cm.tolist()}, open(cfg["metrics_path"],"w"), indent=2)

    RocCurveDisplay.from_predictions(yte, scores); plt.savefig(cfg["roc_path"]); plt.close()
    PrecisionRecallDisplay.from_predictions(yte, scores); plt.savefig(cfg["pr_path"]); plt.close()
    plt.hist(scores, bins=50); plt.savefig(cfg["hist_path"]); plt.close()
    plt.matshow(cm, cmap="Blues"); plt.savefig(cfg["cm_path"]); plt.close()

    print("Training done.")
    print("ROC-AUC:",roc," PR-AUC:",ap," F1:",f1)

def predict():
    model, scaler = joblib.load(cfg["model_path"]), joblib.load(cfg["scaler_path"])
    df = pd.read_csv(cfg["dataset_path"])
    X = df.drop(columns=[cfg["target"]]+cfg["drop"], errors="ignore")
    X = scaler.transform(X)
    scores = model.decision_function(X)
    preds = (scores >= model.threshold_).astype(int)
    df["score"], df["prediction"] = scores, preds
    df.to_csv("predictions.csv", index=False)
    print("Saved predictions.csv")

train()