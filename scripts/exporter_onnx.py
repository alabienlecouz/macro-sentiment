"""Conversion du modèle zero-shot DeBERTa en ONNX (à lancer une fois, sur le Mac).

Pourquoi ONNX : sur la VM, on remplace torch (~700 Mo installé, ~300 Mo en mémoire) par
ONNX Runtime (~50 Mo). Le calcul est identique (mêmes poids, float32).

Pourquoi pas de quantification int8 : testée sur les 45 phrases annotées, elle fait
chuter l'exactitude de 82 % à 22 % (58 % au mieux, par canal). DeBERTa-v3 a des valeurs
d'activation extrêmes que 8 bits ne représentent pas assez finement.

Les poids sont écrits dans un fichier séparé (model.onnx.data) : ONNX Runtime peut alors
les mapper en mémoire au lieu de les copier, ce qui réduit le pic de RAM au chargement.

Sortie : modeles/deberta_zeroshot_onnx/ (model.onnx, model.onnx.data, tokenizer.json, etiquettes.json)
Usage : python scripts/exporter_onnx.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import onnx
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from fxmacro.config import RACINE_PROJET, charger_config


def exporter(nom_modele: str, dossier: Path) -> Path:
    dossier.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(nom_modele)
    modele = AutoModelForSequenceClassification.from_pretrained(nom_modele).float().eval()
    exemple = tokenizer(["The bank raised rates."], ["This is hawkish."], return_tensors="pt")

    chemin_temp = dossier / "model_temp.onnx"
    torch.onnx.export(
        modele, (exemple["input_ids"], exemple["attention_mask"]), str(chemin_temp),
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "lot", 1: "longueur"}, "attention_mask": {0: "lot", 1: "longueur"},
                      "logits": {0: "lot"}},
        opset_version=17, dynamo=False,
    )
    graphe = onnx.load(str(chemin_temp))
    chemin = dossier / "model.onnx"
    onnx.save_model(graphe, str(chemin), save_as_external_data=True, all_tensors_to_one_file=True,
                    location="model.onnx.data", size_threshold=1024)
    chemin_temp.unlink()

    tokenizer.backend_tokenizer.save(str(dossier / "tokenizer.json"))
    (dossier / "etiquettes.json").write_text(json.dumps(
        {"modele_source": nom_modele, "label2id": modele.config.label2id}, indent=2), encoding="utf-8")
    return chemin


if __name__ == "__main__":
    params = charger_config()["sentiment"]
    chemin = exporter(params["modele_zero_shot"], RACINE_PROJET / params["dossier_onnx"])
    taille = sum(f.stat().st_size for f in chemin.parent.iterdir()) / 1e6
    print(f"Modèle ONNX : {chemin.parent} ({taille:.0f} Mo)")
