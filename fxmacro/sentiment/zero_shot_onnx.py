"""Zero-shot NLI avec le modèle DeBERTa exporté en ONNX (sans torch ni transformers).

Même calcul que zero_shot.py : logit d'implication par hypothèse, puis softmax sur les
étiquettes. Seul le moteur change (ONNX Runtime, bien plus léger que torch).
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


@lru_cache(maxsize=2)
def charger_modele_onnx(dossier: str):
    """(tokenizer, session ONNX, indice de la classe 'entailment')."""
    import onnxruntime as ort
    from tokenizers import Tokenizer

    dossier = Path(dossier)
    tokenizer = Tokenizer.from_file(str(dossier / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=256)
    tokenizer.enable_padding()
    # Réglages pour une petite machine : pas de copies supplémentaires des poids en mémoire
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2  # la VM a 2 vCPU
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.add_session_config_entry("session.disable_prepacking", "1")
    session = ort.InferenceSession(str(dossier / "model.onnx"), options, providers=["CPUExecutionProvider"])
    etiquettes = json.loads((dossier / "etiquettes.json").read_text(encoding="utf-8"))
    return tokenizer, session, etiquettes["label2id"]["entailment"]


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)  # stabilité numérique : exp de grands nombres
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def scorer_zero_shot_onnx(phrases: list[str], hypotheses: dict[str, str], dossier_modele: str,
                          taille_lot: int = 16) -> pd.DataFrame:
    """Probabilités p_<étiquette> pour chaque phrase (une colonne par hypothèse)."""
    tokenizer, session, indice_implication = charger_modele_onnx(str(dossier_modele))
    etiquettes = list(hypotheses)
    paires = [(phrase, hypotheses[e]) for phrase in phrases for e in etiquettes]

    logits_implication = []
    for i in range(0, len(paires), taille_lot):
        encodages = tokenizer.encode_batch(paires[i:i + taille_lot])
        entrees = {"input_ids": np.array([e.ids for e in encodages], dtype=np.int64),
                   "attention_mask": np.array([e.attention_mask for e in encodages], dtype=np.int64)}
        logits = session.run(["logits"], entrees)[0]
        logits_implication.append(logits[:, indice_implication])

    z = np.concatenate(logits_implication).reshape(len(phrases), len(etiquettes))
    return pd.DataFrame(softmax(z), columns=[f"p_{e}" for e in etiquettes])
