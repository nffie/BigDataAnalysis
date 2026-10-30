# ------------- ✦✦✦ IMPS ✦✦✦ -------------
# RuBERT + две головы (бинарная и многоклассовая), комбинированная функция потерь 0.3/0.7.
# Пометки [К1]–[К14] в комментариях отмечают исправления относительно первой версии кода.
import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from data_utils import (
    CLASS_NAMES,
    compute_metrics,
    evaluation_report,
    load_data,
    save_json,
    save_quarantine,
    split_fingerprint,
    split_records,
)

# ------------- ✦✦✦ LOG ✦✦✦ ------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------- ✦✦✦ HELPERS ✦✦✦ -------------
# ------------- ✦✦ DATA SET UP ✦✦ -------------
class VulnerabilityDataset(Dataset):
    def __init__(self, texts, is_vul_labels, vul_class_labels, tokenizer, max_length=512):
        self.texts = texts
        self.is_vul_labels = is_vul_labels
        self.vul_class_labels = vul_class_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # [К10] без padding="max_length": батч дополняется до самой длинной
        # последовательности в collate_batch (тексты короче 512 токенов, обучение быстрее)
        encoding = self.tokenizer(str(self.texts[idx]), truncation=True,
                                  max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "is_vul_label": torch.tensor(self.is_vul_labels[idx], dtype=torch.float),
            "vul_class_label": torch.tensor(self.vul_class_labels[idx], dtype=torch.long),
        }


def make_collate(pad_token_id):
    def collate_batch(batch):
        return {
            "input_ids": pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=pad_token_id),
            "attention_mask": pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0),
            "is_vul_label": torch.stack([b["is_vul_label"] for b in batch]),
            "vul_class_label": torch.stack([b["vul_class_label"] for b in batch]),
        }
    return collate_batch


# ------------- ✦✦ SET UP ✦✦ -------------
class VulnerabilityClassifier(nn.Module):

    # ------------- ✦ ARCHITECTURE ✦ -------------
    def __init__(self, model_name, num_classes=3, dropout_rate=0.3, bert=None):
        super().__init__()
        # bert передаётся при загрузке сохранённой модели (без скачивания весов)
        self.bert = bert if bert is not None else AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)

        # ------------- ✦ BINARY HEAD [vul/no-vul] ✦ -------------
        # [К6] без Sigmoid: голова возвращает логит, сигмоида — в BCEWithLogitsLoss и в predict
        self.binary_head = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
        )

        # ------------- ✦ MULTYCLASS HEAD [vul class] ✦ -------------
        self.multi_class_head = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        # ------------- ✦ EMBEDDING ✦ -------------
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        # ------------- ✦ DROPOUT ✦ -------------
        pooled_output = self.dropout(pooled_output)

        # ------------- ✦ FORECAST ✦ -------------
        # [К6] squeeze(-1) вместо squeeze(): при батче из 1 примера форма не схлопывается в скаляр
        binary_logit = self.binary_head(pooled_output).squeeze(-1)
        multi_class_pred = self.multi_class_head(pooled_output)

        return binary_logit, multi_class_pred


# ------------- ✦✦ PIPLINE ✦✦ -------------
class VulnerabilityDetectionPipeline:
    # ------------- ✦ PIPLINE ✦ -------------
    def __init__(self, model_name="DeepPavlov/rubert-base-cased", max_length=512, tokenizer=None):
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.threshold = 0.5

        # ------------- ✦ AVAILABLE CLASSES ✦ -------------
        # [К3] три типа из поля «Класс уязвимости»; индекс = выход многоклассовой головы.
        # В исходной версии было 4 имени при 3 выходах, и «многофакторная» попадала в «None».
        self.class_mapping = dict(enumerate(CLASS_NAMES))
        self.reverse_class_mapping = {v: k for k, v in self.class_mapping.items()}

    # ------------- ✦ LOAD AND PREPROCCESS DATA ✦ -------------
    def load_and_preprocess_data(self, non_vul_file, vul_csv_file, quarantine_file=None):
        # [К1][К2][К4] CSV читается CSV-парсером (кавычки, многострочные поля, без «;» в тексте),
        # пустые записи и дубликаты удаляются, записи без метки и с противоречивыми
        # метками — в карантин (метки не угадываются). Реализация — data_utils.load_data.
        records, report = load_data(vul_csv_file, non_vul_file)
        logger.info(f"Очистка данных: {json.dumps(report['counts'], ensure_ascii=False)}")
        if quarantine_file:
            save_quarantine(report, quarantine_file)
        self.data_report = report["counts"]
        return records

    # ------------- ✦ PREPARE DATA ✦ -------------
    def prepare_datasets(self, records):
        """Подготовка датасетов для обучения"""
        # [К5] 70/10/20 со стратификацией по 4 итоговым классам; почти-дубликаты
        # (тексты, отличающиеся цифрами/пунктуацией) не разносятся по разным выборкам
        train, val, test = split_records(records)
        # отпечаток разбиения: одинаковое значение означает одинаковый состав выборок
        self.split_fingerprint = split_fingerprint(train, val, test)
        self.test_records = test
        # длина текстов в токенах: проверка утверждения статьи о покрытии max_length
        lengths = np.array([len(ids) for ids in self.tokenizer([r["text"] for r in records])["input_ids"]])
        self.token_stats = {"median": float(np.median(lengths)), "p95": float(np.percentile(lengths, 95)),
                            "max": int(lengths.max()), "share_truncated": float(np.mean(lengths > self.max_length))}
        logger.info(f"Длина текстов в токенах: {self.token_stats}")

        def to_ds(part):
            return VulnerabilityDataset([r["text"] for r in part], [r["is_vul"] for r in part],
                                        [r["vul_class"] for r in part], self.tokenizer, self.max_length)

        train_dataset, val_dataset, test_dataset = to_ds(train), to_ds(val), to_ds(test)
        logger.info(f"Размеры датасетов - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        return train_dataset, val_dataset, test_dataset

    def make_loader(self, dataset, batch_size, shuffle=False):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          collate_fn=make_collate(self.tokenizer.pad_token_id))

    # ------------- ✦ TRAIN MODEL ✦ -------------
    def train_model(self, train_dataset, val_dataset, epochs=10, batch_size=16, learning_rate=2e-5,
                    warmup_ratio=0.1, max_patience=3, checkpoint="best_vulnerability_model.pth"):

        # ------------- ✦ INIT MODEL ✦ -------------
        self.model = VulnerabilityClassifier(self.model_name, num_classes=len(CLASS_NAMES)).to(self.device)

        # ------------- ✦ CREATE DATA LOADERS ✦ ------------
        train_loader = self.make_loader(train_dataset, batch_size, shuffle=True)
        val_loader = self.make_loader(val_dataset, batch_size)

        # ------------- ✦ LOSS FUNC OPTIMIZER ✦ -------------
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        # [К9] линейный warmup (10 % шагов) и линейное затухание — как заявлено в статье
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, int(warmup_ratio * total_steps), total_steps)

        # [К7] веса классов по обучающей выборке: при сбалансированных классах равны 1
        is_vul = np.array(train_dataset.is_vul_labels)
        cls = np.array(train_dataset.vul_class_labels)[is_vul == 1]
        counts = np.bincount(cls, minlength=len(CLASS_NAMES)).astype(float)
        class_weights = counts.sum() / (len(CLASS_NAMES) * np.maximum(counts, 1))
        pos_weight = (is_vul == 0).sum() / max((is_vul == 1).sum(), 1)
        binary_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float, device=self.device))
        multi_class_criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float, device=self.device))
        self.train_config = {
            "model_name": self.model_name, "epochs": epochs, "batch_size": batch_size,
            "learning_rate": learning_rate, "weight_decay": 0.01, "warmup_ratio": warmup_ratio,
            "max_length": self.max_length, "dropout": 0.3, "loss": "0.3*BCEWithLogits + 0.7*CE",
            "pos_weight": float(pos_weight), "class_weights": class_weights.round(4).tolist(),
            "early_stopping": f"val e2e macro-F1, patience={max_patience}",
        }

        best_val_f1 = -1.0
        patience_counter = 0
        self.history = []

        logger.info("Начало обучения...")

        for epoch in range(epochs):
            # ------------- ✦ TRAIN ✦ -------------
            self.model.train()
            total_loss = 0
            batch_count = 0

            for batch in train_loader:
                optimizer.zero_grad()

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                is_vul_labels = batch["is_vul_label"].to(self.device)
                vul_class_labels = batch["vul_class_label"].to(self.device)

                # ------------- ✦ FORWARD GO ✦ -------------
                binary_logit, multi_class_pred = self.model(input_ids, attention_mask)

                # ------------- ✦ LOSSES ✦ -------------
                binary_loss = binary_criterion(binary_logit, is_vul_labels)

                # ------------- ✦ LOSS: ONLY [+] FOR MULTYCLASS ✦ -------------
                mask = is_vul_labels == 1
                if mask.sum() > 0:
                    # ------------- ✦ FILTER ONLY [+] FOR MULTYCLASS ✦ -------------
                    pos_multi_pred = multi_class_pred[mask]
                    pos_vul_class_labels = vul_class_labels[mask]

                    multi_class_loss = multi_class_criterion(pos_multi_pred, pos_vul_class_labels)
                    total_loss_epoch = 0.3 * binary_loss + 0.7 * multi_class_loss
                else:
                    total_loss_epoch = binary_loss

                # ------------- ✦ REVERSE GO ✦ -------------
                total_loss_epoch.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += total_loss_epoch.item()
                batch_count += 1

            # ------------- ✦ VALIDATION ✦ -------------
            val_metrics = self.evaluate_model(val_loader)

            logger.info(f"Эпоха {epoch+1}/{epochs}")
            logger.info(f"Train Loss: {total_loss/batch_count:.4f}")
            logger.info(f"Val Binary macro-F1: {val_metrics['binary_f1']:.4f}")
            logger.info(f"Val Multi macro-F1: {val_metrics['multi_f1']:.4f}")
            logger.info(f"Val E2E macro-F1: {val_metrics['e2e_f1']:.4f}")
            self.history.append({"epoch": epoch + 1, "train_loss": total_loss / batch_count,
                                 **{k: val_metrics[k] for k in ("binary_f1", "multi_f1", "e2e_f1")}})

            # ------------- ✦ ERALY STOPPING ✦ -------------
            # [К8] критерий — сквозная macro-F1 на val (F1 класса «уязвимость» ≈ 1 с первой эпохи)
            if val_metrics["e2e_f1"] > best_val_f1:
                best_val_f1 = val_metrics["e2e_f1"]
                patience_counter = 0
                # ------------- ✦ BEST MODEL SAVE ✦ -------------
                torch.save(self.model.state_dict(), checkpoint)
            else:
                patience_counter += 1

            if patience_counter >= max_patience:
                logger.info("Ранняя остановка обучения")
                break

        # ------------- ✦ BEST MODEL UPLOAD ✦ -------------
        self.model.load_state_dict(torch.load(checkpoint, map_location=self.device, weights_only=True))
        logger.info("Обучение завершено")

    # ------------- ✦ EVAL MODEL ✦ -------------
    def evaluate_model(self, data_loader, records=None):
        """Без records — краткие метрики (для val). С records (те же записи и в том же порядке,
        что в data_loader) — полный отчёт data_utils.evaluation_report: метрики трёх уровней,
        показатели по классам, матрицы ошибок и 95 % бутстреп-интервалы."""
        self.model.eval()

        probs, type_preds, is_vul_true, type_true = [], [], [], []
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                binary_logit, multi_class_pred = self.model(input_ids, attention_mask)
                probs.extend(torch.sigmoid(binary_logit).cpu().tolist())
                type_preds.extend(multi_class_pred.argmax(-1).cpu().tolist())
                is_vul_true.extend(batch["is_vul_label"].int().tolist())
                type_true.extend(batch["vul_class_label"].tolist())

        # ------------- ✦ GETTING METRICS ✦ -------------
        # [К11] macro-F1 по полному списку классов (исходный compute_f1 подменял
        # многоклассовую F1 бинарной, если в выборке встречались только 2 типа)
        is_vul_pred = [int(p >= self.threshold) for p in probs]
        m = compute_metrics(is_vul_true, is_vul_pred, type_true, type_preds)
        if records is not None:
            assert [r["is_vul"] for r in records] == is_vul_true, "порядок записей не совпадает с data_loader"
            return evaluation_report(records, np.array(is_vul_pred), np.array(type_preds))
        return {"binary_f1": m["binary"]["macro_f1"], "multi_f1": m["type"]["macro_f1"], "e2e_f1": m["e2e"]["macro_f1"]}

    # ------------- ✦ SAVE / LOAD ✦ -------------
    # [К13] модель сохраняется вместе с конфигурацией энкодера и токенизатором,
    # поэтому предсказание работает без повторного обучения и без интернета
    def save_model(self, model_dir="vulnerability_classifier_model"):
        d = Path(model_dir)
        d.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": self.model.state_dict(), "model_name": self.model_name,
                    "class_mapping": self.class_mapping, "max_length": self.max_length,
                    "threshold": self.threshold}, d / "model.pth")
        self.model.bert.config.save_pretrained(d / "bert_config")
        self.tokenizer.save_pretrained(d / "tokenizer")
        logger.info(f"Модель сохранена в '{d}'")

    @classmethod
    def load(cls, model_dir="vulnerability_classifier_model"):
        d = Path(model_dir)
        ckpt = torch.load(d / "model.pth", map_location="cpu", weights_only=True)
        pipe = cls(ckpt["model_name"], ckpt["max_length"], tokenizer=AutoTokenizer.from_pretrained(d / "tokenizer"))
        bert = AutoModel.from_config(AutoConfig.from_pretrained(d / "bert_config"))
        pipe.model = VulnerabilityClassifier(ckpt["model_name"], len(CLASS_NAMES), bert=bert)
        pipe.model.load_state_dict(ckpt["model_state_dict"])
        pipe.model.to(pipe.device).eval()
        pipe.threshold = ckpt["threshold"]
        return pipe

    # ------------- ✦ PREDICTIONS ✦ -------------
    def predict(self, texts):
        if self.model is None:
            raise ValueError("Модель не обучена и не загружена")

        self.model.eval()

        results = []

        with torch.no_grad():
            for text in texts:
                # ------------- ✦ PREDICTION ✦ -------------
                encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_tensors="pt")

                input_ids = encoding["input_ids"].to(self.device)
                attention_mask = encoding["attention_mask"].to(self.device)

                # ------------- ✦ PREDICTION ✦ -------------
                binary_logit, multi_class_pred = self.model(input_ids, attention_mask)

                prob_vul = torch.sigmoid(binary_logit).item()
                is_vul = prob_vul >= self.threshold

                result = {
                    "text": text,
                    "is_vul": is_vul,
                    "proba_vul": prob_vul,
                    "vul_class": None,
                    "proba_class": {},
                }

                if is_vul:
                    # ------------- ✦ MULTYCLASS PROBS ✦ -------------
                    # [К12] индексы вероятностей совпадают с class_mapping (исходно — IndexError)
                    class_probs = torch.softmax(multi_class_pred[0], dim=0).cpu().numpy()
                    result["vul_class"] = self.class_mapping[int(class_probs.argmax())]
                    for i, class_name in self.class_mapping.items():
                        result["proba_class"][class_name] = float(class_probs[i])

                results.append(result)

        return results


def print_predictions(predictions):
    print("\n------------- ✦ РЕЗУЛЬТАТЫ ✦ -------------")
    for pred in predictions:
        print(f"\nТекст: {pred['text'][:80]}...")
        print(f"Уязвимость: {'Да' if pred['is_vul'] else 'Нет'}")
        print(f"Вероятность уязвимости: {pred['proba_vul']:.3f}")
        if pred["vul_class"]:
            print(f"Класс уязвимости: {pred['vul_class']}")
            print("Вероятности по классам (при условии, что это уязвимость):")
            for class_name, prob in pred["proba_class"].items():
                print(f"  {class_name}: {prob:.3f}")


# ------------- ✦✦✦ MAIN ✦✦✦ -------------
def main():
    ap = argparse.ArgumentParser(description="RuBERT: определение и классификация описаний уязвимостей")
    ap.add_argument("mode", nargs="?", choices=["train", "predict"], default="train")
    ap.add_argument("--model-name", default="DeepPavlov/rubert-base-cased")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--model-dir", default="vulnerability_classifier_model")
    ap.add_argument("--text", action="append", help="текст для predict (можно несколько)")
    ap.add_argument("--input", help="файл для predict: один текст на строку")
    args = ap.parse_args()

    # ------------- ✦ PREDICT WITHOUT RETRAINING ✦ -------------
    if args.mode == "predict":
        classifier = VulnerabilityDetectionPipeline.load(args.model_dir)
        texts = list(args.text or [])
        if args.input:
            texts += [x for x in Path(args.input).read_text(encoding="utf-8").splitlines() if x.strip()]
        print_predictions(classifier.predict(texts))
        return

    # ------------- ✦ 1. INIT ✦ -------------
    set_seed(42)
    classifier = VulnerabilityDetectionPipeline(args.model_name, args.max_length)

    # ------------- ✦ 2. DATA IMP ✦ -------------
    # [К14] без «демо-данных» при отсутствии файлов: ошибка вместо обучения на 6 примерах
    records = classifier.load_and_preprocess_data(
        non_vul_file="data/non-vul-desc.txt",
        vul_csv_file="data/vul-desc-class.csv",
        quarantine_file="results/quarantine.csv",
    )

    # ------------- ✦ 3. DATA SET UP ✦ -------------
    train_dataset, val_dataset, test_dataset = classifier.prepare_datasets(records)

    # ------------- ✦ 4. TRAIN MODEL ✦ -------------
    # [К9] параметры совпадают со статьёй: 10 эпох, batch 16, lr 2e-5
    classifier.train_model(train_dataset, val_dataset, epochs=args.epochs,
                           batch_size=args.batch_size, learning_rate=args.lr)

    # ------------- ✦ 5. CHECK ON TEST SET ✦ -------------
    ev = classifier.evaluate_model(classifier.make_loader(test_dataset, 64), records=classifier.test_records)
    t = ev["test"]
    logger.info(f"Финальные метрики (test) - Binary macro-F1: {t['binary_f1']:.4f}, "
                f"Multi macro-F1: {t['multi_f1']:.4f}, E2E macro-F1: {t['e2e_f1']:.4f}")
    save_json({"model": "RuBERT (main.py)",
               "data_report": classifier.data_report, "token_stats": classifier.token_stats,
               "split_sizes": {"train": len(train_dataset), "val": len(val_dataset), "test": len(test_dataset)},
               "split_fingerprint": classifier.split_fingerprint, "config": classifier.train_config,
               "history": classifier.history, **ev}, "results/rubert_metrics.json")

    # ------------- ✦ 6. MY DATA SUPPLY ✦ -------------
    demo_texts = [
        "Система логирования обрабатывает JSON события с ежедневной ротацией файлов",  # НЕ уязвимость
        "Текстовый редактор содержит уязвимость переполнения буфера при обработке файлов",  # Уязвимость. Уязвимость кода
        "API endpoint с аутентификацией через OAuth 2.0",  # НЕ уязвимость
        "Микропрограммное обеспечение содержит предустановленные пароли по умолчанию",  # Уязвимость. Уязвимость архитектуры
    ]

    # ------------- ✦ 7. PREDICTIONS ✦ -------------
    print_predictions(classifier.predict(demo_texts))

    # ------------- ✦ 9. SAVE MODEL ✦ -------------
    classifier.save_model(args.model_dir)


# ------------- ✦✦✦ INIT MAIN ✦✦✦ -------------
if __name__ == "__main__":
    main()
