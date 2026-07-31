"""
evaluation.py  —  ASAMA 3: Degerlendirme & Karsilastirma (fonksiyon kutuphanesi)
================================================================================
Egitilmis modelleri test seti uzerinde degerlendirir ve tum ciktilari uretir:
  - Metrikler: Accuracy, Precision, Recall, F1-Score, ROC-AUC
  - Confusion Matrix (her model)
  - ROC Curve (tum modeller tek grafikte)
  - Feature Importance (Random Forest ve boosting modeli)
  - Uc modeli tek karsilastirma tablosunda birlestirme

Neden bu metrikler?
- Accuracy tek basina dengesiz veride yaniltir (cogunluk sinifini tahmin edip
  yuksek accuracy alinabilir). Bu yuzden Precision/Recall/F1 ve ozellikle
  ROC-AUC birlikte raporlanir.
- Recall (Dropout icin) egitim baglaminda kritik: terk edecek ogrenciyi
  kacirmamak (yanlis negatifi azaltmak) mudahale acisindan onemlidir.
- ROC-AUC esik-bagimsiz genel ayirt etme gucunu olcer; modelleri adil kiyaslar.
- PR-AUC (Precision-Recall egrisi alani): Dengesiz veride ROC-AUC'tan daha
  bilgilendiricidir cunku dogrudan azinlik (pozitif) sinifina odaklanir.
- 5-katli Cross-Validation: Tek bir test setine degil, verinin tamamina dayali
  daha guvenilir bir performans tahmini ve modelin kararliligini (std) verir.
- Egitim vs Test skoru: Ikisi arasindaki buyuk fark overfitting (asiri uyum)
  isaretidir; bu kontrol modelin genellenebilirligini kanitlar.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, roc_curve,
    precision_recall_curve, classification_report,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

import config


# ===========================================================================
# METRIK HESAPLAMA
# ===========================================================================
def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    """
    Tek bir model icin tum metrikleri hesaplar ve bir sozluk dondurur.

    - predict:  sinif tahmini (0/1) -> Accuracy/Precision/Recall/F1 icin.
    - predict_proba[:,1]: pozitif sinif (Dropout) olasiligi -> ROC-AUC ve
      PR-AUC icin. Bu metrikler olasilik gerektirir; sinif etiketi yeterli degil.

    pos_label=1 -> Precision/Recall/F1'i POZITIF sinif (Dropout) icin hesaplar,
    cunku ilgilendigimiz olay terk tespitidir.

    Overfitting kontrolu: Egitim ve test dogrulugu birlikte hesaplanir. Ikisi
    arasindaki fark (Train-Test Gap) buyukse model ezberlemis (overfit) demektir.
    """
    # Test seti tahminleri.
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Egitim seti dogrulugu (overfitting kontrolu icin).
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, y_pred)

    metrics = {
        "Model": name,
        "Accuracy": test_acc,
        "Precision": precision_score(y_test, y_pred, pos_label=1),
        "Recall": recall_score(y_test, y_pred, pos_label=1),
        "F1-Score": f1_score(y_test, y_pred, pos_label=1),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "PR-AUC": average_precision_score(y_test, y_proba),  # dengesiz veri icin
        "Train-Acc": train_acc,                              # overfitting kontrolu
        "Overfit-Gap": train_acc - test_acc,                 # fark = overfit isareti
    }

    # Konsola detayli siniflandirma raporu da yazdiralim (per-sinif detay).
    print(f"\n--- {name} — Siniflandirma Raporu (test seti) ---")
    print(classification_report(
        y_test, y_pred,
        target_names=[config.NEGATIVE_CLASS, config.POSITIVE_CLASS],
        digits=3,
    ))
    print(f"    Egitim dogrulugu: {train_acc:.3f} | Test dogrulugu: {test_acc:.3f} "
          f"| Fark (overfit): {train_acc - test_acc:.3f}")
    return metrics, y_pred, y_proba


# ===========================================================================
# CONFUSION MATRIX
# ===========================================================================
def plot_confusion_matrix(name, y_test, y_pred):
    """
    Karisiklik matrisini isi haritasi olarak kaydeder.
    Satirlar gercek sinif, sutunlar tahmin. Kosegen dogru tahminlerdir.
    Yanlis negatif (gercek Dropout ama Graduate tahmin) hucresi, egitim
    baglaminda en maliyetli hatadir; bu gorsel onu acikca gosterir.
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=[config.NEGATIVE_CLASS, config.POSITIVE_CLASS],
                yticklabels=[config.NEGATIVE_CLASS, config.POSITIVE_CLASS])
    plt.title(f"Confusion Matrix — {name}")
    plt.ylabel("Gercek Sinif")
    plt.xlabel("Tahmin Edilen Sinif")
    plt.tight_layout()
    # Dosya adinda bosluklari alt cizgi yap (temiz dosya adi).
    fname = f"cm_{name.lower().replace(' ', '_')}.png"
    out = config.FIGURES_DIR / fname
    plt.savefig(out, dpi=config.FIGURE_DPI)
    plt.close()
    print(f"  [Gorsel] Kaydedildi: {out.name}")


# ===========================================================================
# ROC CURVE (tum modeller tek grafikte)
# ===========================================================================
def plot_roc_curves(roc_data, y_test):
    """
    Tum modellerin ROC egrilerini tek grafikte cizer.
    roc_data: {model_adi: y_proba} sozlugu.

    ROC egrisi, farkli esiklerde True Positive Rate'e karsi False Positive
    Rate'i gosterir. Egri ne kadar sol-ust koseye yakinsa model o kadar iyidir.
    Kosegen (0.5 AUC) rastgele tahmin cizgisidir; referans olarak eklenir.
    """
    plt.figure(figsize=(7, 6))
    for name, y_proba in roc_data.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc:.3f})")

    # Rastgele tahmin referans cizgisi.
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Rastgele (AUC = 0.500)")
    plt.title("ROC Egrileri — Model Karsilastirmasi")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    out = config.FIGURES_DIR / "roc_curves_comparison.png"
    plt.savefig(out, dpi=config.FIGURE_DPI)
    plt.close()
    print(f"  [Gorsel] Kaydedildi: {out.name}")


# ===========================================================================
# PRECISION-RECALL CURVE (tum modeller tek grafikte)
# ===========================================================================
def plot_pr_curves(roc_data, y_test):
    """
    Tum modellerin Precision-Recall egrilerini tek grafikte cizer.
    roc_data: {model_adi: y_proba} sozlugu (ROC ile ayni olasiliklar).

    Neden PR egrisi? Dengesiz veride ROC egrisi fazla iyimser gorunebilir.
    PR egrisi dogrudan pozitif sinifin (Dropout) precision/recall dengesini
    gosterir; azinlik sinifi performansini daha durust yansitir.
    Referans cizgisi = pozitif sinif oranidir (rastgele siniflandiricinin
    ulasacagi precision seviyesi).
    """
    plt.figure(figsize=(7, 6))
    for name, y_proba in roc_data.items():
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        plt.plot(recall, precision, linewidth=2, label=f"{name} (PR-AUC = {ap:.3f})")

    # Rastgele referans: pozitif sinif orani.
    baseline = y_test.mean()
    plt.axhline(y=baseline, color="k", linestyle="--", linewidth=1,
                label=f"Rastgele (={baseline:.3f})")
    plt.title("Precision-Recall Egrileri — Model Karsilastirmasi")
    plt.xlabel("Recall (Dropout)")
    plt.ylabel("Precision (Dropout)")
    plt.legend(loc="lower left")
    plt.tight_layout()
    out = config.FIGURES_DIR / "pr_curves_comparison.png"
    plt.savefig(out, dpi=config.FIGURE_DPI)
    plt.close()
    print(f"  [Gorsel] Kaydedildi: {out.name}")


# ===========================================================================
# CROSS-VALIDATION (5-katli, tum veride)
# ===========================================================================
def run_cross_validation(models_dict, X, y):
    """
    Her model icin 5-katli STRATIFIED cross-validation calistirir ve ortalama
    +/- standart sapma F1 skorunu dondurur.

    Neden CV? Tek bir train/test bolunmesi 'sansli' veya 'sanssiz' olabilir.
    CV, veriyi 5 farkli parcaya bolup her birini sirayla test seti yaparak
    modeli 5 kez degerlendirir. Ortalama daha guvenilir bir performans tahmini,
    standart sapma ise modelin KARARLILIGINI (tutarliligini) gosterir.

    Onemli: CV, ENCODE+SCALE edilmis TUM veri (X, y) uzerinde calistirilir.
    Skorlama 'f1' cunku dengesiz veride azinlik sinifini yansitir. Modeller
    zaten class_weight/sample_weight ile dengelidir; ancak GradientBoosting
    sample_weight'i cross_val_score icinde otomatik almaz, bu yuzden onun icin
    CV'de dengeleme model parametresi uzerinden degil, F1 skorunun kendisi
    uzerinden yorumlanir (yine de adil, cunku tum modeller ayni veriyle test edilir).
    """
    print("\n" + "=" * 70)
    print("5-KATLI CROSS-VALIDATION (F1 skoru, tum veri)")
    print("=" * 70)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    cv_results = []
    for name, model in models_dict.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=-1)
        cv_results.append({
            "Model": name,
            "CV F1 (mean)": round(scores.mean(), 4),
            "CV F1 (std)": round(scores.std(), 4),
        })
        print(f"  {name:20s} -> F1 = {scores.mean():.4f} (+/- {scores.std():.4f})")

    cv_df = pd.DataFrame(cv_results).sort_values("CV F1 (mean)", ascending=False)
    out = config.TABLES_DIR / "cross_validation.csv"
    cv_df.to_csv(out, index=False)
    print(f"\n  [Tablo] Kaydedildi: {out.name}")
    return cv_df


# ===========================================================================
# FEATURE IMPORTANCE (Random Forest ve boosting modeli)
# ===========================================================================
def plot_feature_importance(name, model, feature_names, top_n=15):
    """
    Agac tabanli modellerin ozellik onem skorlarini cizer (ilk top_n).

    Neden yalnizca RF ve boosting? Bu modeller .feature_importances_ saglar
    (her ozelligin bolmelerdeki toplam katkisi). Decision Tree de saglar ama
    yonerge ozellikle RF ve XGBoost/boosting icin ister; en anlamli ve kararli
    onem skorlari bu ensemble modellerden gelir.

    Cikti hem gorsel (.png) hem tablo (.csv) olarak kaydedilir; boylece
    Discussion'da hangi faktorlerin terk riskini surukledigi yorumlanabilir.
    """
    importances = model.feature_importances_
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    # Tabloyu kaydet (tum ozellikler).
    fname_csv = f"feature_importance_{name.lower().replace(' ', '_')}.csv"
    imp_df.to_csv(config.TABLES_DIR / fname_csv, index=False)

    # Ilk top_n ozelligi cizdir.
    top = imp_df.head(top_n).iloc[::-1]  # ters: en onemli en ustte gorunsun
    plt.figure(figsize=(9, 6))
    sns.barplot(x="importance", y="feature", data=top,
                hue="feature", palette="viridis", legend=False)
    plt.title(f"Ozellik Onemi (ilk {top_n}) — {name}")
    plt.xlabel("Onem Skoru")
    plt.ylabel("Ozellik")
    plt.tight_layout()
    fname_png = f"feature_importance_{name.lower().replace(' ', '_')}.png"
    out = config.FIGURES_DIR / fname_png
    plt.savefig(out, dpi=config.FIGURE_DPI)
    plt.close()
    print(f"  [Gorsel] Kaydedildi: {out.name}")
    print(f"  [Tablo]  Kaydedildi: {fname_csv}")
    return imp_df


# ===========================================================================
# KARSILASTIRMA TABLOSU (uc model tek tabloda)
# ===========================================================================
def build_comparison_table(all_metrics):
    """
    Tum modellerin metriklerini tek bir DataFrame'de birlestirir, ROC-AUC'a
    gore siralar, konsola yazar ve CSV olarak kaydeder.

    Bu tablo raporun Results bolumunun merkezidir: uc modeli ayni metriklerle
    yan yana koyarak adil karsilastirma saglar.
    """
    df = pd.DataFrame(all_metrics)
    # Metrikleri okunur sekilde yuvarla.
    metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC",
                   "PR-AUC", "Train-Acc", "Overfit-Gap"]
    df[metric_cols] = df[metric_cols].round(4)
    # En iyi genel ayirt etme gucune gore sirala (ROC-AUC).
    df = df.sort_values("ROC-AUC", ascending=False).reset_index(drop=True)

    out = config.TABLES_DIR / "model_comparison.csv"
    df.to_csv(out, index=False)

    print(f"\n{'='*70}\nMODEL KARSILASTIRMA TABLOSU (test seti)\n{'='*70}")
    print(df.to_string(index=False))
    print(f"\n  [Tablo] Kaydedildi: {out.name}")
    return df


def plot_comparison_bar(comparison_df):
    """
    Karsilastirma tablosunu gruplu cubuk grafige cevirir; tum metrikleri
    tum modeller icin gorsel olarak yan yana gosterir. Rapordaki Results
    bolumunu destekler.
    """
    metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    plot_df = comparison_df.set_index("Model")[metric_cols]

    ax = plot_df.plot(kind="bar", figsize=(11, 6), width=0.8)
    ax.set_title("Model Performans Karsilastirmasi (tum metrikler)")
    ax.set_ylabel("Skor")
    ax.set_xlabel("Model")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right", ncol=5)
    plt.xticks(rotation=0)
    plt.tight_layout()
    out = config.FIGURES_DIR / "model_comparison_bar.png"
    plt.savefig(out, dpi=config.FIGURE_DPI)
    plt.close()
    print(f"  [Gorsel] Kaydedildi: {out.name}")
