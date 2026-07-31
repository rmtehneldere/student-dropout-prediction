"""
preprocessing.py  —  ASAMA 1: Veri On Isleme & EDA (fonksiyon kutuphanesi)
==========================================================================
Bu dosya Asama 1'in TUM adimlarini birbirinden bagimsiz, test edilebilir
fonksiyonlar halinde barindirir. Hicbir fonksiyon calisir calismaz otomatik
tetiklenmez; hepsi "01_run_preprocessing.py" tarafindan sirayla cagrilir.
Bu tasarim "her asama bagimsiz ama birlikte calisabilir" gereksinimini karsilar.

Asama 1 adimlari (yonerge ile birebir):
  1. Veri setini yukle
  2. Enrolled sinifini cikar -> ikili siniflandirma (Dropout / Graduate)
  3. Veri yapisini incele
  4. Eksik deger analizi
  5. Aykiri deger analizi
  6. Veri tiplerini belirle (sayisal / kategorik ayrimi)
  7. EDA gorselleri
  8. One-Hot Encoding
  9. StandardScaler
 10. Train/Test Split (%80/%20)
 11. Class imbalance analizi
 12. Gerekirse SMOTE veya class_weight (bu projede karar: class_weight)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Ekran (GUI) olmadan grafik uretmek icin - sunucu/Colab uyumlu
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config


# ===========================================================================
# ADIM 1 — VERI SETINI YUKLE
# ===========================================================================
def load_raw_data():
    """
    Ham UCI veri setini okur.

    Neden sep=';'? UCI 'Predict Students' Dropout' veri seti noktali virgul
    ile ayrilmistir; varsayilan virgul kullanilirsa tek sutun olarak okunur.
    Ayirici config.CSV_SEPARATOR'da tanimlidir (sihirli sabit yok).
    """
    df = pd.read_csv(config.RAW_DATA_PATH, sep=config.CSV_SEPARATOR)

    # UCI dosyasinda bazi sutun adlarinda bas/son bosluk ve sekme (\t)
    # karakteri bulunur (orn. "Daytime/evening attendance\t"). Bunlari
    # temizlemek, ilerideki sutun secimlerinde hata olmamasi icin gereklidir.
    df.columns = df.columns.str.strip()
    return df


# ===========================================================================
# ADIM 2 — ENROLLED SINIFINI CIKAR (Binary Classification'a donustur)
# ===========================================================================
def make_binary(df):
    """
    Uc sinifli (Dropout / Graduate / Enrolled) problemi ikili siniflandirmaya
    donusturur.

    Neden 'Enrolled' cikariliyor?
    - 'Enrolled' ogrenciler henuz mezun olmamis VE terk etmemis, yani sonucu
      belirsiz ogrencilerdir. Amac "terk (Dropout) mu, mezun (Graduate) mu"
      ayrimini net yapmak oldugu icin belirsiz sinif analiz disi birakilir.
    - Bu, hem modelin ogrenmesini kolaylastirir hem de sonuclarin yorumunu
      netlestirir (yonergede acikca belirtilen yaklasim).

    Ayrica hedef degisken sayisal koda cevrilir:
      Graduate -> 0 (negatif),  Dropout -> 1 (pozitif).
    Cunku ilgilendigimiz olay 'Dropout' tespitidir.
    """
    # Once Enrolled satirlarini at.
    df_binary = df[df[config.TARGET_COLUMN] != config.CLASS_TO_REMOVE].copy()

    # Metin etiketleri sayisal koda cevir (0/1).
    df_binary[config.TARGET_COLUMN] = df_binary[config.TARGET_COLUMN].map(
        config.CLASS_MAPPING
    )
    return df_binary


# ===========================================================================
# ADIM 3 — VERI YAPISINI INCELE
# ===========================================================================
def inspect_structure(df, title):
    """
    Veri setinin boyutunu, sutunlarini ve hedef dagilimini ekrana yazar.
    Bu bir 'kontrol' adimidir: her donusumden sonra verinin beklenen
    durumda oldugunu dogrulamak icin cagirilir.
    """
    print(f"\n{'='*70}\n{title}\n{'='*70}")
    print(f"Boyut (satir, sutun): {df.shape}")
    if config.TARGET_COLUMN in df.columns:
        counts = df[config.TARGET_COLUMN].value_counts().sort_index()
        print(f"Hedef dagilimi:\n{counts.to_string()}")
    return df.shape


# ===========================================================================
# ADIM 4 — EKSIK DEGER ANALIZI
# ===========================================================================
def analyze_missing_values(df):
    """
    Her sutundaki eksik (NaN) deger sayisini hesaplar.

    Neden bu adim onemli? Eksik veri, model egitimini bozar ve yanlis
    sonuclara yol acar. UCI kaynak sayfasi 'Has Missing Values? No' dese de
    bunu KABUL ETMEK yerine kodla DOGRULARIZ (yonerge: 'varsayim yapma').
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0]  # yalnizca eksigi olanlari goster
    if len(missing) == 0:
        print("\n[Eksik Deger] Hicbir sutunda eksik deger yok. (dogrulandi)")
    else:
        print("\n[Eksik Deger] Eksik iceren sutunlar:")
        print(missing.to_string())
    return missing


# ===========================================================================
# ADIM 5 — AYKIRI DEGER ANALIZI
# ===========================================================================
def analyze_outliers(df, numeric_cols):
    """
    IQR (Interquartile Range) yontemiyle sayisal sutunlardaki aykiri deger
    ORANINI hesaplar. Sadece raporlama amaclidir; deger SILMEZ.

    Neden silmiyoruz?
    - Bu veri setinde 'aykiri' gorunen degerlerin cogu gercek ve anlamli
      bilgidir (orn. yasi buyuk ogrenci, notu 0 olan ogrenci terk sinyali
      olabilir). Silmek bilgi kaybina yol acar.
    - Kullanacagimiz agac tabanli modeller (Decision Tree, Random Forest,
      Gradient Boosting/XGBoost) aykiri degerlere karsi DAYANIKLIDIR; bolme
      esiklerine dayandiklari icin uc degerlerden az etkilenirler.
    Bu yuzden karar: aykiri degerleri raporla, ama koru.

    IQR yontemi: Q1 ve Q3 ceyreklikleri arasindaki mesafenin 1.5 katindan
    uzaktaki degerler aykiri sayilir.
    """
    outlier_summary = {}
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        pct = 100 * n_outliers / len(df)
        outlier_summary[col] = round(pct, 2)

    summary_series = pd.Series(outlier_summary).sort_values(ascending=False)
    print("\n[Aykiri Deger] IQR yontemine gore aykiri oran (%) - ilk 10:")
    print(summary_series.head(10).to_string())
    print("Karar: Aykiri degerler KORUNDU (agac tabanli modeller dayanikli).")
    return summary_series


# ===========================================================================
# ADIM 6 — VERI TIPLERINI BELIRLE (sayisal / kategorik ayrimi)
# ===========================================================================
def identify_feature_types(df):
    """
    Ozellikleri 'kategorik' ve 'sayisal' olarak ayirir.

    Neden bu ayrim gerekli?
    - Kategorik degiskenlere One-Hot Encoding uygulanacak.
    - Sayisal degiskenlere StandardScaler uygulanacak.
    Yanlis ayrim, yanlis on isleme yol acar.

    Bu veri setinde COGU sutun tam sayidir ama aslinda KATEGORIKTIR
    (kodlanmis kategoriler: orn. 'Marital status'=1..6, 'Course'=kod).
    Bunlari saf sayisal saymak yanlis olur. Bu yuzden alan bilgisine dayanarak
    gercekten SURELI/OLCEKLI (continuous) olan sutunlari acikca listeleriz;
    geri kalan tum ozellikler kategorik kabul edilir.

    Bu liste UCI veri seti dokumantasyonundaki degisken tanimlarina dayanir
    (varsayim degil, kaynak temelli secim).
    """
    # Gercekten sayisal/olcekli (continuous veya sayim) olan sutunlar:
    numeric_features = [
        "Previous qualification (grade)",   # 0-200 arasi not
        "Admission grade",                  # 0-200 arasi not
        "Age at enrollment",                # yas
        "Curricular units 1st sem (credited)",
        "Curricular units 1st sem (enrolled)",
        "Curricular units 1st sem (evaluations)",
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 1st sem (without evaluations)",
        "Curricular units 2nd sem (credited)",
        "Curricular units 2nd sem (enrolled)",
        "Curricular units 2nd sem (evaluations)",
        "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem (grade)",
        "Curricular units 2nd sem (without evaluations)",
        "Unemployment rate",                # makroekonomik oran
        "Inflation rate",                   # makroekonomik oran
        "GDP",                              # makroekonomik gosterge
    ]

    # Hedef disindaki tum sutunlar arasindan, sayisal listede OLMAYANLAR
    # kategoriktir. (Once hedefi disla.)
    all_features = [c for c in df.columns if c != config.TARGET_COLUMN]

    # Guvenlik: sayisal listedeki bir sutun veride yoksa sessizce atla.
    numeric_features = [c for c in numeric_features if c in df.columns]
    categorical_features = [c for c in all_features if c not in numeric_features]

    print(f"\n[Veri Tipleri] Sayisal ozellik sayisi : {len(numeric_features)}")
    print(f"[Veri Tipleri] Kategorik ozellik sayisi: {len(categorical_features)}")
    return numeric_features, categorical_features


# ===========================================================================
# ADIM 7 — EDA GORSELLERI
# ===========================================================================
def plot_class_distribution(df):
    """Hedef sinif dagilimini cubuk grafik olarak kaydeder (class imbalance gorseli)."""
    plt.figure(figsize=(6, 4))
    counts = df[config.TARGET_COLUMN].value_counts().sort_index()
    labels = [config.NEGATIVE_CLASS, config.POSITIVE_CLASS]  # 0->Graduate, 1->Dropout
    ax = sns.barplot(x=labels, y=counts.values, hue=labels,
                     palette=["#4C72B0", "#C44E52"], legend=False)
    ax.set_title("Hedef Sinif Dagilimi (Binary)")
    ax.set_ylabel("Ogrenci Sayisi")
    ax.set_xlabel("Sinif")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 20, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    out = config.FIGURES_DIR / "01_class_distribution.png"
    plt.savefig(out, dpi=config.FIGURE_DPI)
    plt.close()
    print(f"  [Gorsel] Kaydedildi: {out.name}")


def plot_correlation_heatmap(df, numeric_cols):
    """
    Sayisal ozellikler + hedef arasindaki Pearson korelasyon isi haritasi.
    Neden? Hangi ozelliklerin Dropout ile iliskili oldugunu gormek ve
    ozellikler arasi yuksek korelasyonu (redundans) tespit etmek icin.
    """
    cols = numeric_cols + [config.TARGET_COLUMN]
    corr = df[cols].corr()
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.7})
    plt.title("Sayisal Ozellikler Arasi Korelasyon (Pearson)")
    plt.tight_layout()
    out = config.FIGURES_DIR / "02_correlation_heatmap.png"
    plt.savefig(out, dpi=config.FIGURE_DPI)
    plt.close()
    print(f"  [Gorsel] Kaydedildi: {out.name}")


def plot_top_correlations_with_target(df, numeric_cols):
    """
    Hedef ile en yuksek (mutlak) korelasyona sahip ilk 10 sayisal ozelligi
    cubuk grafikle gosterir. Hangi degiskenlerin terk ile en cok iliskili
    oldugunu ozetler - Discussion bolumunde yorumlanacaktir.
    """
    cols = numeric_cols + [config.TARGET_COLUMN]
    corr_target = df[cols].corr()[config.TARGET_COLUMN].drop(config.TARGET_COLUMN)
    top = corr_target.reindex(corr_target.abs().sort_values(ascending=False).index).head(10)
    plt.figure(figsize=(8, 5))
    colors = ["#C44E52" if v > 0 else "#4C72B0" for v in top.values]
    ax = sns.barplot(x=top.values, y=top.index, hue=top.index,
                     palette=colors, legend=False)
    ax.set_title("Dropout ile En Yuksek Korelasyonlu 10 Ozellik")
    ax.set_xlabel("Pearson Korelasyon Katsayisi")
    plt.tight_layout()
    out = config.FIGURES_DIR / "03_top_correlations_target.png"
    plt.savefig(out, dpi=config.FIGURE_DPI)
    plt.close()
    print(f"  [Gorsel] Kaydedildi: {out.name}")


def plot_key_numeric_distributions(df, numeric_cols):
    """
    Sinifa gore (Dropout vs Graduate) birkac onemli sayisal ozelligin
    dagilimini karsilastirir. Ornek: onaylanan ders sayisi, notlar, yas.
    Bu, terk eden ve mezun olan ogrenciler arasindaki farklari gorsellestirir.
    """
    # Alan bilgisine gore en anlamli/ayirt edici birkac ozellik secildi.
    key_cols = [
        "Curricular units 2nd sem (approved)",
        "Curricular units 1st sem (approved)",
        "Curricular units 2nd sem (grade)",
        "Age at enrollment",
    ]
    key_cols = [c for c in key_cols if c in numeric_cols]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()
    for i, col in enumerate(key_cols):
        for cls_val, cls_name, color in [(0, config.NEGATIVE_CLASS, "#4C72B0"),
                                         (1, config.POSITIVE_CLASS, "#C44E52")]:
            subset = df[df[config.TARGET_COLUMN] == cls_val][col]
            axes[i].hist(subset, bins=25, alpha=0.6, label=cls_name, color=color)
        axes[i].set_title(col, fontsize=10)
        axes[i].legend()
    fig.suptitle("Sinifa Gore Onemli Sayisal Ozellik Dagilimlari", fontsize=13)
    plt.tight_layout()
    out = config.FIGURES_DIR / "04_numeric_distributions_by_class.png"
    plt.savefig(out, dpi=config.FIGURE_DPI)
    plt.close()
    print(f"  [Gorsel] Kaydedildi: {out.name}")


def generate_descriptive_statistics(df, numeric_cols):
    """
    Sayisal ozelliklerin betimsel istatistiklerini (ortalama, std, min, max,
    ceyreklikler) bir CSV tablosu olarak kaydeder. Rapordaki Dataset
    Description bolumunde kullanilacaktir.
    """
    stats = df[numeric_cols].describe().T
    out = config.TABLES_DIR / "descriptive_statistics.csv"
    stats.to_csv(out)
    print(f"  [Tablo] Kaydedildi: {out.name}")
    return stats


# ===========================================================================
# ADIM 8 & 9 — ONE-HOT ENCODING + STANDARDSCALER
# ===========================================================================
def encode_and_scale(X_train, X_test, numeric_cols, categorical_cols):
    """
    Kategorik degiskenlere One-Hot Encoding, sayisal degiskenlere
    StandardScaler uygular.

    KRITIK NOKTA - Veri sizintisini (data leakage) onleme:
    - Scaler SADECE egitim verisi (X_train) uzerinde fit edilir.
    - Ayni scaler ile hem egitim hem test verisi DONUSTURULUR (transform).
    - Boylece test verisinin istatistigi egitime 'sizmaz'. Bu, dogru ve
      durust bir degerlendirme icin zorunludur.

    One-Hot Encoding neden?
    - Kategorik kodlar (orn. Course=171) aslinda sirasal anlam tasimaz;
      dogrudan sayi olarak verilirse model yanlis bir sira/buyukluk iliskisi
      ogrenir. One-Hot her kategoriyi ayri 0/1 sutununa cevirerek bunu onler.
    - Egitim ve test setleri ayni sutunlara sahip olsun diye once birlestirip
      encode eder, sonra ayni indekslerle tekrar ayiririz (align).

    StandardScaler neden?
    - Sayisal ozellikleri ortalama=0, std=1 olacak sekilde olcekler.
    - Agac modelleri olceklemeye duyarsizdir ama olcekleme zarar da vermez;
      tutarli ve genellenebilir bir pipeline icin tum sayisal sutunlara
      uygulanir (ayni on isleme her modelde kullanilir).
    """
    # --- One-Hot Encoding ---
    # Egitim ve testi gecici olarak isaretleyip birlestirerek ayni kategori
    # sutunlarinin olusmasini garantile.
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train["__split__"] = "train"
    X_test["__split__"] = "test"
    combined = pd.concat([X_train, X_test], axis=0)

    combined_encoded = pd.get_dummies(
        combined, columns=categorical_cols, drop_first=True
    )
    # get_dummies bool sutunlar uretir; modeller icin int'e cevir.
    bool_cols = combined_encoded.select_dtypes(include="bool").columns
    combined_encoded[bool_cols] = combined_encoded[bool_cols].astype(int)

    # Tekrar egitim/test olarak ayir.
    X_train_enc = combined_encoded[combined_encoded["__split__"] == "train"].drop(
        columns="__split__"
    )
    X_test_enc = combined_encoded[combined_encoded["__split__"] == "test"].drop(
        columns="__split__"
    )

    # --- StandardScaler (yalnizca sayisal sutunlara) ---
    scaler = StandardScaler()
    # fit YALNIZCA egitim verisinde:
    X_train_enc[numeric_cols] = scaler.fit_transform(X_train_enc[numeric_cols])
    # transform ayni scaler ile testte:
    X_test_enc[numeric_cols] = scaler.transform(X_test_enc[numeric_cols])

    print(f"\n[Encoding+Scaling] Encoding sonrasi ozellik sayisi: {X_train_enc.shape[1]}")
    print(f"[Encoding+Scaling] Egitim seti: {X_train_enc.shape}, Test seti: {X_test_enc.shape}")
    return X_train_enc, X_test_enc, scaler


# ===========================================================================
# ADIM 10 — TRAIN/TEST SPLIT (%80/%20)
# ===========================================================================
def split_data(df, numeric_cols, categorical_cols):
    """
    Veriyi once ozellik (X) ve hedef (y) olarak ayirir, ardindan %80/%20
    egitim/test olarak boler.

    Neden stratify=y?
    - Sinif dagilimi dengesiz oldugu icin (Dropout < Graduate), bolme
      sirasinda oranlarin korunmasi gerekir. stratify, hem egitim hem test
      setinde ayni Dropout/Graduate oranini saglar. Aksi halde test seti
      temsili olmaz ve metrikler yaniltir.

    Not: Encoding ve scaling'i BOLMEDEN SONRA yapariz (encode_and_scale ile),
    cunku scaler'in test istatistigini gormemesi gerekir (leakage onleme).
    """
    X = df.drop(columns=config.TARGET_COLUMN)
    y = df[config.TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,  # sinif oranlarini koru
    )
    print(f"\n[Split] Egitim: {X_train.shape[0]} ornek | Test: {X_test.shape[0]} ornek")
    print(f"[Split] Egitim Dropout orani: {y_train.mean():.3f} | "
          f"Test Dropout orani: {y_test.mean():.3f}  (stratify ile esitlendi)")
    return X_train, X_test, y_train, y_test


# ===========================================================================
# ADIM 11 — CLASS IMBALANCE ANALIZI
# ===========================================================================
def analyze_class_balance(y):
    """
    Sinif dengesizligini sayisal olarak raporlar ve dengesizlik oranini
    hesaplar. Bu oran, hangi dengesizlik stratejisinin gerektigine karar
    vermek icin kullanilir.
    """
    counts = y.value_counts().sort_index()
    n_neg = counts.get(0, 0)  # Graduate
    n_pos = counts.get(1, 0)  # Dropout
    ratio = n_neg / n_pos if n_pos > 0 else float("inf")
    print(f"\n[Class Balance] Graduate (0): {n_neg} | Dropout (1): {n_pos}")
    print(f"[Class Balance] Dengesizlik orani (Graduate/Dropout): {ratio:.2f}")
    return ratio


# ===========================================================================
# ADIM 12 — DENGESIZLIK STRATEJISI KARARI
# ===========================================================================
def decide_imbalance_strategy(ratio):
    """
    Class imbalance icin strateji KARARINI verir ve gerekcesini yazar.

    KARAR: 'class_weight="balanced"' kullanilacak (SMOTE degil).

    Neden class_weight, SMOTE degil?
    - Dengesizlik cok siddetli degil (oran ~1.5:1 civari). Bu seviyede
      sentetik ornek uretmeye (SMOTE) gerek yoktur.
    - class_weight="balanced" hicbir sentetik veri URETMEDEN, azinlik sinifi
      hatalarini egitimde daha agir cezalandirir. Bu, veriyi bozmadan
      dengesizligi ele almanin en sade ve seffaf yoludur.
    - Ayrica secilen uc modelin (Decision Tree, Random Forest, Gradient
      Boosting) hepsi ornek agirliklandirmayi dogal olarak destekler; boylece
      tek ve tutarli bir strateji tum modellerde uygulanabilir.
    - Yonerge 'gerekirse SMOTE VEYA class_weight' der; ikisinden birini
      secmek yeterlidir. Sadelik ve yeniden uretilebilirlik acisindan
      class_weight tercih edilmistir.

    Bu fonksiyon bir bilgi/karar adimidir; agirliklandirma asil modelleme
    asamasinda (Asama 2) modellere parametre olarak verilir.
    """
    print("\n[Imbalance Karari] Strateji: class_weight='balanced'")
    print("  Gerekce: Dengesizlik orta duzeyde; sentetik veri (SMOTE) yerine")
    print("  azinlik sinifini agirliklandirmak daha sade, seffaf ve")
    print("  yeniden uretilebilirdir. Tum secili modeller bunu destekler.")
    return "class_weight"
