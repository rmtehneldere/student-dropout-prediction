"""
01_run_preprocessing.py  —  ASAMA 1 CALISTIRICI
================================================
Bu script Asama 1'in tum adimlarini SIRAYLA calistirir ve ciktilari uretir:
temiz veri seti, EDA gorselleri ve betimsel istatistik tablosu.

Nasil calistirilir?
    python 01_run_preprocessing.py

Ciktilari:
    data/processed_data.csv           -> Enrolled cikarilmis, temiz ikili veri
    outputs/figures/01..04_*.png      -> EDA gorselleri
    outputs/tables/descriptive_statistics.csv

Tasarim: Bu dosya yalnizca 'orkestrasyon' yapar (adimlari sirayla cagirir).
Asil is mantigi src/preprocessing.py icindeki fonksiyonlardadir. Boylece
kod modulerdir ve her fonksiyon bagimsizca test edilebilir.
"""

import sys
from pathlib import Path

# src/ klasorunu iceri aktarma yoluna ekle (modulleri bulabilmek icin).
sys.path.append(str(Path(__file__).resolve().parent / "src"))

import config
import preprocessing as pp


def main():
    print("#" * 70)
    print("# ASAMA 1 — VERI ON ISLEME & EDA")
    print("#" * 70)

    # 0) Cikti klasorlerinin varligini garanti et.
    config.ensure_directories()

    # 1) Ham veriyi yukle.
    df_raw = pp.load_raw_data()
    pp.inspect_structure(df_raw, "HAM VERI (3 sinif dahil)")

    # 2) Enrolled sinifini cikar -> ikili siniflandirma.
    df = pp.make_binary(df_raw)
    pp.inspect_structure(df, "IKILI VERI (Enrolled cikarildi: 0=Graduate, 1=Dropout)")

    # 3) Eksik deger analizi (kaynak 'yok' dese de kodla dogrula).
    pp.analyze_missing_values(df)

    # 4) Veri tiplerini belirle (sayisal / kategorik ayrimi).
    numeric_cols, categorical_cols = pp.identify_feature_types(df)

    # 5) Aykiri deger analizi (raporla, koru).
    pp.analyze_outliers(df, numeric_cols)

    # 6) EDA gorselleri ve betimsel istatistikler.
    print("\n[EDA] Gorseller ve istatistik tablosu uretiliyor...")
    pp.plot_class_distribution(df)
    pp.plot_correlation_heatmap(df, numeric_cols)
    pp.plot_top_correlations_with_target(df, numeric_cols)
    pp.plot_key_numeric_distributions(df, numeric_cols)
    pp.generate_descriptive_statistics(df, numeric_cols)

    # 7) Class imbalance analizi + strateji karari.
    ratio = pp.analyze_class_balance(df[config.TARGET_COLUMN])
    pp.decide_imbalance_strategy(ratio)

    # 8) Temiz ikili veriyi kaydet (Asama 2/3 bunu kullanacak).
    #    Not: Encoding ve scaling'i BURADA yapmayiz; onlar train/test
    #    bolunmesinden SONRA (leakage onlemek icin) modelleme scriptinde
    #    uygulanir. Burada yalnizca temiz, birlesik veriyi saklariz.
    df.to_csv(config.PROCESSED_DATA_PATH, index=False)
    print(f"\n[Kaydet] Temiz veri kaydedildi: {config.PROCESSED_DATA_PATH.name}")

    print("\n" + "#" * 70)
    print("# ASAMA 1 TAMAMLANDI — Ciktilari 'outputs/' klasorunde dogrulayin.")
    print("#" * 70)


if __name__ == "__main__":
    main()
