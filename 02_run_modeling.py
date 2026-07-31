"""
02_run_modeling.py  —  ASAMA 2 & 3 CALISTIRICI
==============================================
Bu script sirasiyla sunlari yapar:
  ASAMA 2 — Model Egitimi:
    * Temiz veriyi yukle, train/test bol (%80/%20, stratify)
    * One-Hot Encoding + StandardScaler (leakage-onlemeli)
    * Logistic Regression, Decision Tree, Random Forest,
      XGBoost/Gradient Boosting egit (4 model, 2 farkli aile)
    * Random Forest icin GridSearchCV ile hiperparametre optimizasyonu
  ASAMA 3 — Degerlendirme & Karsilastirma:
    * Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC hesapla
    * Overfitting kontrolu (egitim vs test dogrulugu)
    * 5-katli Cross-Validation (F1, tum veri)
    * Confusion Matrix (her model), ROC Curve + PR Curve (tek grafik)
    * Feature Importance (Random Forest ve boosting modeli)
    * Dort modeli tek karsilastirma tablosunda birlestir

Nasil calistirilir?
    python 02_run_modeling.py
    (Once 01_run_preprocessing.py calistirilmis olmali; degilse ham veriden
     de calisir cunku ayni on isleme adimlari burada tekrar uygulanir.)

Neden Asama 2 ve 3 ayni scriptte?
- Kullanici bunlarin 'ayni anda barinmasini' istedi. Egitim ve degerlendirme
  ayni calistirmada, ayni veri bolunmesi ve ayni seed ile yapilirsa sonuclar
  tam tutarli ve yeniden uretilebilir olur.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "src"))

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

import config
import preprocessing as pp
import models as md
import evaluation as ev


def load_processed_or_build():
    """
    Temiz veriyi (processed_data.csv) yukler. Yoksa ham veriden ureterek
    devam eder. Bu, Asama 2'nin Asama 1'den BAGIMSIZ da calisabilmesini
    saglar (her asama bagimsiz calisabilir gereksinimi).
    """
    if config.PROCESSED_DATA_PATH.exists():
        print(f"[Veri] Temiz veri bulundu: {config.PROCESSED_DATA_PATH.name}")
        df = pd.read_csv(config.PROCESSED_DATA_PATH)
    else:
        print("[Veri] Temiz veri yok; ham veriden uretiliyor...")
        df_raw = pp.load_raw_data()
        df = pp.make_binary(df_raw)
    return df


def optimize_random_forest(X_train, y_train):
    """
    Random Forest icin GridSearchCV ile hiperparametre optimizasyonu.

    Neden yalnizca Random Forest?
    - Yonerge 'gerekirse GridSearchCV' der; tum modeller icin zorunlu degildir.
    - Random Forest, ince ayardan belirgin fayda goren ve makul surede
      taranabilen dengeli bir modeldir. Boylece optimizasyonun etkisini
      gosteririz ama egitim suresini asiri uzatmayiz (sadelik ilkesi).

    Arama uzayi kucuk ve anlamli tutulmustur (agac sayisi, derinlik, bolme
    icin gereken minimum ornek). 5 katli STRATIFIED cross-validation, dengesiz
    veride her katta sinif oranini korur. Skorlama 'f1' cunku dengesiz veride
    F1, azinlik sinifi performansini accuracy'den daha iyi yansitir.
    """
    print("\n[GridSearchCV] Random Forest hiperparametre optimizasyonu...")
    param_grid = {
        "n_estimators": [200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    }
    base_rf = RandomForestClassifier(
        class_weight="balanced",
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    grid = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    print(f"[GridSearchCV] En iyi parametreler: {grid.best_params_}")
    print(f"[GridSearchCV] En iyi CV F1 skoru : {grid.best_score_:.4f}")
    return grid.best_estimator_


def main():
    print("#" * 70)
    print("# ASAMA 2 & 3 — MODEL EGITIMI, DEGERLENDIRME VE KARSILASTIRMA")
    print("#" * 70)

    config.ensure_directories()

    # -----------------------------------------------------------------
    # ASAMA 2.1 — Veri hazirligi (bol + encode + scale)
    # -----------------------------------------------------------------
    df = load_processed_or_build()

    # Veri tiplerini belirle (encode/scale icin gerekli).
    numeric_cols, categorical_cols = pp.identify_feature_types(df)

    # Train/test bol (%80/%20, stratify). Encoding/scaling BOLMEDEN SONRA.
    X_train, X_test, y_train, y_test = pp.split_data(df, numeric_cols, categorical_cols)

    # Class balance'i egitim setinde raporla.
    pp.analyze_class_balance(y_train)

    # One-Hot Encoding + StandardScaler (leakage-onlemeli: fit yalnizca train).
    X_train_enc, X_test_enc, _scaler = pp.encode_and_scale(
        X_train, X_test, numeric_cols, categorical_cols
    )
    feature_names = X_train_enc.columns.tolist()

    # -----------------------------------------------------------------
    # ASAMA 2.2 — Modelleri kur ve egit
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MODEL EGITIMI")
    print("=" * 70)
    if md.XGBOOST_AVAILABLE:
        print("[Bilgi] XGBoost bulundu -> boosting modeli olarak XGBoost kullanilacak.")
    else:
        print("[Bilgi] XGBoost bulunamadi -> scikit-learn Gradient Boosting kullanilacak")
        print("        (ayni boosting ailesi, yonergeye uygun). Colab'da XGBoost hazirdir.")

    trained = {}
    model_defs = md.get_models()
    for name, model in model_defs.items():
        print(f"\n[Egitim] {name} egitiliyor...")
        trained[name] = md.fit_model(name, model, X_train_enc, y_train)
        print(f"[Egitim] {name} tamamlandi.")

    # Random Forest'i GridSearchCV ile optimize et ve OPTIMIZE edilmis
    # surumle degistir (rapor edilebilir bir iyilestirme adimi).
    best_rf = optimize_random_forest(X_train_enc, y_train)
    trained["Random Forest"] = best_rf
    print("[Egitim] Random Forest, GridSearchCV en iyi modeliyle guncellendi.")

    # -----------------------------------------------------------------
    # ASAMA 3 — Degerlendirme
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DEGERLENDIRME (test seti)")
    print("=" * 70)

    all_metrics = []       # karsilastirma tablosu icin
    roc_data = {}          # ROC ve PR grafikleri icin {model: y_proba}

    for name, model in trained.items():
        # Not: evaluate_model artik egitim setini de alir (overfitting kontrolu).
        metrics, y_pred, y_proba = ev.evaluate_model(
            name, model, X_train_enc, y_train, X_test_enc, y_test
        )
        all_metrics.append(metrics)
        roc_data[name] = y_proba

        # Confusion matrix (her model).
        ev.plot_confusion_matrix(name, y_test, y_pred)

        # Feature importance yalnizca Random Forest ve boosting modeli icin.
        if name in ("Random Forest", "XGBoost", "Gradient Boosting"):
            ev.plot_feature_importance(name, model, feature_names, top_n=15)

    # ROC egrileri (tum modeller tek grafik).
    print("\n[ROC] Birlesik ROC grafigi uretiliyor...")
    ev.plot_roc_curves(roc_data, y_test)

    # Precision-Recall egrileri (dengesiz veri icin daha bilgilendirici).
    print("[PR] Birlesik Precision-Recall grafigi uretiliyor...")
    ev.plot_pr_curves(roc_data, y_test)

    # 5-katli cross-validation (tum encode+scale edilmis veride).
    # Tum veriyi birlestir: egitim + test (ayni on isleme uygulanmis haliyle).
    import pandas as _pd
    X_all = _pd.concat([X_train_enc, X_test_enc], axis=0)
    y_all = _pd.concat([y_train, y_test], axis=0)
    ev.run_cross_validation(trained, X_all, y_all)

    # Karsilastirma tablosu + gruplu cubuk grafik.
    comparison_df = ev.build_comparison_table(all_metrics)
    ev.plot_comparison_bar(comparison_df)

    print("\n" + "#" * 70)
    print("# ASAMA 2 & 3 TAMAMLANDI — Tum ciktilar 'outputs/' klasorunde.")
    print("#" * 70)


if __name__ == "__main__":
    main()
