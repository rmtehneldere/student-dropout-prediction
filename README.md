# Student Dropout Prediction using Data Mining Techniques

**BIL 476 — Data Mining | Summer 2026 | Term Project (Undergraduate)**

Bu proje, üniversite öğrencilerinin kayıt sırasında toplanan demografik, sosyoekonomik ve akademik bilgilerini kullanarak **okulu bırakma (Dropout)** riskini erken aşamada tahmin eden bir sınıflandırma çalışmasıdır. Problem, ikili sınıflandırma (Dropout / Graduate) olarak ele alınmış ve **iki farklı model ailesinden (doğrusal + ağaç tabanlı) dört algoritma** karşılaştırılmıştır.

---

## Araştırma Sorusu

> Üniversite öğrencilerinin demografik, sosyoekonomik ve akademik bilgileri kullanılarak okulu bırakma riski ne kadar doğru tahmin edilebilir ve farklı sınıflandırma algoritmaları arasında en başarılı yöntem hangisidir?

---

## Veri Seti

- **Ad:** Predict Students' Dropout and Academic Success
- **Kaynak:** UCI Machine Learning Repository — <https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success>
- **Boyut:** 4.424 kayıt, 36 özellik, 1 hedef değişken
- **Hedef:** Üç sınıf (Dropout / Enrolled / Graduate). Bu projede `Enrolled` çıkarılarak ikili probleme dönüştürülmüştür → **3.630 kayıt** (2.209 Graduate, 1.421 Dropout).

---

## Proje Yapısı

```
student-dropout-prediction/
├── data/
│   ├── data.csv                    # Ham UCI veri seti (girdi)
│   └── processed_data.csv          # Asama 1 ciktisi (Enrolled cikarilmis temiz veri)
├── src/                            # Moduler kaynak kod (fonksiyon kutuphaneleri)
│   ├── config.py                   # Tum ayarlar, yollar, random seed
│   ├── preprocessing.py            # Asama 1: on isleme + EDA fonksiyonlari
│   ├── models.py                   # Asama 2: 4 model tanimi + egitim
│   └── evaluation.py               # Asama 3: metrikler + grafikler + CV + karsilastirma
├── outputs/
│   ├── figures/                    # Tum grafikler (.png)
│   └── tables/                     # Tum tablolar (.csv)
├── report/                         # IEEE formatinda akademik rapor (Asama 4)
├── 01_run_preprocessing.py         # Asama 1 calistirici
├── 02_run_modeling.py              # Asama 2 + 3 calistirici
├── requirements.txt                # Python bagimliliklari
├── .gitignore
└── README.md                       # Bu dosya
```

**Tasarım felsefesi:** İş mantığı `src/` içindeki modüllerdedir; kök dizindeki `01_`/`02_` scriptleri yalnızca bu fonksiyonları sırayla çağıran orkestratörlerdir. Böylece her aşama bağımsız test edilebilir ama birlikte de çalışır.

---

## Kurulum ve Çalıştırma (VS Code / Terminal)

### 1. Projeyi açın ve klasöre girin

```bash
cd student-dropout-prediction
```

### 2. (Önerilen) Sanal ortam oluşturun ve etkinleştirin

```bash
# Windows (PowerShell):
python -m venv venv
venv\Scripts\activate

# macOS / Linux:
python3 -m venv venv
source venv/bin/activate
```

### 3. Bağımlılıkları kurun

```bash
pip install -r requirements.txt
```

### 4. Aşamaları sırayla çalıştırın

```bash
# Asama 1 — Veri on isleme ve EDA
python 01_run_preprocessing.py

# Asama 2 + 3 — Model egitimi, degerlendirme ve karsilastirma
python 02_run_modeling.py
```

Tüm grafikler `outputs/figures/`, tüm tablolar `outputs/tables/` altına kaydedilir. Konsol çıktısı tüm metrikleri ve adımları ekrana yazar.

> **Not (XGBoost):** `requirements.txt` XGBoost'u kurar. Kod, XGBoost kuruluysa boosting modeli olarak **XGBoost**'u, kurulu değilse otomatik olarak scikit-learn **Gradient Boosting**'i kullanır (aynı boosting ailesi, yönergeye uygun). Bu esneklik, kodun her ortamda çalışmasını garanti eder.

---

## Uygulanan Veri Madenciliği Süreci

### Aşama 1 — Veri Ön İşleme ve EDA
- Veri yükleme ve sütun adı temizliği
- `Enrolled` sınıfının çıkarılması (ikili sınıflandırmaya dönüşüm)
- Eksik değer analizi (kodla doğrulandı: eksik yok)
- Aykırı değer analizi (IQR; ağaç tabanlı modeller dayanıklı olduğu için korundu)
- Veri tiplerinin belirlenmesi (18 sayısal + 18 kategorik)
- EDA görselleri (sınıf dağılımı, korelasyon ısı haritası, hedefle korelasyon, dağılımlar)
- One-Hot Encoding + StandardScaler (veri sızıntısını önleyecek şekilde: fit yalnızca eğitim setinde)
- Train/Test Split (%80/%20, `stratify` ile sınıf oranı korunarak)
- Class imbalance analizi (oran ≈ 1.55) ve strateji kararı: **`class_weight='balanced'`**

### Aşama 2 — Model Eğitimi (4 model, 2 farklı aile)
- **Logistic Regression** (doğrusal model / baseline)
- **Decision Tree** (tek ağaç / yorumlanabilir baseline)
- **Random Forest** (bagging ensemble)
- **XGBoost / Gradient Boosting** (boosting ensemble)
- Random Forest için **GridSearchCV** ile hiperparametre optimizasyonu (5 katlı stratified CV, F1 skoru)

### Aşama 3 — Değerlendirme ve Karşılaştırma
- Temel metrikler: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **PR-AUC** (Precision-Recall eğrisi alanı — dengesiz veri için daha bilgilendirici)
- **Overfitting kontrolü** (eğitim vs test doğruluğu farkı)
- **5-katlı Cross-Validation** (F1, tüm veri — kararlılık analizi)
- Confusion Matrix (her model)
- ROC Curve + Precision-Recall Curve (tüm modeller tek grafikte)
- Feature Importance (Random Forest ve boosting modeli)
- Dört modelin tek tabloda karşılaştırılması

---

## Sonuçlar (Test Seti)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Overfit-Gap |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.920 | 0.869 | **0.937** | 0.902 | **0.976** | **0.974** | **-0.001** |
| **Random Forest** | **0.935** | **0.931** | 0.901 | **0.916** | 0.972 | 0.969 | 0.050 |
| XGBoost | 0.913 | 0.875 | 0.909 | 0.891 | 0.972 | 0.970 | 0.077 |
| Decision Tree (baseline) | 0.897 | 0.869 | 0.866 | 0.868 | 0.937 | 0.920 | 0.032 |

**5-Katlı Cross-Validation (F1):** modellerin kararlılığı `cross_validation.csv` içinde raporlanmıştır (her model için ortalama ± standart sapma).

> Değerler `RANDOM_STATE = 42` ile tam olarak yeniden üretilebilir. Boosting modeli olarak XGBoost kullanılmıştır; XGBoost'un kurulu olmadığı bir ortamda kod otomatik olarak scikit-learn Gradient Boosting'e düşer (aynı boosting ailesi, yönergeye uygun).

**Özet bulgular:**
- **En yüksek Accuracy/F1:** Random Forest. **En yüksek ROC-AUC/PR-AUC ve en iyi genelleme (overfit-gap ≈ 0):** Logistic Regression.
- Overfit-gap sütunu gösteriyor ki ensemble ağaç modelleri (Random Forest, XGBoost) eğitim verisine daha çok uyuyor; doğrusal model ise neredeyse hiç ezberlemiyor.
- Özellik önemi analizine göre öğrencinin **2. dönem onaylanan ders sayısı ve notları** terk riskinin en güçlü belirleyicileridir.

---

## Üretilen Çıktılar

**Grafikler (`outputs/figures/`):**
`01_class_distribution.png`, `02_correlation_heatmap.png`, `03_top_correlations_target.png`, `04_numeric_distributions_by_class.png`, `cm_*.png` (4 confusion matrix), `roc_curves_comparison.png`, `pr_curves_comparison.png`, `feature_importance_*.png` (2 adet), `model_comparison_bar.png`

**Tablolar (`outputs/tables/`):**
`descriptive_statistics.csv`, `model_comparison.csv`, `cross_validation.csv`, `feature_importance_*.csv` (2 adet)

---

## Yeniden Üretilebilirlik (Reproducibility)

- Tüm rastgelelik `src/config.py` içindeki tek `RANDOM_STATE` ile kontrol edilir.
- Train/test bölünmesi, model eğitimi, GridSearchCV ve Cross-Validation aynı tohumu kullanır.
- Kod iki kez çalıştırıldığında birebir aynı metrikleri üretir (doğrulanmıştır).

---

## Akademik Dürüstlük ve AI Kullanımı

Bu proje BIL 476 yönergesine uygun olarak hazırlanmıştır. Yapay zeka araçlarının kullanımı, rapor içindeki **AI Assistance Declaration** ve ayrı **Personal Reflection & AI Use Disclosure Form** içinde şeffaf biçimde beyan edilecektir.

---

## Kaynak

M. V. Martins, D. Tolledo, J. Machado, L. M. T. Baptista, and V. Realinho, "Early prediction of student's performance in higher education: A case study," in *Trends and Applications in Information Systems and Technologies*, 2021.
