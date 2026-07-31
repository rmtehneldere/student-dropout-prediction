"""
config.py
=========
Projedeki tum ayarlarin, yollarin ve sabitlerin tek merkezden yonetildigi dosya.

Neden bu dosya var?
-------------------
- Yeniden uretilebilirlik (reproducibility): RANDOM_STATE tek bir yerde tanimlanir
  ve butun asamalar ayni tohumu (seed) kullanir. Boylece her calistirmada ayni
  train/test bolunmesi, ayni model sonuclari elde edilir.
- Bakim kolayligi: Bir yol veya parametre degisince tek dosyayi guncellemek yeterlidir.
- Sihirli sabit (magic number) kullanmaktan kacinmak: Kod icine gomulu rakamlar
  yerine anlamli isimli sabitler kullanilir.

BIL476 yonergesi ile iliskisi:
- "Tum deneyler yeniden uretilebilir olmali" gereksinimini karsilar.
- "%80/%20 train/test split" gereksinimini TEST_SIZE ile sabitler.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# 1) YENIDEN URETILEBILIRLIK (REPRODUCIBILITY)
# ---------------------------------------------------------------------------
# Tum rastgelelik iceren islemlerde (train/test split, model egitimi, SMOTE)
# bu tohum kullanilir. Ayni tohum -> ayni sonuc.
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 2) DOSYA YOLLARI
# ---------------------------------------------------------------------------
# __file__ bu config.py dosyasinin konumudur. .parent.parent ile proje kok
# klasorune ulasiriz. Boylece proje baska bir bilgisayara tasinsa bile
# yollar dogru calisir (mutlak yol gomulu degildir).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "data.csv"                  # Ham UCI veri seti
PROCESSED_DATA_PATH = DATA_DIR / "processed_data.csv"  # Asama 1 ciktisi (temiz veri)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"   # Tum grafikler (.png) buraya kaydedilir
TABLES_DIR = OUTPUT_DIR / "tables"     # Tum tablolar (.csv) buraya kaydedilir

# ---------------------------------------------------------------------------
# 3) VERI SETI PARAMETRELERI
# ---------------------------------------------------------------------------
# UCI "Predict Students' Dropout and Academic Success" veri seti noktali
# virgul (;) ile ayrilmistir. Bu yuzden ayirici acikca belirtilir.
CSV_SEPARATOR = ";"

# Hedef degiskenin sutun adi (veri setinde son sutun).
TARGET_COLUMN = "Target"

# Problemi ikili siniflandirmaya (binary classification) donusturmek icin
# cikarilacak sinif. Yonerge geregi "Enrolled" ogrencileri analiz disi birakilir;
# boylece net bir Dropout vs Graduate problemi kalir.
CLASS_TO_REMOVE = "Enrolled"

# Kalan iki sinif ve sayisal kodlari.
# Pozitif sinif (1) = Dropout. Cunku projenin amaci "riskli/terk eden" ogrenciyi
# tespit etmektir; ilgilendigimiz olay Dropout oldugu icin onu pozitif secmek
# Precision/Recall/ROC-AUC yorumunu anlamli kilar.
POSITIVE_CLASS = "Dropout"    # -> 1
NEGATIVE_CLASS = "Graduate"   # -> 0
CLASS_MAPPING = {NEGATIVE_CLASS: 0, POSITIVE_CLASS: 1}

# ---------------------------------------------------------------------------
# 4) EGITIM/TEST BOLUNMESI
# ---------------------------------------------------------------------------
# Yonerge ve orijinal UCI calismasi ile uyumlu olarak %80 egitim, %20 test.
TEST_SIZE = 0.20

# Bolme sirasinda sinif oranlarini korumak icin stratify kullanilacaktir
# (kod tarafinda). Bu, dengesiz sinif dagiliminda test setinin temsili
# olmasini saglar.

# ---------------------------------------------------------------------------
# 5) GORSEL AYARLARI
# ---------------------------------------------------------------------------
# Tum grafikler ayni cozunurlukte kaydedilsin diye tek yerde tanimli.
FIGURE_DPI = 150

# ---------------------------------------------------------------------------
# 6) YARDIMCI FONKSIYON
# ---------------------------------------------------------------------------
def ensure_directories():
    """
    Cikti klasorlerinin var oldugundan emin olur; yoksa olusturur.
    Her asama scripti baslangicta bunu cagirarak "klasor yok" hatasini onler.
    """
    for directory in (DATA_DIR, OUTPUT_DIR, FIGURES_DIR, TABLES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
