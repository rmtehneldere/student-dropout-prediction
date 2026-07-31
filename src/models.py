"""
models.py  —  ASAMA 2: Model Egitimi (fonksiyon kutuphanesi)
============================================================
Dort siniflandirma modelini tanimlar ve egitir:
  1. Logistic Regression  (dogrusal model / ikinci baseline)
  2. Decision Tree        (tek agac / yorumlanabilir baseline)
  3. Random Forest        (bagging tabanli ensemble)
  4. XGBoost              (boosting tabanli ensemble)  -- yoksa Gradient Boosting

Neden bu dort model?
- Yonerge (Topic 7) birden fazla yaklasimin karsilastirilmasini ister ve
  ornek olarak Decision Tree ile bir ensemble yontemi (Random Forest / XGBoost)
  onerir. Biz IKI FARKLI MODEL AILESINDEN temsili modeller seceriz; boylece
  karsilastirma yalnizca agac tabanli yontemlerle sinirli kalmaz:
    * Logistic Regression: DOGRUSAL model. Ozelliklerin dogrusal bir
      kombinasyonuyla olasilik uretir. Basit, hizli ve yorumlanabilir; agac
      tabanli modellere karsi guclu bir kiyas noktasi (baseline) olusturur.
    * Decision Tree: tek agac, yorumlanabilir ama overfit egilimli -> baseline.
    * Random Forest: bircok agacin ortalamasi (bagging) -> daha kararli.
    * XGBoost / Gradient Boosting: agaclari ardisik hatalari duzeltecek sekilde
      ekler (boosting) -> tabular veride cogunlukla en yuksek performans.
- Dogrusal (Logistic Regression) ve agac tabanli (digerleri) aileleri birlikte
  sunmak, "birden fazla FARKLI yaklasimi karsilastir" beklentisini fazlasiyla
  karsilar ve Discussion'a zengin bir zemin hazirlar.

XGBoost opsiyonel neden?
- XGBoost harici bir kutuphanedir ve her ortamda kurulu olmayabilir.
  Kuruluysa XGBoost kullanilir; degilse scikit-learn'un GradientBoosting'i
  (ayni boosting ailesinden, yonergeye uygun) otomatik devreye girer.
  Boylece kod HER ortamda calisir (teslim edilebilirlik gereksinimi).
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

import config

# XGBoost'u guvenli sekilde ice aktarmayi dene. Yoksa bayrak False olur ve
# GradientBoosting'e geri duseriz. Kullaniciya durum bildirilir.
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def get_models():
    """
    Uc modeli bir sozluk halinde dondurur: {model_adi: model_nesnesi}.

    Ortak tasarim kararlari:
    - random_state=config.RANDOM_STATE: yeniden uretilebilirlik (ayni sonuc).
    - class_weight='balanced' (Decision Tree ve Random Forest icin): Asama 1'de
      verilen dengesizlik kararinin uygulanmasi. Azinlik sinifi (Dropout)
      hatalarini agirliklandirir; sentetik veri uretmeden dengesizligi ele alir.
    - Hiperparametreler makul ve sade tutulmustur (asiri karmasiklik yok).
      Ornegin agac derinlikleri ve agac sayilari, overfit ile performans
      arasinda dengeli secilmistir. Ince ayar (GridSearchCV) Asama 2'de
      ayrica Random Forest icin uygulanir.
    """
    models = {}

    # --- 1) Logistic Regression (DOGRUSAL BASELINE) ---
    # Dogrusal siniflandirici: ozelliklerin agirlikli toplamini bir sigmoid
    # fonksiyonundan gecirerek Dropout olasiligini tahmin eder. Agac tabanli
    # modellerden farkli bir aileden geldigi icin karsilastirmaya cesitlilik
    # katar. StandardScaler uygulanan veride iyi calisir (olceklenmis girdiler
    # optimizasyonu kolaylastirir - bu yuzden Asama 1'de scaling yaptik).
    # max_iter yuksek tutulur ki cozum kesin yakinssin (yakinsama uyarisi olmasin).
    models["Logistic Regression"] = LogisticRegression(
        max_iter=2000,               # yakinsama icin yeterli iterasyon
        class_weight="balanced",     # dengesizlik karari (Asama 1 ile tutarli)
        random_state=config.RANDOM_STATE,
    )

    # --- 2) Decision Tree (AGAC BASELINE) ---
    # Tek karar agaci: kolay yorumlanir, kurallari gorseldir. Ancak tek basina
    # egitim verisine asiri uyum (overfitting) egilimindedir; bu yuzden
    # 'baseline' (kiyas noktasi) olarak kullanilir. max_depth ile asiri
    # buyumeyi hafifce sinirlariz.
    models["Decision Tree"] = DecisionTreeClassifier(
        max_depth=6,                 # asiri derin/overfit agaci onlemek icin
        class_weight="balanced",     # dengesizlik karari
        random_state=config.RANDOM_STATE,
    )

    # --- 3) Random Forest (BAGGING ENSEMBLE) ---
    # Cok sayida karar agacini farkli veri/ozellik alt kumeleriyle egitip
    # oylarini birlestirir. Tek agaca gore varyansi dusurur, overfit'i azaltir,
    # genellikle daha yuksek dogruluk verir. Ayrica feature importance saglar.
    models["Random Forest"] = RandomForestClassifier(
        n_estimators=300,            # agac sayisi (daha kararli tahmin)
        max_depth=None,              # agaclar tam buyusun; orman ortalama alir
        class_weight="balanced",     # dengesizlik karari
        n_jobs=-1,                   # tum cekirdekleri kullan (hiz)
        random_state=config.RANDOM_STATE,
    )

    # --- 4) XGBoost VEYA Gradient Boosting (BOOSTING ENSEMBLE) ---
    if XGBOOST_AVAILABLE:
        # XGBoost: gradyan artirmali agaclar. Tabular veride cogu zaman en iyi
        # performansi verir. Dengesizlik icin scale_pos_weight kullanilir
        # (pozitif/azinlik sinifin agirligi). Bu deger, egitim setindeki
        # sinif oranindan modelleme scriptinde hesaplanip atanacaktir; burada
        # varsayilan olarak nesne olusturulur.
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,             # boosting'de sig agaclar tercih edilir
            learning_rate=0.1,       # ogrenme adimi (kucuk = daha kararli)
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        # XGBoost yoksa: scikit-learn Gradient Boosting. Ayni boosting
        # ailesinden olup yonergeye tam uygundur. GradientBoosting'in dogrudan
        # class_weight parametresi yoktur; bunun yerine egitim sirasinda
        # sample_weight verecegiz (modelleme scriptinde). Model adini yine de
        # "XGBoost" DEGIL, gercekte ne kullanildigini yansitacak sekilde
        # etiketleriz ki rapor dogru olsun.
        models["Gradient Boosting"] = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.9,
            random_state=config.RANDOM_STATE,
        )

    return models


def fit_model(name, model, X_train, y_train):
    """
    Tek bir modeli egitir. Boosting modeli class_weight parametresini
    dogrudan desteklemiyorsa (GradientBoosting) dengesizligi sample_weight
    ile uygularar.

    Neden ozel muamele?
    - Decision Tree ve Random Forest zaten class_weight='balanced' aldi.
    - XGBoost, scale_pos_weight ile dengelenir (asagida hesaplanir).
    - GradientBoosting ne class_weight ne scale_pos_weight destekler; tek yol
      fit sirasinda sample_weight vermektir. compute_sample_weight('balanced')
      her ornege sinif frekansiyla ters orantili agirlik atar -> ayni
      dengeleme etkisi.
    Boylece HANGI boosting modeli kullanilirsa kullanilsin, dengesizlik
    stratejisi TUTARLI kalir.
    """
    if name == "XGBoost":
        # Pozitif sinif agirligi = negatif ornek sayisi / pozitif ornek sayisi.
        n_pos = (y_train == 1).sum()
        n_neg = (y_train == 0).sum()
        scale = n_neg / n_pos if n_pos > 0 else 1.0
        model.set_params(scale_pos_weight=scale)
        model.fit(X_train, y_train)

    elif name == "Gradient Boosting":
        # Dengeli ornek agirliklari ile egit.
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        model.fit(X_train, y_train, sample_weight=sample_weight)

    else:
        # Decision Tree ve Random Forest: class_weight zaten ayarlandi.
        model.fit(X_train, y_train)

    return model
