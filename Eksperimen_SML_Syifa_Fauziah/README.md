# Eksperimen_SML_Syifa_Fauziah

Repository untuk eksperimen Machine Learning pada dataset Wine Quality.

## Author
**Nama:** Syifa Fauziah  
**Course:** Membangun Sistem Machine Learning - Dicoding Indonesia

## Dataset
- **Sumber:** UCI Machine Learning Repository
- **URL:** https://archive.ics.uci.edu/ml/datasets/wine+quality
- **Deskripsi:** Wine Quality Dataset berisi 6,497 sampel wine (red & white) dengan 11 fitur physicochemical

## Struktur Repository

```
Eksperimen_SML_Syifa_Fauziah/
├── .github/
│   └── workflows/
│       └── preprocessing.yml      # GitHub Actions workflow (Advanced)
├── preprocessing/
│   ├── Eksperimen_Syifa_Fauziah.ipynb    # Notebook eksperimen
│   ├── automate_Syifa_Fauziah.py         # Script automation (Skilled)
│   └── winequality_preprocessing/        # Output preprocessed data
├── winequality_raw/               # Raw data storage
└── README.md
```

## GitHub Actions Workflow

Workflow akan **otomatis berjalan** ketika:
- Push ke branch `main` atau `master`
- Pull request ke branch `main` atau `master`
- Manual trigger via "Run workflow"

### Cara Trigger Manual:
1. Buka tab **Actions** di repository
2. Pilih **Data Preprocessing Pipeline**
3. Klik **Run workflow**

## Fitur Notebook
1. **Dataset Introduction** - Penjelasan sumber dan deskripsi dataset
2. **Data Loading** - Fetch data dari UCI Repository
3. **EDA (Exploratory Data Analysis)** - Analisis statistik dan visualisasi
4. **Preprocessing** - Handling duplicates, outliers, dan missing values
5. **Feature Engineering** - Pembuatan fitur baru
6. **Data Splitting** - Train/test split dengan stratified sampling
7. **Feature Scaling** - StandardScaler normalization
8. **Export Artifacts** - Simpan scaler, encoder, dan preprocessed data

## Links
- **GitHub:** https://github.com/syfauziah/Eksperimen_SML_Syifa_Fauziah
- **Workflow-CI:** https://github.com/syfauziah/Workflow-CI

## License
Educational use - Dicoding Indonesia
