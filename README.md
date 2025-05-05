# 🎬 Movie Recommendation System (Content-Based)

This project implements a content-based movie recommendation system using Python, Pandas, Scikit-learn, and NumPy. It suggests movies similar to a given movie by analyzing textual and numeric features of the dataset.

---

## 🚀 Features

- Combines textual features: `Genre`, `Lead Studio`, and `Year`
- Normalizes numeric features: `Audience Score %`, `Profitability`, `Rotten Tomatoes %`, and `Worldwide Gross`
- Uses `TF-IDF Vectorization` for text-based features
- Computes `Cosine Similarity` between movie vectors
- Recommends top 5 similar movies based on a selected title

---

## 🧾 Dataset

The system uses a CSV file (`movies.csv`) which must contain the following columns:

- `Film`
- `Genre`
- `Lead Studio`
- `Year`
- `Audience score %`
- `Profitability`
- `Rotten Tomatoes %`
- `Worldwide Gross`

> **Note:** Clean and format the columns to remove dollar signs, commas, and missing values.

---

## 📦 Requirements

Install required Python libraries using:

```bash
pip install pandas numpy scikit-learn
