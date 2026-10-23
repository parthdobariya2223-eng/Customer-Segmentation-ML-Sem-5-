# ============================================================
# Customer Segmentation Using K-Means Clustering
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
DATA_PATH = DATA_DIR / "online_retail_sample_20000.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 2. CHECK DATASET PATH
# ------------------------------------------------------------
print("=" * 60)
print("CUSTOMER SEGMENTATION USING K-MEANS")
print("=" * 60)
print("\nProject folder:")
print(BASE_DIR)
print("\nDataset path:")
print(DATA_PATH)

if not DATA_PATH.exists():

    print("\nERROR: Dataset file was not found!")

    print("\nExpected folder structure:")
    print(r"""
Customer Segmentation project
│
├── customer_segmentation.py
│
├── dataset
│   └── online_retail_sample_20000.csv
│
└── outputs
""")
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

# ------------------------------------------------------------
# 3. LOAD DATA
# ------------------------------------------------------------
print("\nLoading dataset...")
df = pd.read_csv(DATA_PATH,encoding="latin1")
print("Dataset loaded successfully!")
print("Rows:", len(df))
print("Columns:", len(df.columns))

# ------------------------------------------------------------
# 4. FIX COLUMN NAMES
# ------------------------------------------------------------
print("\nOriginal columns:")
print(df.columns.tolist())

# Remove BOM / unwanted characters
df.columns = (df.columns.str.replace("\ufeff", "", regex=False).str.replace("ï»¿", "", regex=False).str.strip())
print("\nCleaned columns:")
print(df.columns.tolist())

# Make sure InvoiceNo exists
if "InvoiceNo" not in df.columns:
    print("\nERROR: InvoiceNo column was not found.")
    print("Available columns:")
    print(df.columns.tolist())
    raise KeyError("InvoiceNo column is missing from the dataset.")

# ------------------------------------------------------------
# 5. DATA CLEANING
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

# Remove duplicate records
df = df.drop_duplicates()

# Convert InvoiceDate
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"],errors="coerce")

# Convert Quantity
df["Quantity"] = pd.to_numeric(df["Quantity"],errors="coerce")

# Convert UnitPrice
df["UnitPrice"] = pd.to_numeric(df["UnitPrice"],errors="coerce")

# Convert CustomerID
df["CustomerID"] = pd.to_numeric(df["CustomerID"],errors="coerce")

# Remove missing important values
df = df.dropna(
    subset=["InvoiceDate","Quantity","UnitPrice","CustomerID"])

# Remove returns / negative quantities
df = df[df["Quantity"] > 0]

# Remove invalid prices
df = df[df["UnitPrice"] > 0]

# Calculate transaction amount
df["Amount"] = (df["Quantity"] * df["UnitPrice"])
print("Rows after cleaning:", len(df))

# ------------------------------------------------------------
# 6. RFM ANALYSIS
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("CREATING RFM FEATURES")
print("=" * 60)

# Reference date
reference_date = (
    df["InvoiceDate"].max()
    + pd.Timedelta(days=1))

# Create RFM table
rfm = df.groupby("CustomerID").agg(
    Recency=("InvoiceDate",lambda x:(reference_date - x.max()).days),
    Frequency=("InvoiceNo","nunique"),
    Monetary=("Amount","sum")).reset_index()
print("\nRFM dataset:")
print(rfm.head())
print("\nNumber of customers:")
print(len(rfm))

# ------------------------------------------------------------
# 7. PREPARE FEATURES
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("PREPARING FEATURES")
print("=" * 60)
features = ["Recency","Frequency","Monetary"]
X = rfm[features].copy()

# Log transformation
X_log = np.log1p(X)

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)
print("Features prepared successfully.")

# ------------------------------------------------------------
# 8. K-MEANS CLUSTERING
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("APPLYING K-MEANS CLUSTERING")
print("=" * 60)
kmeans = KMeans(n_clusters=2,random_state=42,n_init=10)
rfm["Cluster"] = (kmeans.fit_predict(X_scaled))
print("K-Means clustering completed.")

# ------------------------------------------------------------
# 9. MODEL EVALUATION
# ------------------------------------------------------------
silhouette = silhouette_score( X_scaled, rfm["Cluster"])
print("\n" + "=" * 60)
print("MODEL RESULTS")
print("=" * 60)
print("Number of clusters:",2)
print("Silhouette Score:", round(silhouette, 4))

# ------------------------------------------------------------
# 10. CLUSTER PROFILE
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("CLUSTER PROFILE")
print("=" * 60)

profile = rfm.groupby("Cluster").agg(
    Customers=("CustomerID","count"),
    Avg_Recency=("Recency","mean"),
    Avg_Frequency=("Frequency","mean"),
    Avg_Monetary=("Monetary","mean")).round(2)
print(profile)

# ------------------------------------------------------------
# 11. SAVE RESULTS
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

# Customer segmentation
rfm.to_csv(OUTPUT_DIR /"customer_segments.csv",index=False)

# Cluster profile
profile.to_csv( OUTPUT_DIR / "cluster_profile.csv")
print("Customer segments saved successfully.")
print("Cluster profile saved successfully.")

# ------------------------------------------------------------
# 12. VISUALIZATION
# ------------------------------------------------------------
print("\nCreating visualization...")
plt.figure(figsize=(8, 6))
plt.scatter(rfm["Recency"],rfm["Monetary"],c=rfm["Cluster"],alpha=0.6)
plt.xlabel("Recency")
plt.ylabel("Monetary Value")
plt.title("Customer Segmentation using K-Means")
plt.tight_layout()
# Save graph
plt.savefig(OUTPUT_DIR /"customer_segments.png",dpi=300)
plt.show()

# ------------------------------------------------------------
# 13. FINAL RESULT
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("CUSTOMER SEGMENTATION COMPLETED")
print("=" * 60)
print("Customers:",len(rfm))
print("Clusters:",rfm["Cluster"].nunique())
print("Silhouette Score:",round(silhouette, 4))
print("\nOutput folder:")
print(OUTPUT_DIR)
print("\nGenerated files:")
print("1. customer_segments.csv")
print("2. cluster_profile.csv")
print( "3. customer_segments.png")
print("\n" + "=" * 60)