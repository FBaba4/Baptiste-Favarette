"""
Migration one-shot : donnees_structurees.csv -> maf.db (SQLite)

À lancer une fois (ou à chaque fois que tu veux resynchroniser la base depuis
le CSV, par exemple après un import massif de nouvelles entreprises).
"""
import sqlite3
import pandas as pd
from pathlib import Path

CSV_PATH = "donnees_structurees.csv"
DB_PATH = "maf.db"
TABLE_NAME = "entreprises"

if not Path(CSV_PATH).exists():
    raise FileNotFoundError(f"{CSV_PATH} introuvable — lance ce script depuis la racine de MAF.")

df = pd.read_csv(CSV_PATH)

# Normalisation légère des noms de colonnes pour SQL : espaces et
# caractères spéciaux ne posent pas de problème avec SQLite (les guillemets
# doubles suffisent), mais un accès plus simple sans accents/espaces rend
# les requêtes générées par l'agent plus fiables.
def normaliser_colonne(col: str) -> str:
    remplacements = {
        "é": "e", "è": "e", "ê": "e", "à": "a", "î": "i",
        "ô": "o", "ù": "u", "ç": "c", "’": "", "'": "",
        "(": "", ")": "", "%": "pct", "/": "_", "-": "_", " ": "_",
    }
    resultat = col
    for ancien, nouveau in remplacements.items():
        resultat = resultat.replace(ancien, nouveau)
    return resultat.lower().strip("_")

colonnes_normalisees = {col: normaliser_colonne(col) for col in df.columns}
df = df.rename(columns=colonnes_normalisees)

print("Correspondance des colonnes (original -> normalisé) :")
for original, normalise in colonnes_normalisees.items():
    if original != normalise:
        print(f"  {original!r} -> {normalise!r}")

con = sqlite3.connect(DB_PATH)
df.to_sql(TABLE_NAME, con, if_exists="replace", index=False)
con.close()

print(f"\n✅ {len(df)} lignes migrées dans {DB_PATH}, table '{TABLE_NAME}'.")
print("Tu peux vérifier avec : sqlite3 maf.db 'SELECT * FROM entreprises LIMIT 3;'")