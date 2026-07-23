import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

DOCS_DIR = Path("documents")
INDEXES_DIR = Path("faiss_indexes")

DOCS_DIR.mkdir(exist_ok=True)
INDEXES_DIR.mkdir(exist_ok=True)

# --- Paramètres de batching ---
# On regroupe plusieurs chunks par appel API au lieu d'un appel par chunk.
# Gain : au lieu de N appels espacés de 1 seconde (N secondes de latence pure),
# on fait N / BATCH_SIZE appels, chacun traitant BATCH_SIZE chunks en une seule
# requête HTTP. La pause ne s'applique plus qu'entre les batchs, pas entre
# chaque chunk : le temps d'ingestion devient quasi proportionnel au nombre
# de batchs, pas au nombre de chunks.

BATCH_SIZE = 40         # nombre de chunks envoyés par appel API
PAUSE_ENTRE_BATCHS = 30 # secondes, marge de sécurité anti-quota entre batchs
_embeddings = None  # variable globale pour stocker l'instance d'embeddings


def get_embeddings():
    """Instancie le modèle d'embeddings une seule fois (réutilisé par le CLI
    et par process_and_index_pdf appelé depuis Streamlit)."""
    global _embeddings
    if _embeddings is None:
        print("🔍 Initialisation du modèle d'embeddings Google Gemini...", flush=True)
        _embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    return _embeddings


def lister_documents():
    """Liste les fichiers PDF dans le dossier documents/."""
    return sorted([f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")])


def menu_ingestion():
    """Affiche le menu et retourne le fichier sélectionné."""
    while True:
        fichiers = lister_documents()
        print("\n" + "=" * 40)
        print("📥 MENU D'INGESTION DE DOCUMENTS (Anti-Quota)")
        print("=" * 40)

        if not fichiers:
            print("⚠️ Aucun PDF trouvé dans 'documents/'.")
            print("👉 Déposez un fichier, puis appuyez sur 'r' pour actualiser.")
        else:
            for i, f in enumerate(fichiers, 1):
                dossier_index = INDEXES_DIR / f"{f}_index"
                statut = "[DÉJÀ INDEXÉ]" if dossier_index.exists() else "[À INDEXER]"
                print(f" [{i}] - {f} {statut}")

        print("\nOptions :")
        print(" - Entrez le numéro du document à préparer.")
        print(" - [r] pour Rafraîchir la liste.")
        print(" - [q] pour Quitter.")

        choix = input("\n👉 Votre choix : ").strip().lower()

        if choix == 'q':
            sys.exit(0)
        elif choix == 'r':
            continue
        elif choix.isdigit():
            index_choix = int(choix) - 1
            if 0 <= index_choix < len(fichiers):
                return fichiers[index_choix]
            else:
                print("❌ Numéro invalide.")
        else:
            print("❌ Commande non reconnue.")


def ingerer_par_batch(documents, embeddings, dossier_index):
    """
    Construit l'index FAISS par batchs de chunks plutôt qu'un chunk à la fois.
    Chaque appel à FAISS.from_documents / add_documents déclenche en interne
    embeddings.embed_documents(textes), qui envoie déjà plusieurs textes en
    une seule requête à l'API Gemini. Grouper les chunks en amont réduit donc
    le nombre total de requêtes HTTP, pas seulement le nombre de time.sleep().
    """
    total = len(documents)
    vector_store = None

    for i in range(0, total, BATCH_SIZE):
        batch = documents[i:i + BATCH_SIZE]

        if vector_store is None:
            vector_store = FAISS.from_documents(batch, embeddings)
        else:
            vector_store.add_documents(batch)

        traite = min(i + BATCH_SIZE, total)
        print(f"   -> Indexation : {traite}/{total} segments traités "
              f"(batch de {len(batch)}).", flush=True)

        if traite < total:
            time.sleep(PAUSE_ENTRE_BATCHS)

    return vector_store


def process_and_index_pdf(uploaded_file) -> str:
    nom_fichier = uploaded_file.name
    chemin_pdf = DOCS_DIR / nom_fichier
    chemin_pdf.write_bytes(uploaded_file.getbuffer())

    dossier_index = INDEXES_DIR / f"{nom_fichier}_index"
    if dossier_index.exists():
        return f"{nom_fichier}_index"   # déjà indexé, aucun appel API

    loader = PyPDFLoader(str(chemin_pdf))
    raw_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    documents = text_splitter.split_documents(raw_documents)

    vector_store = ingerer_par_batch(documents, get_embeddings(), dossier_index)
    vector_store.save_local(str(dossier_index))

    return f"{nom_fichier}_index"


# ─────────────────────────────────────────────────────────────────────────
# Registre entreprise -> documents — permet à une conversation d'être liée
# à une ENTREPRISE plutôt qu'à un seul document : plusieurs rapports
# (annuel 2025, T2 2026...) peuvent être associés à la même entreprise, et
# ajoutés à n'importe quel moment (le registre est relu à chaque appel,
# jamais figé dans une conversation).
# ─────────────────────────────────────────────────────────────────────────
REGISTRE_PATH = Path("documents_entreprises.json")


def _charger_registre() -> dict:
    if REGISTRE_PATH.exists():
        return json.loads(REGISTRE_PATH.read_text(encoding="utf-8"))
    return {}


def _sauver_registre(registre: dict):
    REGISTRE_PATH.write_text(json.dumps(registre, ensure_ascii=False, indent=2), encoding="utf-8")


def lister_entreprises() -> list:
    """Toutes les entreprises ayant au moins un document associé."""
    return sorted(_charger_registre().keys())


def documents_pour_entreprise(entreprise: str) -> list:
    """
    Liste des documents associés à une entreprise, chacun sous la forme
    {"index_name": ..., "titre": ..., "fichier": ..., "date_ajout": ...}.
    Relu depuis le disque à chaque appel — pas de cache — pour qu'un
    document ajouté après coup soit immédiatement visible partout.
    """
    return _charger_registre().get(entreprise, [])


def associer_document_entreprise(entreprise: str, index_name: str, titre_affiche: str, fichier: str):
    """
    Associe un document déjà indexé (index_name) à une entreprise. Si le
    document (même index_name) est déjà associé à cette entreprise, ne
    duplique pas l'entrée.
    """
    registre = _charger_registre()
    documents = registre.setdefault(entreprise, [])

    if any(d["index_name"] == index_name for d in documents):
        return  # déjà associé, rien à faire

    documents.append({
        "index_name": index_name,
        "titre": titre_affiche,
        "fichier": fichier,
        "date_ajout": datetime.now().isoformat(),
    })
    _sauver_registre(registre)


def retirer_document_entreprise(entreprise: str, index_name: str):
    """Détache un document d'une entreprise (ne supprime pas l'index FAISS
    lui-même, juste l'association — le document reste réutilisable ailleurs)."""
    registre = _charger_registre()
    if entreprise not in registre:
        return
    registre[entreprise] = [d for d in registre[entreprise] if d["index_name"] != index_name]
    if not registre[entreprise]:
        del registre[entreprise]
    _sauver_registre(registre)


def _normaliser(texte: str) -> str:
    """Minuscule, accents retirés, tout caractère non-alphanumérique
    supprimé — pour comparer un nom d'entreprise à un nom de fichier
    malgré les tirets/underscores/espaces/majuscules qui diffèrent."""
    remplacements = {"é": "e", "è": "e", "ê": "e", "à": "a", "î": "i", "ô": "o", "ù": "u", "ç": "c"}
    t = texte.lower()
    for accent, lettre in remplacements.items():
        t = t.replace(accent, lettre)
    return "".join(c for c in t if c.isalnum())


def associer_documents_automatiquement(entreprise: str) -> list:
    """
    Recherche, parmi TOUS les documents présents dans documents/ (indexés ou
    non), ceux dont le nom de fichier contient le nom de l'entreprise (une
    fois les deux normalisés) — pas de sélection manuelle nécessaire.
    Indexe au passage les documents trouvés qui ne le sont pas encore.

    Retourne la liste des titres nouvellement associés lors de CET appel
    (liste vide si aucun nouveau document trouvé — soit rien ne correspond,
    soit tout était déjà associé).
    """
    entreprise_norm = _normaliser(entreprise)
    if len(entreprise_norm) < 3:
        return []  # évite les faux positifs sur des noms trop courts (ex. "SA", "AI")

    deja_associes = {d["index_name"] for d in documents_pour_entreprise(entreprise)}
    nouveaux = []

    for fichier in lister_documents():
        fichier_norm = _normaliser(fichier)
        if entreprise_norm not in fichier_norm:
            continue

        index_name = f"{fichier}_index"
        if index_name in deja_associes:
            continue

        dossier_index = INDEXES_DIR / index_name
        if not dossier_index.exists():
            class _FichierLocal:
                def __init__(self, path: Path):
                    self.name = path.name
                    self._data = path.read_bytes()
                def getbuffer(self):
                    return self._data
            process_and_index_pdf(_FichierLocal(DOCS_DIR / fichier))

        titre = fichier.replace(".pdf", "").replace("_", " ").replace("-", " ")
        associer_document_entreprise(entreprise, index_name, titre, fichier)
        nouveaux.append(titre)

    return nouveaux


if __name__ == "__main__":
    nom_fichier = menu_ingestion()
    chemin_pdf = DOCS_DIR / nom_fichier
    dossier_index = INDEXES_DIR / f"{nom_fichier}_index"

    if dossier_index.exists():
        print(f"\n✅ L'index pour '{nom_fichier}' existe déjà dans '{dossier_index}'.")
        print("Aucune requête API n'a été consommée. Vous pouvez passer à la requête !")
        sys.exit(0)

    print(f"\n⚙️ Lecture de '{nom_fichier}'...")
    loader = PyPDFLoader(str(chemin_pdf))
    raw_documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    documents = text_splitter.split_documents(raw_documents)

    nb_batchs = (len(documents) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"✂️ {len(documents)} segments générés. "
          f"Vectorisation par batchs de {BATCH_SIZE} "
          f"({nb_batchs} appels API au lieu de {len(documents)})...", flush=True)

    # Correction : on appelle la FONCTION get_embeddings(), pas la variable
    # _embeddings (qui vaut encore None à ce stade puisque rien ne l'avait
    # initialisée avant cette ligne dans le bloc CLI).
    vector_store = ingerer_par_batch(documents, get_embeddings(), dossier_index)

    vector_store.save_local(str(dossier_index))
    print(f"\n✅ Index généré et sauvegardé dans '{dossier_index}' !")
    print("Vous pouvez maintenant lancer le script de requête.")