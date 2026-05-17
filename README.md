# RAG Project — Retrieval Augmented Generation

Application de questions-réponses sur des documents PDF, basée sur l'architecture RAG (Retrieval Augmented Generation). L'utilisateur charge des PDFs, les données sont indexées dans une base vectorielle, et un LLM répond aux questions en se basant uniquement sur le contenu des documents.

---

## Architecture

```
PDF(s)
  │
  ▼
Extraction du texte (pypdf)
  │
  ▼
Découpage en chunks (RecursiveCharacterTextSplitter)
  │
  ▼
Génération des embeddings (sentence-transformers/all-MiniLM-L6-v2)
  │
  ▼
Stockage vectoriel (ChromaDB)
  │
  ▼
Question utilisateur ──► Recherche similarité ──► Contexte pertinent
                                                        │
                                                        ▼
                                              LLM Groq (Llama 3.3 70B)
                                                        │
                                                        ▼
                                                    Réponse
```

---

## Technologies utilisées

| Composant | Technologie |
|---|---|
| Interface | Streamlit |
| Lecture PDF | pypdf |
| Découpage texte | LangChain RecursiveCharacterTextSplitter |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (local) |
| Base vectorielle | ChromaDB |
| LLM | Groq — Llama 3.3 70B Versatile |
| Orchestration | LangChain |
| Gestion dépendances | uv |

---

## Structure du projet

```
RAG-project/
├── rag.py              # Application Streamlit principale
├── RAGV2.ipynb         # Notebook Jupyter d'expérimentation
├── pyproject.toml      # Dépendances du projet (uv)
├── uv.lock             # Versions exactes des dépendances
├── .python-version     # Version Python utilisée
├── .gitignore
└── README.md
```

---

## Prérequis

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) — gestionnaire de paquets
- Une clé API [Groq](https://console.groq.com) (gratuite)

---

## Installation

**1. Cloner le repo**
```bash
git clone https://github.com/kevin20bf/-RAG-project.git
cd -RAG-project
```

**2. Installer les dépendances**
```bash
uv sync
```

**3. Configurer les variables d'environnement**

Créer un fichier `.env` à la racine :
```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx
```

---

## Lancement

```bash
uv run streamlit run rag.py
```

L'application sera accessible sur : http://localhost:8501

---

## Utilisation

1. **Charger des PDFs** — Glisser-déposer ou sélectionner un ou plusieurs fichiers PDF dans la sidebar
2. **Indexer** — Cliquer sur **Soumettre** pour extraire et vectoriser le contenu
3. **Poser une question** — Saisir une question dans le champ texte
4. **Obtenir une réponse** — Le LLM répond en se basant uniquement sur le contenu des PDFs chargés

---

## Notebook (RAGV2.ipynb)

Le notebook permet d'expérimenter l'architecture RAG de façon interactive :
- Chargement de PDFs depuis un dossier `./pdfs/`
- Construction de la base vectorielle avec persistance dans `./store/`
- Interrogation du retriever
- Évaluation de la qualité des réponses avec un système de notation (groundedness)

Pour exécuter le notebook, sélectionner le kernel **Python (ragg)** dans VSCode.

---

## Dépendances principales

```toml
langchain
langchain-community
langchain-groq
langchain-huggingface
langchain-text-splitters
chromadb
sentence-transformers
streamlit
pypdf
python-dotenv
```

---

## Remarques

- Les embeddings sont générés **localement** (aucune clé API nécessaire pour cette étape)
- Le modèle `all-MiniLM-L6-v2` (~90MB) est téléchargé automatiquement au premier lancement
- Le dossier `store/` (base vectorielle) et le dossier `pdfs/` sont exclus du repo git
- Ne jamais committer le fichier `.env`
