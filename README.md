# PDF Search Engine / Moteur de Recherche PDF

A high-performance search engine for PDF document collections featuring lexical (BM25), semantic (vector embedding), and hybrid search modes.

Un moteur de recherche haute performance pour collections de documents PDF offrant des modes de recherche lexical (BM25), semantic (vector embedding), et hybrid.

---

## Table of Contents / Table des Matières

- [Features / Fonctionnalités](#features--fonctionnalités)
- [Search Modes / Modes de Recherche](#search-modes--modes-de-recherche)
- [Requirements / Prérequis](#requirements--prérequis)
- [Installation](#installation)
- [Usage / Utilisation](#usage--utilisation)
- [Configuration](#configuration)
- [Testing / Tests](#testing--tests)
- [Project Structure / Structure du Projet](#project-structure--structure-du-projet)
- [License / Licence](#license--licence)

---

## Features / Fonctionnalités

**English:**
- Three search modes: lexical (BM25), semantic (vector similarity), and hybrid (combined)
- Semantic search using multilingual-e5-large embedding model
- Hybrid search with Reciprocal Rank Fusion (RRF) algorithm
- Full-text search with BM25 relevance ranking
- Support for French accented characters (unicode61 tokenizer)
- Multiple PDF extraction backends (PyPDF2, pdfplumber) with automatic fallback
- Advanced search operators: AND, OR, NOT, phrase search ("exact phrase")
- Prefix wildcard search (aviat*)
- Intelligent text chunking for semantic indexing
- Vector storage using sqlite-vec extension
- Web interface built with Streamlit
- Incremental indexing with duplicate detection
- Pagination and result highlighting

**Français:**
- Trois modes de recherche : lexical (BM25), semantic (similarité vectorielle), et hybrid (combiné)
- Recherche semantic utilisant le modèle embedding multilingual-e5-large
- Recherche hybrid avec algorithme Reciprocal Rank Fusion (RRF)
- Recherche plein texte avec classement par pertinence BM25
- Support des caractères accentués français (tokenizer unicode61)
- Multiples backends d'extraction PDF (PyPDF2, pdfplumber) avec fallback automatique
- Opérateurs de recherche avancés : AND, OR, NOT, recherche par phrase ("phrase exacte")
- Recherche par préfixe (aviat*)
- Chunking intelligent du texte pour l'indexation semantic
- Stockage vectoriel utilisant l'extension sqlite-vec
- Interface web construite avec Streamlit
- Indexation incrémentale avec détection des doublons
- Pagination et mise en évidence des résultats

---

## Search Modes / Modes de Recherche

### Lexical Search (BM25)

**English:**
Traditional keyword-based search using SQLite FTS5 with BM25 ranking. Optimal for exact term matching, document codes, and phrase searches.

**Français:**
Recherche traditionnelle par mots-clés utilisant SQLite FTS5 avec classement BM25. Optimal pour la correspondance exacte de termes, les codes de documents, et les recherches par phrase.

### Semantic Search

**English:**
Vector-based search using the multilingual-e5-large embedding model. Documents and queries are converted to 1024-dimensional vectors, enabling meaning-based matching. Handles synonyms, paraphrasing, and multilingual queries effectively.

**Français:**
Recherche vectorielle utilisant le modèle embedding multilingual-e5-large. Les documents et requêtes sont convertis en vecteurs de 1024 dimensions, permettant une correspondance basée sur le sens. Gère efficacement les synonymes, les reformulations, et les requêtes multilingues.

### Hybrid Search

**English:**
Combines lexical and semantic search results using Reciprocal Rank Fusion (RRF). Provides the best coverage by leveraging both exact matching and semantic understanding. Documents found by both methods receive higher ranking.

**Français:**
Combine les résultats des recherches lexical et semantic en utilisant Reciprocal Rank Fusion (RRF). Offre la meilleure couverture en exploitant à la fois la correspondance exacte et la compréhension semantic. Les documents trouvés par les deux méthodes reçoivent un classement supérieur.

### Mode Selection Guide / Guide de Sélection des Modes

| Use Case / Cas d'usage | Recommended Mode / Mode recommandé |
|------------------------|-----------------------------------|
| Exact terms, document codes / Termes exacts, codes de documents | Lexical |
| Conceptual queries / Requêtes conceptuelles | Semantic |
| General search (default) / Recherche générale (défaut) | Hybrid |
| Phrase search / Recherche par phrase | Lexical |
| Multilingual queries / Requêtes multilingues | Semantic |

---

## Requirements / Prérequis

- Python 3.10+
- SQLite with FTS5 support (included in Python standard library)
- sqlite-vec extension (for semantic search)

---

## Installation

```bash
# Clone the repository / Cloner le dépôt
git clone https://github.com/VirgileOrange/DGAC.git
cd DGAC

# Create virtual environment / Créer l'environnement virtuel
python -m venv venv

# Activate virtual environment / Activer l'environnement virtuel
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies / Installer les dépendances
pip install -r requirements.txt
```

---

## Usage / Utilisation

### 1. Configure paths / Configurer les chemins

Edit `config/config.json` to set your PDF directory:

Modifier `config/config.json` pour définir votre répertoire PDF :

```json
{
  "paths": {
    "data_directory": "path/to/your/pdf/folder",
    "database_path": "output/search.db"
  }
}
```

### 2. Index documents / Indexer les documents

```bash
python -m src.indexer.index_builder
```

Options:
- `--reset` : Clear existing index and rebuild from scratch / Effacer l'index existant et reconstruire

### 3. Launch web interface / Lancer l'interface web

```bash
streamlit run src/gui/app.py
```

The application will be available at `http://localhost:8501`

L'application sera disponible sur `http://localhost:8501`

### 4. Search programmatically / Recherche programmatique

**Hybrid search (recommended) / Recherche hybrid (recommandée):**

```python
from src.search.hybrid_engine import HybridEngine, SearchMode

engine = HybridEngine()
results, stats = engine.search(
    query="aviation civile",
    mode=SearchMode.HYBRID,
    limit=20
)

for result in results:
    print(f"{result.filename} - Page {result.page_num}")
    print(f"  Score: {result.score:.4f}")
    print(f"  Source: {result.source}")  # 'lexical', 'semantic', or 'both'
    print(f"  {result.snippet}")
```

**Lexical search only / Recherche lexical uniquement:**

```python
from src.search.bm25_engine import BM25Engine
from src.search.models import SearchQuery

engine = BM25Engine()
query = SearchQuery(text="aviation civile", limit=20)
results, stats = engine.search(query)

for result in results:
    print(f"{result.filename} - Page {result.page_num}")
    print(f"  Score: {result.display_score:.2f}")
    print(f"  {result.snippet}")
```

**Semantic search only / Recherche semantic uniquement:**

```python
from src.search.semantic_engine import SemanticEngine

engine = SemanticEngine()
results, stats = engine.search(query="procedures de securite", limit=20)

for result in results:
    print(f"{result.filename} - Page {result.page_num}")
    print(f"  Similarity: {result.similarity:.4f}")
    print(f"  {result.snippet}")
```

### Advanced Search Syntax / Syntaxe de Recherche Avancée

| Operator | Example | Description (EN) | Description (FR) |
|----------|---------|------------------|------------------|
| AND | `aviation AND security` | Both terms required | Les deux termes requis |
| OR | `aviation OR maritime` | Either term | L'un ou l'autre terme |
| NOT | `aviation NOT military` | Exclude term | Exclure un terme |
| "..." | `"civil aviation"` | Exact phrase | Phrase exacte |
| * | `aviat*` | Prefix search | Recherche par préfixe |

---

## Configuration

The `config/config.json` file contains all configurable parameters:

Le fichier `config/config.json` contient tous les paramètres configurables :

### Core Parameters / Paramètres Principaux

| Section | Parameter | Description (EN) | Description (FR) |
|---------|-----------|------------------|------------------|
| paths | data_directory | PDF source folder | Dossier source des PDF |
| paths | database_path | SQLite database location | Emplacement de la base SQLite |
| extraction | primary_backend | First extraction method (pypdf2/pdfplumber) | Méthode d'extraction principale |
| extraction | fallback_backend | Backup extraction method | Méthode d'extraction de secours |
| indexing | batch_size | Documents per batch | Documents par lot |
| indexing | skip_existing | Skip already indexed files | Ignorer les fichiers déjà indexés |
| search | default_limit | Default results per page | Résultats par page par défaut |
| search | snippet_length | Result snippet length | Longueur des extraits |

### Semantic Search Parameters / Paramètres de Recherche Semantic

| Section | Parameter | Description (EN) | Description (FR) |
|---------|-----------|------------------|------------------|
| semantic | enabled | Enable semantic indexing | Activer l'indexation semantic |
| semantic | endpoint | Embedding API endpoint | Endpoint de l'API embedding |
| semantic | api_key | API key for embedding service | Clé API pour le service embedding |
| semantic | embedding_model | Embedding model name | Nom du modèle embedding |
| semantic | embedding_dimensions | Vector size (1024 for E5-large) | Taille du vecteur (1024 pour E5-large) |
| semantic | max_chunk_chars | Max characters per chunk | Caractères maximum par chunk |
| semantic | chunk_overlap_chars | Overlap between chunks | Chevauchement entre chunks |
| semantic | embedding_batch_size | Batch size for embedding | Taille de lot pour embedding |

### Hybrid Search Parameters / Paramètres de Recherche Hybrid

| Section | Parameter | Description (EN) | Description (FR) |
|---------|-----------|------------------|------------------|
| hybrid | default_mode | Default search mode (lexical/semantic/hybrid) | Mode de recherche par défaut |
| hybrid | rrf_k | RRF dampening constant (default: 60) | Constante d'amortissement RRF (défaut : 60) |
| hybrid | default_lexical_weight | Weight for lexical results | Poids pour les résultats lexical |
| hybrid | default_semantic_weight | Weight for semantic results | Poids pour les résultats semantic |

---

## Testing / Tests

The project includes a comprehensive test suite covering all modules.

Le projet inclut une suite de tests complète couvrant tous les modules.

```bash
# Run all tests / Exécuter tous les tests
pytest

# Run with coverage / Exécuter avec couverture
pytest --cov=src --cov-report=html

# Run specific test file / Exécuter un fichier de test spécifique
pytest tests/test_search/test_bm25_engine.py -v
```

### Test Structure / Structure des Tests

```
tests/
├── conftest.py              # Shared fixtures / Fixtures partagées
├── test_core/               # Core module tests
├── test_database/           # Database tests
├── test_extraction/         # PDF extraction tests
├── test_indexer/            # Indexing tests
├── test_search/             # Search engine tests
├── test_utils/              # Utility function tests
└── test_integration/        # End-to-end tests
```

---

## Project Structure / Structure du Projet

```
├── config/
│   └── config.json              # Application configuration
├── src/
│   ├── core/                    # Configuration, logging, exceptions
│   ├── database/
│   │   ├── connection.py        # SQLite connection management
│   │   ├── schema.py            # FTS5 table definitions
│   │   ├── repository.py        # Document CRUD operations
│   │   └── vector_repository.py # Vector embedding storage (sqlite-vec)
│   ├── extraction/
│   │   ├── extractor.py         # PDF text extraction with fallback
│   │   └── semantic_chunker.py  # Text chunking for embedding
│   ├── indexer/
│   │   ├── index_builder.py     # Lexical indexing pipeline
│   │   └── semantic_indexer.py  # Semantic indexing pipeline
│   ├── search/
│   │   ├── bm25_engine.py       # Lexical search (BM25)
│   │   ├── semantic_engine.py   # Semantic search (vector similarity)
│   │   ├── hybrid_engine.py     # Hybrid search with RRF fusion
│   │   ├── embedding_service.py # Embedding generation service
│   │   └── query_parser.py      # FTS5 query parsing
│   ├── utils/                   # File and text utilities
│   └── gui/                     # Streamlit web interface
├── tests/                       # Test suite
├── assets/                      # CSS styles, logo
├── .github/workflows/           # CI/CD pipeline
├── requirements.txt             # Python dependencies
└── pytest.ini                   # Test configuration
```

---

## CI/CD

This project uses GitHub Actions for continuous integration:

Ce projet utilise GitHub Actions pour l'intégration continue :

- Automated testing on Python 3.13
- Code coverage reporting
- Linting with Ruff

Tests run automatically on every push and pull request.

Les tests s'exécutent automatiquement à chaque push et pull request.

---

## Architecture

### Search Pipeline / Pipeline de Recherche

```
User Query
    |
    v
+------------------+
|  Hybrid Engine   |  <-- Entry point / Point d'entrée
+--------+---------+
         |
    +----+----+
    |         |
    v         v
+-------+  +----------+
| BM25  |  | Semantic |
+---+---+  +----+-----+
    |           |
    v           v
  FTS5      sqlite-vec
  Index       Index
    |           |
    +-----+-----+
          |
          v
    RRF Fusion
          |
          v
   Unified Results / Résultats Unifiés
```

### Indexing Pipeline / Pipeline d'Indexation

```
PDF Files
    |
    v
+---------------+
| PDF Extractor |
+-------+-------+
        |
   +----+----+
   |         |
   v         v
+-------+  +-----------+
| FTS5  |  | Semantic  |
| Index |  | Indexer   |
+-------+  +-----+-----+
                 |
           +-----+-----+
           |           |
           v           v
      +--------+  +-----------+
      |Chunker |  | Embedding |
      +--------+  | Service   |
                  +-----------+
                       |
                       v
                  sqlite-vec
                    Index
```


