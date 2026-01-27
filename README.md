# PDF Search Engine / Moteur de Recherche PDF

A high-performance lexical search engine for PDF document collections using SQLite FTS5 and the BM25 ranking algorithm.

Un moteur de recherche lexical haute performance pour collections de documents PDF utilisant SQLite FTS5 et l'algorithme de classement BM25.

---

## Table of Contents / Table des Matières

- [Features / Fonctionnalités](#features--fonctionnalités)
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
- Full-text search with BM25 relevance ranking
- Support for French accented characters (unicode61 tokenizer)
- Multiple PDF extraction backends (PyPDF2, pdfplumber) with automatic fallback
- Advanced search operators: AND, OR, NOT, phrase search ("exact phrase")
- Prefix wildcard search (aviat*)
- Web interface built with Streamlit
- Incremental indexing with duplicate detection
- Pagination and result highlighting

**Français:**
- Recherche plein texte avec classement par pertinence BM25
- Support des caractères accentués français (tokenizer unicode61)
- Multiples backends d'extraction PDF (PyPDF2, pdfplumber) avec fallback automatique
- Opérateurs de recherche avancés : AND, OR, NOT, recherche par phrase ("phrase exacte")
- Recherche par préfixe (aviat*)
- Interface web construite avec Streamlit
- Indexation incrémentale avec détection des doublons
- Pagination et mise en évidence des résultats

---

## Requirements / Prérequis

- Python 3.10+
- SQLite with FTS5 support (included in Python standard library)

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
│   └── config.json          # Application configuration
├── src/
│   ├── core/                # Configuration, logging, exceptions
│   ├── database/            # SQLite connection, schema, repository
│   ├── extraction/          # PDF text extraction backends
│   ├── indexer/             # Document indexing pipeline
│   ├── search/              # BM25 search engine, query parser
│   ├── utils/               # File and text utilities
│   └── gui/                 # Streamlit web interface
├── tests/                   # Test suite
├── assets/                  # CSS styles, logo
├── .github/workflows/       # CI/CD pipeline
├── requirements.txt         # Python dependencies
└── pytest.ini              # Test configuration
```

---

## CI/CD

This project uses GitHub Actions for continuous integration:

Ce projet utilise GitHub Actions pour l'intégration continue :

- Automated testing on Python 3.10, 3.11, 3.12
- Code coverage reporting
- Linting with Ruff

Tests run automatically on every push and pull request.

Les tests s'exécutent automatiquement à chaque push et pull request.


