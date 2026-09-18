# Smart Fridge GenAI POC -- Phase 2

A minimal proof of concept that tests how accurately **Qwen3-VL 8B** (a
local multimodal vision-language model, run through **Ollama**) can
identify food ingredients from a photo of a refrigerator, then persists
those ingredients as refrigerator inventory in **PostgreSQL**, reports the
current meal period, and suggests recipes by vector similarity search
over an embedded recipe dataset (**pgvector**).

Phase 1 was deliberately framework-free raw Python with no database. Phase
2 adds inventory storage, meal-time detection, and a minimal recipe RAG
retriever -- while still avoiding agents and FastAPI until those are
actually needed.

```
fridge.jpg -> Python -> Ollama -> Qwen3-VL 8B -> structured JSON
    -> Pydantic validation -> PostgreSQL inventory -> console

local system time -> deterministic Python -> current meal period

inventory + meal period -> text query -> Ollama embedding -> pgvector
    cosine search over indexed recipes -> ranked recipe suggestions
```

## 1. What this POC does

1. Loads `uploads/fridge.jpg`.
2. Validates it's a real, non-corrupted JPEG/PNG/WEBP of a reasonable size.
3. Base64-encodes it and sends it, with a detailed instruction prompt, to
   Qwen3-VL 8B via the local Ollama HTTP API.
4. Parses the model's text response as JSON.
5. Validates that JSON against a Pydantic schema (rejecting malformed
   output, enforcing `0.0 <= confidence <= 1.0`, and flagging
   low-confidence ingredients rather than silently dropping them).
6. Normalizes each ingredient name (strips descriptive words like "red" or
   "fresh") and upserts it into the `inventory` table in PostgreSQL, keyed
   by that normalized name -- re-detecting an ingredient updates its
   existing row rather than duplicating it (see "Known limitations" below
   for what this does and doesn't handle).
7. Prints the validated ingredient list, the full persisted inventory, and
   the current meal period (breakfast/lunch/snacks/dinner/other) computed
   deterministically from local system time -- not from the LLM.
8. Builds a text query from current inventory + meal period, embeds it,
   and prints the 5 nearest recipes from the pgvector index (section 11)
   by cosine similarity -- retrieval only, no LLM-generated explanation.

Qwen3-VL is a **vision-language model**: it can take both an image and a
text prompt as input and reason about what's in the image in natural
language. Here we're testing its raw visual food-recognition ability --
nothing more.

## 2. Prerequisites

- macOS
- Python 3.9+
- [Ollama](https://ollama.com) installed and runnable locally
- The `qwen3-vl:8b` model pulled into Ollama
- [Postgres.app](https://postgresapp.com) (PostgreSQL 17, includes pgvector) -- see section 4a below

## 3. Install Python dependencies

```bash
cd smartfridge
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 4. Verify Ollama is running

```bash
ollama list
```

You should see `qwen3-vl:8b` in the output. If Ollama itself isn't running,
you'll get a connection error instead of a model list -- start it with:

```bash
ollama serve
```

(or just open the Ollama desktop app, which runs the server in the
background). Leave that running in its own terminal/process, then re-run
`ollama list` in another terminal to confirm.

## 4a. Set up PostgreSQL

This project uses [Postgres.app](https://postgresapp.com) -- a standalone
`.app`, no Homebrew required -- installed to `/Applications/Postgres.app`,
with pgvector already bundled (used in a later phase for recipe search).

The database cluster lives at `~/postgres-data/smartfridge-pg17` (outside
this project folder on purpose -- a live database shouldn't sit inside a
cloud-synced directory). Because it was created directly via `initdb`
rather than through the Postgres.app GUI's own "Initialize" flow, **you
manage it from the command line, not by clicking the app icon**:

```bash
scripts/start_db.sh   # start PostgreSQL on port 5432
scripts/stop_db.sh    # stop it
```

One-time setup (already done if you're using the included database):

```bash
PG_BIN="/Applications/Postgres.app/Contents/Versions/17/bin"
"$PG_BIN/initdb" -D ~/postgres-data/smartfridge-pg17 -U postgres --auth=trust -E UTF8
scripts/start_db.sh
"$PG_BIN/createdb" -p 5432 -U postgres smart_fridge
"$PG_BIN/psql" -p 5432 -U postgres -d smart_fridge -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

`main.py` creates the `inventory` table itself on first run (`CREATE TABLE
IF NOT EXISTS`) -- no separate migration step needed for this phase.

To clear all persisted inventory rows and start fresh:

```bash
python scripts/reset_inventory.py
```

## 5. Download / run Qwen3-VL 8B

If `qwen3-vl:8b` isn't listed yet, pull it:

```bash
ollama pull qwen3-vl:8b
```

This downloads the ~6 GB model. `main.py` does **not** pull or manage the
model for you -- it only calls the model and gives you a clear error with
this exact command if the model is missing.

## 6. Where to place the fridge photo

Put your image at:

```
uploads/fridge.jpg
```

A sample image is already included there so you can run the POC
immediately. Supported formats are JPEG, PNG, and WEBP -- detected by
actual file content, not the file extension. Swap in your own photo by
overwriting `uploads/fridge.jpg`.

## 7. Configuration (optional)

Copy `.env.example` to `.env` to override defaults:

```bash
cp .env.example .env
```

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_VISION_MODEL` | `qwen3-vl:8b` | Model tag to call |
| `VISION_CONFIDENCE_THRESHOLD` | `0.60` | Ingredients below this confidence get `needs_confirmation: true` |
| `MAX_IMAGE_SIZE_MB` | `10` | Reject images larger than this |
| `DATABASE_URL` | `postgresql://postgres@localhost:5432/smart_fridge` | PostgreSQL connection string for inventory storage |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model for recipe search (section 11) |
| `EMBEDDING_DIMENSIONS` | `768` | Must match the embedding model's actual output size |

## 8. Run it

```bash
python main.py
```

## 9. Example output

```json
{
  "ingredients": [
    {
      "name": "potato",
      "estimated_quantity": "3",
      "unit": "pieces",
      "confidence": 0.94,
      "needs_confirmation": false
    },
    {
      "name": "tomato",
      "estimated_quantity": "4",
      "unit": "pieces",
      "confidence": 0.91,
      "needs_confirmation": false
    },
    {
      "name": "onion",
      "estimated_quantity": "2",
      "unit": "pieces",
      "confidence": 0.88,
      "needs_confirmation": false
    }
  ]
}
```

After the ingredient JSON, you'll also see the full persisted inventory,
the current meal period, and (if the recipe index from section 11 has
been built) recipe suggestions, e.g.:

```
Saved 13 ingredient(s) to inventory (source=vision).

Full inventory (13 row(s), persisted across runs):

  #1 apple: 1 piece (conf=0.95, source=vision)
  #2 banana: 5 pieces (conf=0.9, source=vision)
  ...

Current meal period: {"meal_period":"dinner","current_time":"19:42","timezone":"local"}

Recipe suggestions for dinner based on current inventory (nearest by embedding distance):

  1. Radicchio and Apple Salad with Parmesan Crisps (distance=0.3383)
     Preheat oven to 350°F. On a silicone mat-lined baking sheet...
  2. Mulled Pears and Apples (distance=0.3441)
     Fill a large, heavy pot with apple juice. Tie the cinnamon sticks...
```

The actual ingredients, quantities, and confidence scores depend entirely
on what Qwen3-VL sees in your specific `fridge.jpg` -- nothing here is
hardcoded.

## 10. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `Image validation failed: Image file not found` | No file at `uploads/fridge.jpg`. Add one. |
| `Image validation failed: Unsupported image format` | File isn't a real JPEG/PNG/WEBP (check with `file uploads/fridge.jpg`). |
| `Ollama check failed: Could not reach Ollama` | Ollama isn't running. Run `ollama serve` or open the Ollama app. |
| `Model not found` / `ollama pull qwen3-vl:8b` message | Model hasn't been pulled yet. Run the suggested `ollama pull` command. |
| `Ollama timed out` | The model is slow on your hardware (CPU-only Macs can take minutes for an 8B vision model). Try again, or use a smaller/quantized model. |
| `Model returned an invalid response` | The model didn't return valid JSON matching the schema. Usually transient -- re-run. Consistent failures may mean the model needs a lower `temperature` or a reworded prompt. |
| Response is slow | Vision models are compute-heavy. Performance depends heavily on your Mac's RAM/GPU (Apple Silicon with more unified memory will be noticeably faster). |
| `Inventory not saved -- Could not connect to the inventory database` | PostgreSQL isn't running. Run `scripts/start_db.sh`. |
| `No recipe suggestions -- run scripts/build_recipe_index.py first.` | The `recipes` table is empty. Follow section 11. |

## 11. Recipe search index (pgvector)

A subset of a public recipe dataset is embedded and stored for similarity
search. `main.py` uses it automatically (step 8 above) -- this section
covers how the index itself is built.

Dataset: [josephrmartinez/recipe-dataset](https://github.com/josephrmartinez/recipe-dataset)
(13,501 recipes with title, ingredients, and instructions). The CSV isn't
committed to this repo (25 MB, easily re-downloaded) -- fetch it yourself:

```bash
mkdir -p data/recipes
curl -L -o data/recipes/13k-recipes.csv \
  https://raw.githubusercontent.com/josephrmartinez/recipe-dataset/main/13k-recipes.csv
```

Pull the embedding model (small, ~274 MB, purpose-built for this -- not
the vision model):

```bash
ollama pull nomic-embed-text
```

Then build the index (creates tables if needed, embeds a reproducible
random subset of 1,000 recipes, skips re-running if already populated):

```bash
python scripts/init_db.py
python scripts/build_recipe_index.py
```

On this machine this took ~17 seconds for 1,000 recipes. Quick sanity
check that the embeddings are actually meaningful (not part of the app --
just a manual verification query):

```bash
python3 -c "
from pgvector.psycopg2 import register_vector
from app.inventory.inventory_service import get_connection
from app.rag.embeddings import embed_text

conn = get_connection()
register_vector(conn)
qvec = embed_text('chicken potato tomato onion dinner')
with conn.cursor() as cur:
    cur.execute('SELECT name, embedding <=> %s::vector AS distance FROM recipes ORDER BY embedding <=> %s::vector LIMIT 5', (qvec, qvec))
    for name, distance in cur.fetchall():
        print(f'{distance:.4f}  {name}')
"
```

## 11a. Second recipe dataset: Indian cuisine

A second dataset adds 6,865 Indian recipes into the *same* `recipes`
table, alongside the 1,000 from section 11. Place your copy at:

```
data/recipes/IndianFoodDatasetCSV.csv
```

(not committed to this repo, like the other dataset -- get your own copy;
it's the commonly available "Indian Food 101" / Archana's Kitchen recipe
CSV with columns `RecipeName, TranslatedRecipeName, Ingredients,
TranslatedIngredients, PrepTimeInMins, CookTimeInMins, TotalTimeInMins,
Servings, Cuisine, Course, Diet, Instructions, TranslatedInstructions,
URL`).

This dataset is richer than the first: it has real `Cuisine`, `Course`
(meal type), `Servings`, `CookTimeInMins`, and `Diet` values, which
populate the `cuisine` / `meal_type` / `servings` / `cooking_time_minutes`
/ `vegetarian` columns that stayed `NULL` for every row from section 11.

Build it (indexes the **full** dataset, not a subset -- at the ~17s/1,000
rate measured in section 11, 6,865 recipes takes about 2.5 minutes; skips
entirely if already indexed):

```bash
python scripts/build_indian_recipe_index.py
```

`vegetarian` is only set from unambiguous `Diet` values ("Vegetarian",
"Vegan", "Eggetarian", "No Onion No Garlic (Sattvic)" -> true;
"Non Vegeterian", "High Protein Non Vegetarian" -> false). Ambiguous
labels like "Diabetic Friendly" or "Gluten Free" say nothing about meat
content, so those are left `NULL` rather than guessed -- see
`_infer_vegetarian()` in `scripts/build_indian_recipe_index.py`.

## 12. How recipe suggestions are built

`app/rag/recipe_retriever.py` is the actual "Recipe RAG" piece:

1. Reads distinct ingredient names from the `inventory` table.
2. Gets the current meal period (section 4's meal-time logic).
3. Joins them into one text query, e.g. `"apple banana egg milk tomato dinner"`.
4. Embeds that query with the same `nomic-embed-text` model used for indexing.
5. Runs a pgvector cosine-similarity search (`<=>` operator) against the
   `recipes` table, optionally filtered, and returns the closest matches.

This is retrieval only -- an LLM never ranks or explains the results.

**Optional filters** (blank/unset by default -- no behavior change), set in `.env`:

| Variable | Example | Effect |
|---|---|---|
| `RECIPE_VEGETARIAN_ONLY` | `true` | Only rows with `vegetarian = true` (or `false` for non-veg only) |
| `RECIPE_CUISINE_FILTER` | `Indian` | Only rows where `cuisine` contains this text (case-insensitive) |
| `RECIPE_MAX_COOK_TIME_MINUTES` | `30` | Excludes rows with a known cook time over this; rows with no cook time recorded are never excluded |

Filters only affect rows that actually have that metadata -- every row
from section 11a has it, every row from section 11 doesn't (see its
Known limitations entry). If a filter excludes every match, `main.py`
says so explicitly rather than telling you to rebuild the index.

**A real bug found and fixed here, worth knowing if you touch this code:**
combining a WHERE filter with the HNSW vector index can silently return
*zero* rows even when hundreds of matches exist. `EXPLAIN` on a filtered
query showed Postgres running the approximate HNSW index scan first and
applying `Filter: (NOT vegetarian)` *after* -- since HNSW only explores a
small candidate window, if none of that window happens to match the
filter, everything gets filtered out despite matches existing elsewhere
in the table. Fixed in `find_similar()` by forcing an exact scan
(`SET LOCAL enable_indexscan = off`) whenever a filter is present -- cheap
and fully correct at this table's size (a few thousand rows), though it
wouldn't scale to a much larger corpus without a different approach (e.g.
pgvector's per-query `hnsw.ef_search` tuning, or a partial index per
filter value).

## Known limitations (Phase 2)

- **Re-detecting an ingredient updates it in place, it doesn't remove
  stale ones.** Inventory is upserted by case-insensitive name (a unique
  index on `LOWER(name)`) -- re-running on the same fridge photo refreshes
  quantity/confidence/`updated_at` for ingredients still detected, rather
  than piling up duplicate rows. But if an ingredient disappears from a
  later photo (used up, thrown out), its row is simply never touched again
  -- nothing removes or expires it. A "last seen" timestamp already exists
  (`updated_at`) for a future staleness check; nothing acts on it yet.
- **Descriptive words are stripped before matching, which can over-merge.**
  The model alternates between e.g. "apple" and "red apple" for the same
  fruit across runs, so `normalize_ingredient_name()` in
  `inventory_service.py` strips a fixed list of color/ripeness/size words
  before storing/matching a name. This correctly merges "apple"/"red
  apple", but will also merge genuinely distinct items sharing a base word
  -- e.g. "green onion" and "onion" become one row, losing a real
  distinction. Edit `_DESCRIPTIVE_WORDS` if a specific case matters to you.
  When two entries in the *same* photo normalize to the same name (e.g.
  one red apple + one green apple), `_merge_same_name_ingredients()` sums
  their quantities (1 + 1 = 2) rather than silently keeping only one --
  but only when both quantities are plain integers with matching units;
  otherwise it falls back to the higher-confidence entry's quantity/unit,
  since e.g. "unknown" and "2" or "piece" and "bag" can't be added.
- **`needs_confirmation` isn't always reliable from the model itself.**
  The Pydantic validator force-flags anything below
  `VISION_CONFIDENCE_THRESHOLD`, but it never un-flags something the model
  already marked `true` at higher confidence -- over-flagging is the safe
  direction.
- **Cuisine/meal-type/servings/vegetarian metadata exists for only part
  of the corpus.** The section 11 dataset lacks those fields entirely
  (still `NULL` for those 1,000 rows); the section 11a dataset has them.
  Either way, nothing in the retriever uses this metadata yet -- recipe
  suggestions are ranked by embedding similarity only, not filtered by
  diet or cook-time, even though the data to do so is now partly there.
- **The retriever is retrieval-only, not a recommendation engine.** It
  finds the nearest recipes by vector distance; it doesn't check whether
  you actually have *all* the ingredients a recipe needs, weigh
  `needs_confirmation` items differently, or explain *why* a recipe was
  suggested. An LLM-based ranking/explanation step is a further phase.

## What's explicitly out of scope for this POC

By design, this project still does **not** include: LangChain, LangGraph,
CrewAI, MCP, an LLM-based agent with tool-calling, YOLO, FastAPI, or any
external search/API integration. Retrieval (section 12) is now real and
working, but recipe ranking/explanation and multi-step agent behavior are
still out of scope. The goal remains to add one real capability at a time
rather than reaching for a framework before it's needed.
