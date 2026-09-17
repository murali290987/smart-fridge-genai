# Smart Fridge GenAI POC -- Phase 2

A minimal proof of concept that tests how accurately **Qwen3-VL 8B** (a
local multimodal vision-language model, run through **Ollama**) can
identify food ingredients from a photo of a refrigerator, then persists
those ingredients as refrigerator inventory in **PostgreSQL** and reports
the current meal period.

Phase 1 was deliberately framework-free raw Python with no database. Phase
2 adds the two smallest useful next steps -- inventory storage and
meal-time detection -- while still avoiding RAG, agents, and FastAPI until
those are actually needed.

```
fridge.jpg -> Python -> Ollama -> Qwen3-VL 8B -> structured JSON
    -> Pydantic validation -> PostgreSQL inventory -> console

local system time -> deterministic Python -> current meal period
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
6. Saves each detected ingredient as a new row in the `inventory` table in
   PostgreSQL (append-only -- it does not merge/dedupe against previous
   runs; see "Known limitations" below).
7. Prints the validated ingredient list, the full persisted inventory, and
   the current meal period (breakfast/lunch/snacks/dinner/other) computed
   deterministically from local system time -- not from the LLM.

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

After the ingredient JSON, you'll also see the full persisted inventory and
the current meal period, e.g.:

```
Saved 13 ingredient(s) to inventory (source=vision).

Full inventory (13 row(s), persisted across runs):

  #1 apple: 1 piece (conf=0.95, source=vision)
  #2 banana: 5 pieces (conf=0.9, source=vision)
  ...

Current meal period: {"meal_period":"dinner","current_time":"19:42","timezone":"local"}
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

## 11. Recipe search index (pgvector)

pgvector is now actually used: a subset of a public recipe dataset is
embedded and stored for similarity search, as groundwork for recipe RAG in
a later phase. **No retriever/recommendation logic is wired up yet** --
this section only loads and indexes the data.

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

## Known limitations (Phase 2)

- **No deduplication across runs.** Each run appends detected ingredients
  as new inventory rows; it does not merge "3 tomatoes" from yesterday with
  "1 tomato" from today. That's a deliberate next step, not implemented
  here to keep this phase's scope small.
- **The model doesn't always merge duplicates within a single response**
  either (e.g. "apple" and "red apple" as separate entries), despite the
  prompt asking it to -- a model accuracy limitation, not a code bug.
- **`needs_confirmation` isn't always reliable from the model itself.**
  The Pydantic validator force-flags anything below
  `VISION_CONFIDENCE_THRESHOLD`, but it never un-flags something the model
  already marked `true` at higher confidence -- over-flagging is the safe
  direction.
- **The recipe index has no retriever yet.** 1,000 recipes are embedded and
  searchable with raw SQL (section 11's sanity-check query), but nothing
  in the app combines current inventory + meal period into a recipe
  recommendation yet -- that's the next phase.
- **The indexed recipes have no cuisine/meal-type/servings/vegetarian
  metadata.** The source CSV doesn't include those fields, so the
  `recipes` table has nullable columns for them, left empty for now rather
  than guessed at.

## What's explicitly out of scope for this POC

By design, this project still does **not** include: LangChain, LangGraph,
CrewAI, MCP, RAG retrieval logic, agents, YOLO, FastAPI, or any external
search/API integration. PostgreSQL and pgvector are in place and now hold
1,000 embedded recipes (section 11), but nothing queries them as part of
an actual recommendation flow yet. The goal remains to add one real
capability at a time rather than reaching for a framework before it's
needed.
