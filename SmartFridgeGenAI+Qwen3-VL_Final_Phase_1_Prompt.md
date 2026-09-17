# Smart Fridge GenAI Food Assistant — Phase 1 POC

## Role

Act as a senior Python, GenAI, Computer Vision, RAG, and agentic-AI engineer.

Build a complete working **Smart Fridge GenAI Food Assistant Phase 1 POC** in Python.

The POC must be runnable locally on a **MacBook Pro with Apple M5 and 24 GB unified memory**.

The primary local multimodal model will be:

**Qwen3-VL 8B running through Ollama**

The application architecture must be modular so that OpenAI or Gemini can be added later without changing the core application.

Do not over-engineer Phase 1.

---

# 1. Business Goal

The system simulates a smart refrigerator.

A user uploads a photograph of the inside of a refrigerator.

The system should:

1. Analyze the refrigerator image.
2. Identify visible food ingredients.
3. Estimate approximate quantity where possible.
4. Assign confidence scores.
5. Store the detected ingredients as refrigerator inventory.
6. Determine the current meal period from local system time:
   - breakfast
   - lunch
   - snacks
   - dinner
   - other
7. Search a local recipe knowledge base using RAG.
8. Use PostgreSQL + pgvector for recipe storage and vector similarity search.
9. Use an LLM-based Food Agent with tool/function calling.
10. Optionally search external recipe information.
11. Optionally search cooking videos.
12. Cache external search responses in PostgreSQL.
13. Generate family-friendly meal recommendations.
14. Return structured JSON.
15. Provide Swagger/OpenAPI documentation through FastAPI.

---

# 2. Phase 1 Architecture

Implement:

```text
User
  |
  | Upload fridge image
  v
FastAPI
  |
  v
OpenCV / Pillow
(image validation + preprocessing)
  |
  v
Qwen3-VL 8B
via Ollama
  |
  v
Structured Ingredient Detection
  |
  v
Inventory Service
  |
  v
PostgreSQL
  |
  +--------------------------+
  |                          |
  v                          v
Inventory              Recipe Database
                              |
                              v
                       pgvector Embeddings
                              |
                              v
                         Recipe RAG
                              |
                    +---------+---------+
                    |                   |
                    v                   v
              Meal-Time Detector    Food Agent
                                        |
                         +--------------+--------------+
                         |              |              |
                         v              v              v
                    Recipe Tool    Google Search   YouTube/SerpApi
                         |              |              |
                         +--------------+--------------+
                                        |
                                   Search Cache
                                        |
                                        v
                                  Final LLM
                                        |
                                        v
                              Meal Recommendation
```

Do NOT implement MCP in Phase 1.

Use normal Python functions/tools for agent tool calling.

MCP should be documented only as a Phase 2 enhancement.

---

# 3. Local AI Model

Primary model:

```text
Qwen3-VL 8B
```

Runtime:

```text
Ollama
```

Expected Ollama endpoint:

```text
http://localhost:11434
```

The application must NOT download or manage the model automatically.

The README should instruct the user to install Ollama separately and run:

```bash
ollama pull qwen3-vl:8b
```

Then verify:

```bash
ollama list
```

The application should check whether Ollama is reachable during startup or when the vision service is called.

Provide a useful error if Ollama is not running.

---

# 4. Model Provider Abstraction

Do not hard-code Qwen3-VL throughout the application.

Create a provider abstraction.

Example:

```python
class VisionProvider:
    def analyze_image(self, image_path: str) -> dict:
        raise NotImplementedError
```

Implement:

```text
OllamaVisionProvider
```

using:

```text
Qwen3-VL 8B
```

Also create placeholders/interfaces for:

```text
OpenAIVisionProvider
GeminiVisionProvider
```

They do not need to be fully implemented in Phase 1.

Configuration:

```env
VISION_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_VISION_MODEL=qwen3-vl:8b
```

The rest of the application should not care which provider is being used.

---

# 5. Vision Task

Qwen3-VL must analyze the uploaded refrigerator image.

Prompt it as a food inventory detection system.

Use a prompt similar to:

```text
You are a food inventory detection system.

Analyze the refrigerator image.

Identify visible food ingredients.

For every visible ingredient provide:

- name
- estimated_quantity
- unit
- confidence

Do not invent ingredients that are not visibly present.

If quantity cannot reasonably be estimated, return "unknown".

Image-based quantity estimates are approximate.

Do not claim exact weight.

Do not claim freshness from an image.

Return structured JSON only.
```

The application must validate the model response with Pydantic.

---

# 6. Ingredient Schema

Create:

```python
class Ingredient(BaseModel):
    name: str
    estimated_quantity: str
    unit: str
    confidence: float
    needs_confirmation: bool = False
```

Create:

```python
class IngredientDetectionResponse(BaseModel):
    ingredients: list[Ingredient]
```

Confidence must be between:

```text
0.0 and 1.0
```

Use:

```env
VISION_CONFIDENCE_THRESHOLD=0.60
```

If:

```text
confidence < threshold
```

set:

```text
needs_confirmation = true
```

Do not silently discard low-confidence ingredients.

---

# 7. Image Processing

Use:

```text
OpenCV
Pillow
```

Create:

```text
app/vision/image_processor.py
```

Responsibilities:

- validate file type
- validate image size
- decode image
- check image dimensions
- optionally resize very large images
- normalize orientation if possible
- create a temporary processed image
- avoid permanently storing unnecessary copies

Supported formats:

```text
JPEG
PNG
WEBP
```

Configuration:

```env
MAX_IMAGE_SIZE_MB=10
```

Do not perform AI segmentation in Phase 1.

Do not train YOLO as part of the application.

YOLO training is a future model-training track.

---

# 8. Important Computer Vision Design

Phase 1 uses:

```text
OpenCV
+
Qwen3-VL
```

OpenCV is responsible for:

- image validation
- preprocessing
- resizing
- image manipulation

Qwen3-VL is responsible for:

- food recognition
- visual understanding
- semantic interpretation
- approximate ingredient identification

Do NOT claim OpenCV itself recognizes potatoes or tomatoes.

---

# 9. Optional Future YOLO Architecture

Document the future architecture in README:

```text
Fridge Image
      |
      v
YOLO Custom Food Detector
      |
      +---- High confidence ---> Inventory
      |
      +---- Low confidence ----> Qwen3-VL
                                      |
                                      v
                                Confirmation
```

Future YOLO classes may include:

```text
potato
tomato
onion
carrot
cucumber
capsicum
chicken
egg
```

Do not train YOLO in this Phase 1 application.

The dataset/training process should be documented as a future Phase 2/ML track.

---

# 10. Inventory Service

Create:

```text
app/inventory/inventory_service.py
```

Store detected ingredients in PostgreSQL.

Suggested table:

```text
inventory
---------
id
name
estimated_quantity
unit
confidence
needs_confirmation
source
created_at
updated_at
```

Source should be:

```text
vision
```

Provide:

```text
GET /api/v1/inventory
DELETE /api/v1/inventory
```

The delete endpoint can clear the current POC inventory.

---

# 11. Meal-Time Detection

Do NOT ask the LLM to determine the current time.

Use deterministic Python logic based on local system time.

Default configuration:

```json
{
  "breakfast": {
    "start": "06:00",
    "end": "10:30"
  },
  "lunch": {
    "start": "11:30",
    "end": "15:00"
  },
  "snacks": {
    "start": "15:00",
    "end": "18:00"
  },
  "dinner": {
    "start": "18:00",
    "end": "22:30"
  }
}
```

Outside these ranges:

```text
other
```

Create:

```text
app/meal_time/detector.py
```

Expose:

```text
GET /api/v1/meal-period
```

Return:

```json
{
  "meal_period": "lunch",
  "current_time": "13:25",
  "timezone": "local"
}
```

The meal period must be included in the recipe RAG query.

---

# 12. Recipe Knowledge Base

Use PostgreSQL as the primary database.

Use:

```text
pgvector
```

Do NOT use FAISS in Phase 1.

The reason is that PostgreSQL can hold:

- inventory
- recipes
- metadata
- embeddings
- search cache

in one system.

---

# 13. Recipe Table

Create a recipe table with approximately:

```text
id
recipe_id
name
cuisine
meal_type
servings
cooking_time_minutes
vegetarian
ingredients
instructions
embedding
created_at
```

The embedding dimension must be configurable and must match the selected embedding model.

Do not blindly hard-code a dimension without checking the selected embedding model.

---

# 14. Recipe Data

Create at least these sample recipes:

```text
1. Chicken Potato Curry
2. Vegetable Stir Fry
3. Egg Omelette
4. Chicken Rice
5. Vegetable Soup
6. Egg Fried Rice
```

Each recipe should contain:

- recipe name
- cuisine
- meal types
- servings
- cooking time
- vegetarian flag
- ingredient list
- instructions

Store the source recipe content in:

```text
data/recipes/
```

Use JSON files.

---

# 15. Embeddings

Create:

```text
app/rag/embeddings.py
```

Create an embedding provider abstraction:

```python
class EmbeddingProvider:
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError
```

Phase 1 should support a configurable provider.

Recommended initial configuration:

```env
EMBEDDING_PROVIDER=ollama
```

If the selected local embedding model is unavailable, provide a clear setup instruction rather than silently falling back.

Also create interfaces for:

```text
OpenAIEmbeddingProvider
HuggingFaceEmbeddingProvider
```

but they do not need to be fully implemented initially.

Use a suitable local embedding model supported by Ollama and document the exact model used.

---

# 16. Recipe Indexing

Create:

```text
scripts/init_db.py
scripts/build_recipe_index.py
```

Run:

```bash
python scripts/init_db.py
python scripts/build_recipe_index.py
```

The indexing process should:

1. Load recipe JSON files.
2. Build searchable recipe text.
3. Generate embeddings.
4. Store recipe metadata.
5. Store vectors in pgvector.

---

# 17. Recipe RAG

Create:

```text
app/rag/recipe_retriever.py
```

The query should combine:

```text
ingredients
+
meal period
+
optional cuisine
+
optional cooking time
```

Example:

```text
chicken potato tomato onion lunch
```

Perf