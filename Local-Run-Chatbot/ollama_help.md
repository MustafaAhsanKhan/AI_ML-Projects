## 🔹 Model Management

### Download a model

```bash
ollama pull llama3:8b-instruct-q4_0
```

### List downloaded models

```bash
ollama list
```

### Remove a model

```bash
ollama rm llama3:8b-instruct-q4_0
```

### Show model details

```bash
ollama show llama3
```

---

## 🔹 Running Models

### Run a model in interactive mode

```bash
ollama run llama3
```

### Run with a single prompt

```bash
ollama run llama3 "Explain recursion simply"
```

### Stop a running model (free RAM)

```bash
ollama stop llama3
```

---

## 🔹 Server Control

### Start server manually

```bash
ollama serve
```

### Check if server is running

```bash
lsof -i :11434
```

### Kill Ollama completely

```bash
pkill ollama
```

---

## 🔹 Custom Models (Personality Control)

### Create Modelfile

```bash
nano Modelfile
```

Example content:

```
FROM llama3

SYSTEM "You are a concise, logical assistant."
```

### Build custom model

```bash
ollama create myassistant -f Modelfile
```

### Run custom model

```bash
ollama run myassistant
```

---

## 🔹 API Testing

Test via curl:

```bash
curl http://localhost:11434/api/generate \
-d '{
  "model": "llama3",
  "prompt": "Explain neural networks simply",
  "stream": false
}'
```

---