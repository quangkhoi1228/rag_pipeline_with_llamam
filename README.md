---
noteId: '6905a530769f11efa4add19c7c2db0fe'
tags: []
---

# Setup

## Docker db

```bash

docker run --name db_llm -e POSTGRES_PASSWORD=123456 -e POSTGRES_USER=postgres -e POSTGRES_DB=db_llm -p 5432:5432 -d postgres
```

## Run milvus

```sh
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh

bash standalone_embed.sh start

```

## Run Attu milvus UI

```sh
docker run -p 8001:3000 -e MILVUS_URL=172.20.10.2:19530 zilliz/attu:v2.4
```

## Run server backend

```sh
fastapi dev backend/main.py
```

## Run server frontend

```sh
streamlit run frontend/gui.py
```

## Postgres tables

```sh

CREATE TABLE page_content (
    type TEXT,
    id INT,
    url TEXT,
    title TEXT,
    created_date TIMESTAMP,
    updated_date TIMESTAMP,
    content TEXT,
    content_raw TEXT,
    PRIMARY KEY (type, id)
);



CREATE TABLE room (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE chat (
    message Text NOT NULL,
    sender  TEXT NOT NULL,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


CREATE TABLE faq_pool (
    id TEXT PRIMARY KEY,
    faq_id TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


CREATE TABLE public.feedback (
  faq_id text NULL,
  faq_pool_id text NULL,
  feedback text NOT NULL,
  created_date timestamp DEFAULT CURRENT_TIMESTAMP NOT NULL,
  handled bool DEFAULT false NOT NULL,
  updated_date timestamp DEFAULT CURRENT_TIMESTAMP NOT NULL
);


```

## Timeline

28 day
=> 20 days

### 6/9: Done pipeline cơ bản

### Deadline 16/9

- Đảnh: Crawl data QA

- Phong: Migrate/Import data -> Keyword: chunking, search theo dense/ sparse
  ==> Embedding được content

- APP

  - Khôi: Backend
  - Nhân: Frontend
    - Database .x

- Đạt: Benchmark

#################

# 26/9

2. Data embedding: Phong

- Target: Toàn bộ dữ liệu được embedding: Title, content (text)

3. Backend: Khôi / Nhân

- Prompt: Đạt
- Human feedback: Khôi, Nhân

4. Frontend: Nhân

- Regenerate
- Clear
- Human feedback - like / dislike

5. Benchmark: Đạt / Đảnh

- Raw data Q&A: Đảnh
  - Target: 1000 ~ 2000 QA gần nhất

---

- Report

## Ref

- Diagram: https://drive.google.com/file/d/1Z9ETxum10o4LGWEoNcpxDxN5rzFS-ssS/view?usp=sharing
