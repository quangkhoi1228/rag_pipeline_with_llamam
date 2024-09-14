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
fastapi dev main.py
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
    + Khôi: Backend
    + Nhân: Frontend
        - Database .x

- Đạt: Benchmark


*****
- Report 


## Ref
- Diagram: https://drive.google.com/file/d/1Z9ETxum10o4LGWEoNcpxDxN5rzFS-ssS/view?usp=sharing