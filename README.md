# Setup

## Docker db

```bash

docker run --name db_llm -e POSTGRES_PASSWORD=123456 -e POSTGRES_USER=postgres -e POSTGRES_DB=db_llm -p 5432:5432 -d postgres
```

## DB table

```bash
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


CREATE TABLE faq (
    question TEXT,
    answer TEXT,
    PRIMARY KEY (question)
);
INSERT INTO faq (question, answer) VALUES
('What is the return policy?', 'You can return items within 30 days of purchase.'),
('How do I reset my password?', 'Click ''Forgot Password'' on the login page.'),
('What are the store hours?', 'We are open from 9 AM to 9 PM, Monday to Saturday.'),
('Where can I find my order history?', 'Log in to your account and go to ''Order History''.'),
('Do you offer international shipping?', 'Yes, we ship to many countries worldwide.');

```

## Run milvus

```sh
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh

bash standalone_embed.sh start

```


## Run server backend

```sh
fastapi dev main.py
```


## Postgres 

```sh
CREATE TABLE room (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
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


