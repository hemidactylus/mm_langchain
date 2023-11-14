# MM LC

Prepare and source the `.env`

Install the `requirements.txt`

## vanilla vector stores

(i.e. the slight refactor to this new structure)

Write to the store

```
python vanilla_loader.py
```

Try queries

```
python vanilla_querier.py
```

## multimodal vector stores

Write to the store

```
python mm_client_loader.py
```

Try queries

```
python mm_client_querier.py
```

### with a sample web-page MM loader

```
python mm_load_web_page.py
```

(you can query it with the above client querier -
functional test, don't expect meaningful results)

### multimodal misc tests

(just a scratchpad actually)

```
python -m mm_tests.mm_embedding_tests
```

loader test:

```
python -m mm_tests.test_mm_web_loader
```

is code all right?

```
black . && ruff . && mypy .
```
