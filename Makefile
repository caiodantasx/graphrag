.PHONY: up down install build ask app test eval demo

up:            ## sobe o Neo4j (com GDS)
	docker compose up -d neo4j
down:
	docker compose down
install:
	pip install -r requirements.txt
build:         ## schema + ingestão + algoritmos no Neo4j
	python -m graphrag build --reset
app:           ## interface web
	streamlit run app/streamlit_app.py
demo:          ## interface sem Docker (backend em memória)
	GRAPH_BACKEND=memory streamlit run app/streamlit_app.py
test:
	pytest
eval:
	python -m graphrag evaluate && python -m graphrag evaluate --questions data/eval_teste.json
