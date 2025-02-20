run:
	py main.py

small:
	py main.py --small

cached:
	py main.py --cached

add:
	py main.py --add

clean-text:
	py clean.py

clean:
	powershell -c 'rm ./chroma/* -Force -Recurse'
	touch ./chroma/.gitkeep

mangle:
	py mangler.py

docker:
	docker run -d --cpus=1.5 --gpus=all -v ollama:/root/.ollama \
	-p 11434:11434 --name ollama ollama/ollama:latest

kill-docker:
	docker kill ollama
	docker rm ollama