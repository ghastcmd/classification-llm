run:
	py main.py

add:
	py main.py --add

clean:
	powershell -c 'rm ./chroma/*'
	touch ./chroma/.gitkeep