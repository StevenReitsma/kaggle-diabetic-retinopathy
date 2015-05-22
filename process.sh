mogrify -path ./data/processed/test -fuzz 10% -trim -gravity center -resize 256x256 -extent 256x256 -background black -equalize -quality 100 ./data/test/*.jpeg
