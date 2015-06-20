mogrify -path ./../data/processed_512/train -fuzz 10% -trim -gravity center -resize 512x512 -extent 512x512 -background black -equalize -quality 100 ./data/train/*.jpeg
mogrify -path ./../data/processed_512/test -fuzz 10% -trim -gravity center -resize 512x512 -extent 512x512 -background black -equalize -quality 100 ./data/test/*.jpeg
