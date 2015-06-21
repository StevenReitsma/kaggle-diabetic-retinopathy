mogrify -path ./../data/train_magick -fuzz 10% -trim -gravity center -resize 256x256 -extent 256x256 -background black -equalize -quality 100 ./../data/train_circle_masked/*.jpeg
