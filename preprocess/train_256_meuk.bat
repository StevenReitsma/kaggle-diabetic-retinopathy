mogrify -path ./../data/train_256_meuk -fuzz 10%% -trim -gravity center -resize 256x256 -extent 256x256 -background black -quality 100 ./../data/train/*.jpeg
