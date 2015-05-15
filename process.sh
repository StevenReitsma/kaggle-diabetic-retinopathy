mogrify -path ./data/processed/train -fuzz 15% -trim -gravity center -resize 256x256 -extent 256x256 -background black ./data/train/*.jpeg
