mogrify -path ./../data/train_256 -fuzz 12%% -trim -gravity center -resize 256x256 -extent 256x256 -background black -quality 100 ./../data/train/*.jpeg
python hist_eq_parallel.py ./../data/train_256/ ./../data/train_256_clahe/ CLAHE
pause

mogrify -path ./../data/test_256 -fuzz 12%% -trim -gravity center -resize 256x256 -extent 256x256 -background black -quality 100 ./../data/test/*.jpeg
python hist_eq_parallel.py ./../data/test_256/ ./../data/test_256_clahe/ CLAHE
pause