mogrify -path ./../data/train_512 -fuzz 12%% -trim -gravity center -resize 512x512 -extent 512x512 -background black -quality 100 ./../data/train/*.jpeg
python hist_eq_parallel.py ./../data/train_512/ ./../data/train_512_clahe/ CLAHE

mogrify -path ./../data/test_512 -fuzz 12%% -trim -gravity center -resize 512x512 -extent 512x512 -background black -quality 100 ./../data/test/*.jpeg
python hist_eq_parallel.py ./../data/test_512/ ./../data/test_512_clahe/ CLAHE
pause