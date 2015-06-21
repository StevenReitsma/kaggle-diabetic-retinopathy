mogrify -path ./../data/test_256 -fuzz 10%% -trim -gravity center -resize 256x256 -extent 256x256 -background black -quality 100 ./../data/test_circle_masked/*.jpeg
python hist_eq_parallel.py ./../data/test_256/ ./../data/test_256_he/ HE
python hist_eq_parallel.py ./../data/test_256/ ./../data/test_256_clahe/ CLAHE
python hist_eq_parallel.py ./../data/test_256/ ./../data/test_256_clahe_g/ CLAHE_G
pause