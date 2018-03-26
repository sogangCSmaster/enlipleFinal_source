# enlipleML
enliple machine learning code (separate from web development)

# How to use
1. 먼저 data폴더 안에 크롤링 된 네이버 기사를 집어 넣는다.
2. 폴더 디렉토리는 다음과 같이 설정해 놓아야 한다.
3. data/group1[묶어 준 그룹]/cate_264/*.csv(csv 파일들)
4. 22개 그룹으로 묶었을 때 예 : data/cate_01/cate_264/*.csv
5. 데이터폴더와 코드들이 있는 경로 터미널에서 python3 csv2txt.py (python3.5 csv2txt.py 혹은 python3.6 csv2txt.py 컴퓨터 설정에 따라)를 입력 후 잠시 기다린다.
6. python3 cnn_train.py (python3.5 cnn_train.py 혹은 python3.6 cnn_train.py 컴퓨터 설정에 따라)를 입력 후 학습을 한다.
