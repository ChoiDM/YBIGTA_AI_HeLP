# YBIGTA_AMC

### 사용법
1. repo clone
```bash
$ git clone https://github.com/ChoiDM/YBIGTA_AI_HeLP.git
```

2. repo로 이동
```bash
$ cd YBIGTA_AI_HeLP
```

3. docker image 생성
```bash
$ docker build --tag <image name>:0.0.1 .
# 예시: dokcer build --tag test:0.0.1 .
```

4. docker image 압축
```bash
$ docker save <image name>:0.0.1 | gzip > <image name>.tar.gz
# 예시 docker save test:0.0.1 | gzip > test.tar.gz
```

5. 
