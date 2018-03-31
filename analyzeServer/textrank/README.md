# 분석 서버 설치

## JAVA 설치
```
sudo yum -y install java-1.8.0-openjdk java-1.8.0-openjdk-devel
```
- JAVA 설치 이후 경로 설정 (/etc/profile)
```
export LD_LIBRARY_PATH = /usr/local/lib:/usr/lib
export C_INCLUDE_PATH=/usr/local/include:/usr/include
export C_PLUS_INCLUDE_PATH=/usr/local/include/:usr/include
export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.102-1.b14.el7_2.x86_64
export PATH=$PATH:$JAVA_HOME/bin
export CLASSPATH=.:$JAVA_HOME/jre/lib:$JAVA_HOME/lib:$JAVA_HOME/lib/tools.jar
```

## python3.5.1 설치
- python3.5.1 설치
```
sudo yum -y update
sudo yum -y upgrade
sudo yum install -y yum-utils make wget
sudo yum install -y https://centos7.iuscommunity.org/ius-release.rpm
sudo yum-builddep python
wget https://www.python.org/ftp/python/3.5.1/Python-3.5.1.tgz
tar xzf Python-3.5.1.tgz
cd Python-3.5.1
./configure
make
sudo make altinstall
```
- alias 수정 (long run)
```
alias python3='/usr/local/bin/python3.5’
alias pip3='/usr/local/bin/pip3.5’
alias sudo='sudo '
```

- pip3.5 upgrade
```
sudo pip3 install —upgrade pip
혹은
sudo pip3.5 install —upgrade pip
```

## 형태소 분석기 Dev tool 설치
- mecab dependency 설치
```
sudo yum install -y python35u-devel.x86_64
sudo yum install -y python27-python-devel.x86_64
sudo yum install python-devel
sudo yum install -y curl
yum install gcc-c++ libstdc++ -y
sudo yum install -y centos-release-scl
sudo yum install -y devtoolset-4-gcc*
scl enable devtoolset-4 bash
```

- 형태소 분석기 설치
```
# install mecab-ko
cd /tmp
curl -LO https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz
tar zxfv mecab-0.996-ko-0.9.2.tar.gz
cd mecab-0.996-ko-0.9.2
./configure
make
make check
sudo make install
#install mecab-ko-dic
cd /tmp
curl -LO http://ftpmirror.gnu.org/automake/automake-1.11.tar.gz
tar -zxvf automake-1.11.tar.gz
cd automake-1.11
./configure
make
sudo make install
cd /tmp
curl -LO https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.0.3-20170922.tar.gz
tar -zxvf mecab-ko-dic-2.0.3-20170922.tar.gz
cd mecab-ko-dic-2.0.3-20170922
./autogen.sh
./configure
make
sudo sh -c 'echo "dicdir=/usr/local/lib/mecab/dic/mecab-ko-dic" > /usr/local/etc/mecabrc'
sudo make install
#install mecab-python
cd /tmp
git clone https://bitbucket.org/eunjeon/mecab-python-0.996.git
cd mecab-python-0.996
python setup.py build
python setup.py install
if hash "python3" &>/dev/null
then
  python3 setup.py build
  python3 setup.py install
fi
```

- python3 module list
```
bleach==1.5.0
certifi==2018.1.18
chardet==3.0.4
click==6.7
decorator==4.2.1
enum34==1.1.6
eventlet==0.22.1
Flask==0.12.2
greenlet==0.4.13
html5lib==0.9999999
idna==2.6
itsdangerous==0.24
Jinja2==2.10
JPype1-py3==0.5.5.2
konlpy==0.4.4
Markdown==2.6.11
MarkupSafe==1.0
networkx==2.1
numpy==1.14.1
pandas==0.22.0
protobuf==3.5.1
pymongo==3.6.0
PyMySQL==0.8.0
python-dateutil==2.6.1
python-engineio==2.0.2
python-socketio==1.8.4
pytz==2018.3
redis==2.10.6
requests==2.18.4
six==1.11.0
tensorflow==1.4.0
tensorflow-tensorboard==0.4.0
urllib3==1.22
Werkzeug==0.14.1
```

## nodejs 8 설치 (pm2 위해)
- 설치
```
sudo yum install -y gcc-c++ make
sudo curl -sL https://rpm.nodesource.com/setup_8.x | bash -
sudo yum install -y nodejs
```

- pm2 설치
```
sudo npm install -g pm2
```

## 영어 형태소 분석기 설치
1. nltk 모듈 설치
```
pip3.5 install nltk
```

2. python3.5 shell에서 다음과 같은 명령이 입력
```
import nltk
nltk.download('averaged_perception_tagger')
nltk.download('punkt')
```


# textrank Daemon

- pm2 list
- pm2 restart batch
- pm2 stop batch
- pm2 start batch
- pm2 kill (모든 프로세스 끄기)
- pm2 start batch.config.js (최초 시작 또는 pm2 kill 이후 시작)
