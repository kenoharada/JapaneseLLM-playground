## Setup

```
ssh -A abci
git clone git@github.com:kenoharada/JapaneseLLM-playground.git
qrsh -g $GROUP_ID -l rt_AF=1 -l h_rt=12:00:00
cd JapaneseLLM-playground
module load singularitypro
nohup singularity build --nv -f llm.sif llm.def &
cp .env.example .env
singularity exec --bind $PWD:$PWD --pwd $PWD --nv llm.sif /bin/bash
python examples/rinna.py
# ユーザー: 日本のおすすめの観光地を教えてください。<NL>システム: どの地域の観光地が知りたいですか？<NL>ユーザー: 東京都文京区本郷周辺の観光地を教えてください。<NL>システム:
# 東京都文京区本郷周辺の観光地には、日本庭園、東京大学、お茶の水女子大学、根津美術館、根津神社、東洋文庫、本郷キャンパス、湯島天満宮、椿山荘などがあります。</s>
python examples/cyber.py
```

```
git clone git@github.com:kenoharada/JapaneseLLM-playground.git
cd JapaneseLLM-playground
docker build -t llm .
cp .env.example .env
docker run --env-file ./.env --mount type=bind,src=$PWD,dst=$PWD --workdir $PWD -it --ipc host --gpus all --rm --name `whoami`_llm llm /bin/bash
python examples/rinna.py
python examples/cyber.py
```
