# tianchi_textclassify
天池大赛医学影像报告检测初赛26名代码分享（脱敏文本多标签分类）
## 介绍
数据：标签制作为one-hot形式，例如[3,4,6]就转为[0,0,0,1,1,0,1,…0]
模型：采用nezha_large   采用n-gram embedding（具体见代码）
训练：将训练集，测试集放一起，构建专属词表，进行MLM无监督训练，训练属于脱敏文字的预训练模型，然后再在训练集上微调
采用对抗训练（FGM） 10折交叉验证
## 运行
transformers==4.3.2 torch==1.7.1 
main_nezha_pretrain.py是MLM训练的代码，先运行这个得到预训练模型  下载：网盘
然后再在预训练模型上有监督训练（微调）运行main_nezha_kfold.py
数据中,pretrain.tsv是训练测试集的合并
