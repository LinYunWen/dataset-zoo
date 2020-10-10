---
tags: note, project, data, dataset
---
# Dataset Zoo

> 主要是希望蒐集各式資料集，建立一個完整的索引，方便以後做使用或是查詢

## Text
### chinese
- traditional
- simplified
    - [DuReader](https://ai.baidu.com/broad/subordinate?dataset=dureader)


### english
- US
    - [WordNet](https://wordnet.princeton.edu/)
    - [IMDB Reviews](http://ai.stanford.edu/~amaas/data/sentiment/)
    - [Twenty Newsgroups](https://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups)
    - [Sentiment140](http://help.sentiment140.com/for-students/)
    - [Yelp Reviews](https://www.yelp.com/dataset)
    - [The Wikipedia Corpus](http://nlp.cs.nyu.edu/wikipedia-data/)
    - [WebQuestions](https://github.com/brmson/dataset-factoid-webquestions)
    - [SimpleQA](https://github.com/davidgolub/SimpleQA/tree/master/datasets/SimpleQuestions)
    - [cMedQA](https://github.com/zhangsheng93/cMedQA)
    - [ODSQA](https://github.com/chiahsuan156/ODSQA)
- UK
    - [Machine Translation of Various Languages](http://statmt.org/wmt18/index.html)

### taiwanees
- [白話字台語文網站](http://ip194097.ntcu.edu.tw/taigu.asp)
- [Taiwanese Corpus語料](https://github.com/i3thuan5/tai5-uan5_gian5-gi2_hok8-bu7/wiki/Taiwanese-Corpus%E8%AA%9E%E6%96%99)



### Image
### object
- [ImageNet](http://image-net.org/)
    - The images are very diverse and often contain complex scenes with several objects (8.4 per image on average) and the dataset is annotated with image-level labels spanning thousands of classes.

- [COCO](http://cocodataset.org/#home)
- [YFCC100M](https://webscope.sandbox.yahoo.com/catalog.php?datatype=i&did=67&guccounter=1)
    - Yahoo Flickr Creative Commons 100M
- [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html)
- [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html)
    - The CIFAR-10 and CIFAR-100 are labeled subsets of the 80 million tiny images dataset
- [Fasion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
- [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)
- [COIL100](http://www1.cs.columbia.edu/CAVE/software/softlib/coil-100.php)
- [Visual Genome](https://visualgenome.org/)
- [Indoor Scene Recognition](http://web.mit.edu/torralba/www/indoor.html)



### Number
- [SVHN](http://ufldl.stanford.edu/housenumbers/)
    - The Street View House Numbers
- [MNIST](http://yann.lecun.com/exdb/mnist/)

### others
- [VisualQA](https://visualqa.org/)
- [Labelme](http://labelme.csail.mit.edu/Release3.0/browserTools/php/dataset.php)
- [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)



### face
- [MS-Celeb-1M](https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/)
    - Windows 提供 100 萬個名人的臉部照片
    - 这个数据集非常大，没有清洗过，噪声很大，很难用
    - [詳細介紹](https://megapixels.cc/datasets/msceleb/)

    > - [微軟刪除全球最大臉部辨識資料庫 MS-Celeb-1M，內含 10 萬個名人、1 千萬張照片](https://technews.tw/2019/06/13/ms-celeb-1m-was-deleted/)
    > - [備份 1](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo)、[備份 2](https://ibug.doc.ic.ac.uk/resources/lightweight-face-recognition-challenge-workshop/)
- [LFW](http://vis-www.cs.umass.edu/lfw/)
    - Labeled Faces in the Wild.
    - 来自1680的13000张人脸图，数据是从网上搜索来的
    - 基本都是正脸。这个数据集也是最简单的

- [CelebFaces](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
    - 总共包含10177个人的202599张图片，也是从搜索引擎上爬过来的
    - 噪声不算多，适合作为训练集
    - 同时这个数据对人脸有一些二元标签，比如是否微笑，是否戴帽子等。如果需要特定属性的人脸，也可以从中获取。40个属性如下（引用自[芯尚刃：CelebA数据集详细介绍及其属性提取源代码](https://zhuanlan.zhihu.com/p/35975956) ）：

- [CFP](https://link.zhihu.com/?target=http%3A//www.cfpw.io/cfp-dataset.zip)
    - 这个数据集由500个identity的约共7000张图片组成
    - 每个人有10张正面图像和4张侧面图像

- [VGG-Face](http://www.robots.ox.ac.uk/~vgg/data/vgg_face/)
    - 来自2622个人的2百万张图片。
    - 每个人大概要2000+图片，跟MS-Celeb-1M有很多重叠的地方（因为都是从搜索引擎来的）
    - 这个数据集经常作为训练模型的数据，噪声比较小，相对来说能训练出比较好的结果。

- [CASIA-WebFace](https://drive.google.com/file/d/1Of_EVz-yHV7QVWQGihYfvtny9Ne8qXVz/view)
    - 含10K个人的500K张图片，从IMBb网站上搜集来的。
    - 同时做了相似度聚类来去掉一部分噪声。
    - CAISA-WebFace的数据集源和IMDb-Face是一样的，不过因为数据清洗的原因，会比IMDb-Face少一些图片。噪声不算特别多，适合作为训练数据。

- [MegaFace](http://megaface.cs.washington.edu/dataset/download.html)
    - 672K人的4.7M张图片
    - 做过一些清洗，不过依然有噪声，不同人的图片可能混到了一起。
    - 这个数据集是由两个数据集组合而来：Facescrub和FGNet，所以如果你要使用多个数据集，注意有没有重合哦！

> reference:
> - [人脸识别常用数据集介绍（附下载链接）及常用评估指标](https://zhuanlan.zhihu.com/p/54811743)
> - [Dataset-Zoo](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo)


### mechanical



### Video
- [Youtube 8M](https://research.google.com/youtube8m/)
- [SPORTS-1M](https://cs.stanford.edu/people/karpathy/deepvideo/index.html)


### Audio
- [King-ASR](https://kingline.speechocean.com/category.php?id=120)
- [King-TTS](https://kingline.speechocean.com/category.php?id=69)
- [Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset)
- [Free Music Archive (FMA)](https://github.com/mdeff/fma)
- [MIR-Corpora](http://mirlab.org/dataSet/public/)
- [AudioSet](https://research.google.com/audioset/)
- [Ballroom](http://mtg.upf.edu/ismir2004/contest/tempoContest/node5.html)
- [Million Song Dataset](https://labrosa.ee.columbia.edu/millionsong/)
- [LibriSpeech](http://www.openslr.org/12/)
- [AVSpeech](https://looking-to-listen.github.io/avspeech/index.html)
- [VoxCeleb](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/)
- [AudioSet](https://research.google.com/audioset/)
- [Mozilla Common Voice](https://voice.mozilla.org/zh-TW/datasets)
- [freesound](https://freesound.org/search/?q=&f=created:[NOW-7DAY%20TO%20NOW]&s=num_downloads+desc&g=0)
- [Sound-Effects-Libraries](https://www.sound-ideas.com/Collection/4/Sound-Effects-Libraries-by-Category)

### phoneme
- [TIMIT](http://academictorrents.com/details/34e2b78745138186976cbc27939b1b34d18bd5b3)


### instruments
- [Free Wavs Samples](https://freewavesamples.com/)
- 
### Others
### QA Data



> reference:
> - [國網中心資料集平台](https://scidm.nchc.org.tw/)
> - [政府資料開放平臺](https://data.gov.tw/)
> - [25 Open Datasets for Deep Learning Every Data Scientist Must Work With](https://www.analyticsvidhya.com/blog/2018/03/comprehensive-collection-deep-learning-datasets/)
> - [Deep Learning - Datasets](http://deeplearning.net/datasets/)
> - [50 free Machine Learning Datasets: Image Datasets](https://blog.cambridgespark.com/50-free-machine-learning-datasets-image-datasets-241852b03b49)
> - [DataShare](https://datashare.is.ed.ac.uk/)



