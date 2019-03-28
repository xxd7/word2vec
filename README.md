# word2vec
一个基于word2vec的用于文本分类的卷积网络
使用方法为data = pd.read_csv('labeledTrainData.tsv',sep='\t')本地读入训练集
目前的训练集是在kaggle上获取的英语文本
print("Iter " + str(step) + ", Loss= " +  "{}".format(loss) + ",Accuracy= " +  "{}".format(acc))输出丢失率和准确率
