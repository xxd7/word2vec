# coding: utf-8
# In[1]:
import pandas as pd
import re #正则化，把标点符号，特殊符号都丢掉
import numpy as np
from gensim.models import word2vec#训练一个word2vec数据集
# In[2]:
data = pd.read_csv('labeledTrainData.tsv',sep='\t')#pd.read_csv读入我们的label数据
del data['id']#训练集里面的id太大啦，就不要了
# In[3]:
pat = re.compile(r'[A-Za-z]+')
# In[4]:
def zhengli_word(w):
    str_list = pat.findall(w)
    str_list = [x.lower() for x in str_list]
    if len(str_list)>=300:
        str_list = str_list[:300]
    else:
        for i in range(300 - len(str_list)):
            str_list.append('')
    return str_list
# In[5]:
data['review'] = data.review.apply(zhengli_word)
# In[6]:
data['sentiment'] = data.sentiment.apply(lambda x :[0,1] if x == 1 else [1,0])
# In[7]:
y = np.concatenate(data.sentiment.values).reshape(len(data),-1)
# In[8]:
x_text = np.concatenate(data.review.values).reshape(len(data),-1)
# In[9]:
x_text
# In[ ]:
data_t = pd.read_csv('testData.tsv',sep='\t')
def test_l(x):
    x1 = int(x.split('_')[-1])
    if x1 >= 7:
        return 1
    else:
        return 0
data_t['sentiment'] = data_t.id.apply(test_l)
# In[3]:
if not os.path.exists('mymodel'):#查看当前目录下是否存在mymodel,同时也避免以后要多次训练这个model
    if not os.path.exists('imdb_text'):#判断是否存在这个语料库
        data_un = pd.read_csv('unlabeledTrainData.tsv',header=0, delimiter="\t",quoting=3 )#读入我们预设的unlabel数据（目前来说比较小）
        pat = re.compile(r'[A-Za-z]+')
        #提取全部单词（如果是提取汉字的话就....）如果想进一步考虑到标点符号对语义的影响，
        #不妨加上[!@#$%^&*]等等垃（yue）圾（pao）短信中常见的一些符号
        with open('imdb_text','a',encoding = 'utf-8') as f:
            for rev in data_un.review:#对语料库里面的评论进行迭代
                str_list = pat.findall(rev)#先提取出所有单词
              # str_list = [x.lower() for x in str_list]
              #小写化所有单词，但是实际应用中大小写也会影响语义，看情况是否选择最小化
                string = ' '.join(str_list)
                f.write(string + '\n')
                #上述操作以后我们就能得到一个写满处理后的string的文件啦
            del data_un
    sentences =word2vec.Text8Corpus("imdb_text")  # 加载语料
    model =word2vec.Word2Vec(sentences, size=50)  #训练skip-gram模型，词向量长度设置50（不知道会不会有点大），
         #默认window=5，考虑上下5个单词来进行预测，计算词向量，如果一个词出现的次数少于5次那就默认这是一个生僻词，忽略掉，实际操作中这个数字可以改变
    model.save('mymodel')
    #然后保存下来可以用于下一次训练啦~
else:
    model = word2vec.Word2Vec.load('mymodel')
    #如果已经存在之前训练好的model那就直接导入
    #In[3]主要用于训练我们的词向量
# In[4]:
word_vectors = model.wv #单词与向量的对应都在wv里面，把训练好的model复制过来
del model
# In[5]:
data_t['vec'] = data_t.review.apply(lambda x :[word_vectors[w] for w in x.split() if w in word_vectors])
#对每一条评论应用lambda函数转变成词向量，把单词分成一个很长的list，但是如果w不在word_vectors里面的话那就不考虑
#至此训练集构建完成，下一步把训练集转变成词向量
# In[6]:
del data_t['review']
del word_vectors
# In[7]:
import gc
gc.collect()
#清空内存，防止过度占用资源，如果电脑配置高那可以忽略这一块
# In[8]:
data_t = data_t[data_t['vec'].apply(lambda x:len(x)>0)]
#对向量化的评论做一个长度评论，防止某些评论的单词都不在向量空间里面
# In[9]:
data_t.sentiment.value_counts()
#统计不同类型的信息对应的数量（有的信息可能没有被训练成词向量）
# In[10]:
maxlength = max([len(x) for x in data_t.vec])
maxlength
#计算信息的单词最大长度
# In[11]:
sum(data_t.vec.apply(len)>300)
#计算这些信息的单词长度达到300的个数，300根据实际需要作出更改，帮助作出填充/缩减单词长度的决定
#我们的单词都是由一个长度为50的向量来表示
# In[12]:
def pad(x):
    if len(x)>300:
        x1 = x[:300]
    else:
        x1 = np.zeros((300,50))
        #每条信息都是300*50的矩阵
        x1[:len(x)] = x
    return x1
#把每个单词填充成长度为300的单词，每个单词由长度为50的向量来表示
#预处理完毕，把每条信息都填充成同一个长度
# In[13]:
data_t['vec'] = data_t.vec.apply(pad)
# In[14]:
np.shape(data_t.sentiment.values)
# RNN
#定义RNN网络用于文本分类
# In[15]:
import tensorflow as tf
# In[16]:
learning_rate = 0.002
#学习速率
batch_size = 100
#RNN网络一次处理的信息数，电脑性能好数值建议加大
n_input = 50
#RNN是一个沿着时间轴展开的网络
#第一个维度是批次
#第二个维度是step，表示在长度为300的信息中处理到了哪一步，实际上是信息长度
#第三个维度是input,表示输入的向量长度
n_steps = 300
#我们的信息已经都填充到长度300了
n_hidden = 300
#定义的网络有300个隐藏单元
n_classes = 2
#我们判定的类型有两种，即一条信息是正面还是负面
# In[17]:
x = tf.placeholder(tf.float32, [None, n_steps,n_input])
#input的绝对值不是0说明这个不是填充过的数据
y = tf.placeholder(tf.int64, [None])
#训练目标，1或者0
keep_prob = tf.placeholder("float")
防止过拟合
# In[18]:
def length(shuru):
    return tf.reduce_sum(tf.sign(tf.reduce_max(tf.abs(shuru),reduction_indices=2)),reduction_indices=1)
#计算出实际长度，并反馈给RNN网络，提高计算效率
#tf.sign把数据分成0/1 在300*50中的每个向量中计算出这个向量为0/1.然后对这300个计算结果进行相加，以此计算出信息长度
# In[19]:
cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(n_hidden),
                #kernel_initializer = tf.truncated_normal_initializer(stddev=0.0001),
                #bias_initializer = tf.truncated_normal_initializer(stddev=0.0001)
                output_keep_prob = keep_prob)
                #定义循环神经网络的cell，每次训练之后输出就是300，学习能力基本和n_hidden成正比，
                #但是有过拟合的风险
                #给循环神经网络添加drop
                #对输出的值进行随机丢弃
# In[20]:
output, _ = tf.nn.dynamic_rnn(
            cell,
            x,
            dtype=tf.float32,
            sequence_length = length(x)
        )
        #循环神经网络，动态计算序列长度，经过处理以后就免去计算填充数据
        #RNN的另外一个输出并不需要
        #全零时间步的output是如何输出的
# In[21]:
output.get_shape()
#输出
#TensorShape([Dimension(None),Dimension(300),Dimension(300)])
#每一个隐藏单元对应一个输出
# In[22]:
index = tf.range(0,batch_size)*n_steps + (tf.cast(length(x),tf.int32) - 1)
取出每条评论最后的output
flat = tf.reshape(output,[-1,int(output.get_shape()[2])])
#-1：动态计算
last = tf.gather(flat,index)
# In[23]:
weight = tf.Variable(tf.truncated_normal((n_hidden, n_classes), stddev=0.001))
bias = tf.Variable(tf.constant(0.1, shape=[n_classes]))
com_out = tf.matmul(last, weight) + bias
prediction = tf.nn.softmax(com_out)
#对com_out进行softmax分类
# In[24]:
cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = com_out))
#没有使用独热编码，直接使用tf.nn
#计算交叉熵的时候会在内部计算softmax，把交叉熵变成一个标量值
# In[25]:
optimizer = tf.train.AdamOptimizer(learning_rate)
grads = optimizer.compute_gradients(cross_entropy)
for i, (g, v) in enumerate(grads):
    if g is not None:
        grads[i] = (tf.clip_by_norm(g, 5), v)  # clip gradients
train_op = optimizer.apply_gradients(grads)
#梯度裁减防止梯度爆炸
# In[26]:
correct_pred = tf.equal(tf.argmax(prediction,1), y)
#计算正确值
#prediction得出两个预测值，哪个的概率最大就预测哪个
#对prediction求最大索引，求出预测结果，然后与真实的结果y做equal再求均值
#如果是True那就是1，False则为1
#至此把整个网络都定义好了
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# In[27]:
def generatebatch(X,Y,n_examples, batch_size):
    for batch_i in range(n_examples // batch_size):
        start = batch_i*batch_size
        end = start + batch_size
        #产生批次，从数据中取出索引
        batch_xs = X[start:end]
        batch_ys = Y[start:end]
        yield batch_xs, batch_ys
        # 生成每一个batch
        #batch不能过小，否则会影响权值
        #X为训练数据 Y是目标数据 n_examples为数据总量
# In[28]:
sess = tf.Session()
#初始化session
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()
#保存检查点，用于test数据
# In[29]:
for step in range(10):
    index = np.random.permutation(int(len(data_t.vec.values)))
    for batch_x,batch_y in generatebatch(data_t.vec.values[index],data_t.sentiment.values[index],len(data_t.vec.values),batch_size):
        batch_x = np.concatenate(batch_x).reshape(batch_size,300,50)
        batch_x.astype(np.float32)
        sess.run(train_op, feed_dict={x: batch_x, y: batch_y,keep_prob: 0.5})
        #if ii%100==0
        #  print(sess.run([accuracy,cross_entry],feed_dict={x:batch_x,y:batch_y,keep_prob:1}))
        #ii+=1
    acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y,keep_prob: 1})
    loss = sess.run(cross_entropy, feed_dict={x: batch_x, y: batch_y,keep_prob: 1})
    saver.save(sess,'./lesson0',global_step = step)
    print("Iter " + str(step) + ", Minibatch Loss= " +  "{}".format(loss) + ", Training Accuracy= " +  "{}".format(acc))
print("Optimization Finished!")
# In[28]:
ckpt = tf.train.get_checkpoint_state(os.path.dirname('__file__'))
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.restore(sess,ckpt.model_checkpoint_path)
# In[29]:
accu = []
for batch_x,batch_y in generatebatch(data_t.vec.values,data_t.sentiment.values,len(data_t.vec.values),batch_size):
    batch_x = np.concatenate(batch_x).reshape(batch_size,300,50)
    batch_x.astype(np.float64)
    acc = sess.run(accuracy,feed_dict={x: batch_x, y: batch_y,keep_prob: 1})
    accu.append(acc)
print(sess.run(tf.reduce_mean(accu)))
