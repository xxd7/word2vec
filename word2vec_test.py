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
         #默认window=5，考虑上下5个单词来进行预测，计算词向量
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
#对每一条评论应用lambda函数
# In[6]:
del data_t['review']
del word_vectors
# In[7]:
import gc
gc.collect()
# In[8]:
data_t = data_t[data_t['vec'].apply(lambda x:len(x)>0)]
# In[9]:
data_t.sentiment.value_counts()
# In[10]:
maxlength = max([len(x) for x in data_t.vec])
maxlength
# In[11]:
sum(data_t.vec.apply(len)>300)
# In[12]:
def pad(x):
    if len(x)>300:
        x1 = x[:300]
    else:
        x1 = np.zeros((300,50))
        x1[:len(x)] = x
    return x1
# In[13]:
data_t['vec'] = data_t.vec.apply(pad)
# In[14]:
np.shape(data_t.sentiment.values)
# RNN
# In[15]:
import tensorflow as tf
# In[16]:
learning_rate = 0.002
batch_size = 100
n_input = 50
n_steps = 300
n_hidden = 300
n_classes = 2
# In[17]:
x = tf.placeholder(tf.float32, [None, n_steps,n_input])
y = tf.placeholder(tf.int64, [None])
keep_prob = tf.placeholder("float")
# In[18]:
def length(shuru):
    return tf.reduce_sum(tf.sign(tf.reduce_max(tf.abs(shuru),reduction_indices=2)),reduction_indices=1)
# In[19]:
cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(n_hidden),
                output_keep_prob = keep_prob)
# In[20]:
output, _ = tf.nn.dynamic_rnn(
            cell,
            x,
            dtype=tf.float32,
            sequence_length = length(x)
        )
# In[21]:
output.get_shape()
# In[22]:
index = tf.range(0,batch_size)*n_steps + (tf.cast(length(x),tf.int32) - 1)
flat = tf.reshape(output,[-1,int(output.get_shape()[2])])
last = tf.gather(flat,index)
# In[23]:
weight = tf.Variable(tf.truncated_normal((n_hidden, n_classes), stddev=0.001))
bias = tf.Variable(tf.constant(0.1, shape=[n_classes]))
com_out = tf.matmul(last, weight) + bias
prediction = tf.nn.softmax(com_out)
# In[24]:
cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = com_out))
# In[25]:
optimizer = tf.train.AdamOptimizer(learning_rate)
grads = optimizer.compute_gradients(cross_entropy)
for i, (g, v) in enumerate(grads):
    if g is not None:
        grads[i] = (tf.clip_by_norm(g, 5), v)  # clip gradients
train_op = optimizer.apply_gradients(grads)
# In[26]:
correct_pred = tf.equal(tf.argmax(prediction,1), y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# In[27]:
def generatebatch(X,Y,n_examples, batch_size):
    for batch_i in range(n_examples // batch_size):
        start = batch_i*batch_size
        end = start + batch_size
        batch_xs = X[start:end]
        batch_ys = Y[start:end]
        yield batch_xs, batch_ys
        # 生成每一个batch
# In[28]:
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()
# In[29]:
for step in range(10):
    index = np.random.permutation(int(len(data_t.vec.values)))
    for batch_x,batch_y in generatebatch(data_t.vec.values[index],data_t.sentiment.values[index],len(data_t.vec.values),batch_size):
        batch_x = np.concatenate(batch_x).reshape(batch_size,300,50)
        batch_x.astype(np.float32)
        sess.run(train_op, feed_dict={x: batch_x, y: batch_y,keep_prob: 0.5})
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
