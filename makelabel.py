import codecs
num = 1
rootpath,labelfile,newlabelfile,labelidfile = None,None,None,None
if num == 1:
    rootpath = 'data/doneimage/'
    labelfile = 'data/truelabel.csv'
    newlabelfile = 'data/newlabel.csv'
    labelidfile = 'data/labelid.csv'
if num == 2:
    rootpath = 'chinesedata/original/'
    labelfile = 'chinesedata/truelabel.csv'
    newlabelfile = 'chinesedata/newlabel.csv'
    labelidfile = 'chinesedata/labelid.csv'

def convertlabel(name,label2iddict, length=7):
    '''
    将name字符串转换为相应的标签值.这里以10代表小数点，11代表补齐的内容
    name:字符串，即原始标注文件里的标注字符串
    return:一维np array。初步定义长度是6.会进行补齐。
    eg:只是示例，实际不一定字符和标签是这样对应的！
    name:"90.23" return:[9,0,10,2,3]
    name:"78"    return:[7,8,11,11,11]
    name:"78_1"  return:[7,8,11,11,11]
    '''
    #开始转换
    if '\n' in name:
        print(name)
    #resarray = [str(len(label2iddict)) for i in range(length)]
    lengthhere = len(name)
    resarray = [str(0) for i in range(lengthhere)]
    for i in range(lengthhere):
        resarray[i]=str(label2iddict[name[i]])
    return resarray

def makes2s():
    #由于无法一开始就确定需要多少种字符，只能是从标注文件里统计得到并将其对应到数字上
    #并且将这个东西写入到文件里去1.字符和数字2.图片和标注
    labelcontent = codecs.open(labelfile,'r','utf8').readlines()
    label = [i.split('<+++>')[1].replace('\n','') for i in labelcontent]
    label = [i for i in label if len(i)>0]
    labelset = set()
    maxlen = 0
    maxlenid = 0
    for i in label:
        labelset.update(list(i))
        if len(i)>maxlen:
            maxlenid = i
        maxlen = max(maxlen,len(i))
        
    print('numstr:',len(labelset))
    print('maxlen:',maxlen,maxlenid)
    label2iddict = {}
    id2labeldict = {}
    for idx,i in enumerate(labelset):
        label2iddict[i]=idx
        id2labeldict[idx]=i
    #将上面内容写入到文件保持
    out = codecs.open(labelidfile,'w','utf8')
    out.write('maxlen:'+str(maxlen)+'\n')
    for i in label2iddict:
        out.write(i+','+str(label2iddict[i])+'\n')
    out.close()
    #而后就是对所有原标签进行转换
    name2label = [i.split('<+++>') for i in labelcontent]
    name2label = [[i[0],i[1].replace('\n','')] for i in name2label]
    name2label = [[i[0],convertlabel(i[1],label2iddict,maxlen)] for i in name2label]
    newlabelout = codecs.open(newlabelfile,'w','utf8')
    for i in name2label:
        newlabelout.write(str(i[0])+'<+++>'+' '.join(i[1])+'\n')
    newlabelout.close()
makes2s()