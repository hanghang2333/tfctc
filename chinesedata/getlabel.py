#主要是生成图片的完整路径名称和对应的label这样一个文件
#为什么要完整路径?担心图片重名，如果保证不重名的话就不用带路径都放到一个路径即可
import os
import codecs
rootpath = 'original/'
dirlist = os.listdir(rootpath)
outfile = 'truelabel.csv'
writefile = codecs.open(outfile,'w','utf8')
for onedir in dirlist:
    for img in os.listdir(os.path.join(rootpath,onedir)):
        label = onedir
        path = os.path.join(onedir,img)
        writefile.write(path+'<+++>'+label+'\n')