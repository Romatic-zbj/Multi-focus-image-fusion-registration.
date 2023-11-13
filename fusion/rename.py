import os
path="./Fusion"
filelist=os.listdir(path)

for inner_name in filelist:
    oldname=path+"/"+inner_name
    no_ext_file=os.path.splitext(inner_name)[0]#无后缀的文件名
    # no_ext_file=no_ext_file+"_fliter_"+'{}.png'.format(i)
    # i=i+1
    prefix,_ = no_ext_file.split("_")
    # print(prefix+'\n')
    # print(num+'\n')

    no_ext_file = prefix+"_A.png"
    newname=path+"/"+no_ext_file
    os.rename(oldname,newname)
