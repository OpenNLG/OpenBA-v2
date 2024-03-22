import math
fin=open("/data/LEO/doremi/try.txt","r",encoding="utf-8")
sizes=[]
names=[]
lines=fin.readlines()
min_size=900
for line in lines:
    split_list=line.split()
    size,name=split_list[0],split_list[1]
    size=float(size[:-1])
    if size<min_size:
        min_size=size
    sizes.append(size)
    names.append(name)
goal=16.6
min=goal
min_re=""
counts=0
check=[0]*len(sizes)
used_names=[
    # "langchao-055_spancorr.bin",
    # "langchao-056_spancorr.bin",
    # "langchao-059_spancorr.bin",
    # "langchao-082_spancorr.bin",
    # "langchao-057_spancorr.bin",
    # "langchao-060_spancorr.bin",
    # "langchao-061_spancorr.bin",
    # "langchao-071_spancorr.bin",
    # "langchao-073_spancorr.bin",
    # "langchao-074_spancorr.bin",
    # "langchao-081_spancorr.bin",
    # "langchao-083_spancorr.bin",
    # "langchao-062_spancorr.bin",
    # "langchao-063_spancorr.bin",
    # "langchao-072_spancorr.bin",
    "/data/pretrained_data/full/200B_binlingual_split/langchao-064_spancorr",
    "/data/pretrained_data/full/200B_binlingual_split/langchao-065_spancorr",
    "/data/pretrained_data/full/200B_binlingual_split/langchao-066_spancorr",
    "/data/pretrained_data/full/200B_binlingual_split/langchao-067_spancorr",
    "/data/pretrained_data/full/200B_binlingual_split/langchao-068_spancorr",
    "/data/pretrained_data/full/200B_binlingual_split/langchao-069_spancorr",
    "/data/pretrained_data/full/200B_binlingual_split/langchao-070_spancorr",
    "/data/pretrained_data/full/200B_binlingual_split/langchao-075_spancorr",
    "/data/pretrained_data/full/200B_binlingual_split/langchao-076_spancorr",
    "/data/pretrained_data/full/200B_binlingual_split/langchao-077_spancorr",
    "/data/pretrained_data/full/200B_binlingual_split/langchao-078_spancorr",
    "/data/pretrained_data/full/200B_binlingual_split/langchao-079_spancorr",
    "/data/pretrained_data/full/200B_binlingual_split/langchao-080_spancorr",
    "/data/pretrained_data/full/200B_binlingual_split/langchao-084_spancorr",
    "/data/pretrained_data/full/200B_binlingual_split/langchao-085_spancorr",
    "/data/pretrained_data/full/200B_binlingual_split/langchao-086_spancorr",
    "/data/pretrained_data/full/200B_binlingual_split/langchao-087_spancorr",
    "/data/pretrained_data/full/200B_binlingual_split/langchao-088_spancorr",
    "/data/pretrained_data/full/200B_binlingual_split/langchao-091_spancorr",
    "/data/pretrained_data/full/200B_binlingual_split/langchao-092_spancorr",
    "/data/pretrained_data/full/200B_binlingual_split/langchao-095_spancorr",
    "/data/pretrained_data/full/200B_binlingual_split/langchao-096_spancorr",
    "/data/pretrained_data/full/200B_binlingual_split/langchao-099_spancorr",
    "/data/pretrained_data/full/200B_binlingual_split/langchao-101_spancorr",
    "/data/pretrained_data/full/200B_binlingual_split/langchao-102_spancorr",
    "/data/pretrained_data/full/200B_binlingual_split/langchao-105_spancorr",
    "/data/pretrained_data/full/200B_binlingual_split/langchao-106_spancorr",
]
temp=0
for i in range(len(names)):
    if ("/data/pretrained_data/full/200B_binlingual_split/"+names[i][:-4]) in used_names:
        temp+=sizes[i]
print(temp)
# for i in range(len(names)):
#     if names[i] in used_names:
#         check[i]=1
# print(check)
# def dfs(k,n,sum,index,check,re=""):
#     global min
#     global min_re
#     global counts
#     if k==n:
#         counts+=1
#         if abs(sum-goal)<min:
#             min=abs(sum-goal)
#             min_re=re
#     else:
#         if sum>goal:
#             return
#         for i in range(index+1,len(check)):
#             if check[i]==0:
#                 check[i]=1
#                 dfs(k+1,n,sum+sizes[i],i,check,re+"\n/data/pretrained_data/full/200B_binlingual_split/"+names[i])
#                 check[i]=0
# for i in range(1,math.ceil(goal/min_size)+1):
#     dfs(0,i,0,-1,check)
#     print(i)
#     # print(i,min,min_re)
# print(min,min_re,counts)

