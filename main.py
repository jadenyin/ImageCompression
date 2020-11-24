import numpy as np
import imageio
from PIL import Image

def PCA_com(data,alpha):
    #去中心化
    data_mean=np.mean(data,axis=0)
    dec_data=data-data_mean
    #计算去中心化后数据的协方差矩阵
    cov=np.cov(dec_data)
    #对协方差矩阵进行特征值分解
    #计算特征值特征向量
    eig_val,eig_vec=np.linalg.eig(cov)
    #按特征值从大到小顺序排列
    idx=np.argsort(-eig_val)
    eig_val=eig_val[idx]
    eig_vec=eig_vec[:,idx]
    #选取主成分个数k,阈值alpha=0.95
    SUM=sum(eig_val)
    compo=0
    for i in range(len(eig_val)):
        compo+=eig_val[i]
        if (compo/SUM)>=alpha:
            break
    k=i
    #得到前k个特征值所对应的特征向量组成矩阵
    W=eig_vec[:,0:k]
    #降维后的数据,用于计算空间节省和压缩率
    new_data=np.dot(dec_data,W)
    #重构图片,要加上去中心化减去的值
    rec_data=np.dot(new_data,np.transpose(W))+data_mean
    #取绝对值并转化为uint8类型(为了重构图像）
    rec_data=np.uint8(np.absolute(rec_data))
    return rec_data,new_data

a=imageio.imread("airplane/airplane31.tif")
a_np=np.array(a)
a_r=a_np[:,:,0]#R
a_g=a_np[:,:,1]#G
a_b=a_np[:,:,2]#B

a_r_recon,a_r_com=PCA_com(a_r,0.90)
a_g_recon,a_g_com=PCA_com(a_g,0.90)
a_b_recon,a_b_com=PCA_com(a_b,0.90)
recon_color_img=np.dstack((a_r_recon,a_g_recon,a_b_recon))
com_pic_size=2*(a_r_com.size+a_g_com.size+a_b_com.size)
#计算空间节省和压缩率
saved_space=(a.size-com_pic_size)*8/(8*1024)
com_rate=100*(com_pic_size/a.size)
#计算重构误差
recon_err=np.sum((recon_color_img-a)**2)/a.size
#重构图像并显示
img=Image.fromarray(recon_color_img)
img.show()
img.save('01.jpg')
print("节省空间: %f kb" % saved_space)
print("压缩率: %f%%" %com_rate)
print("重构误差: %f" %recon_err)
