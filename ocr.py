#!/usr/bin/env python
# coding: utf-8

# In[11]:

import cv2 as cv
from paddle.fluid.proto import framework_pb2
import paddlehub as hub
import PIL  
import os
import fitz
import sys
import paddle
import cv2
import paddlehub
#os.environ["CUDA_VISIBLE_DEVICES"] ="0"
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# In[12]:


def pdf_image(pdf_path,zoom_x,zoom_y,rotation_angle):
  # 打开PDF文件
    pdf = fitz.open(pdf_path)
    img_path=pdf_path.replace('pdf','')+'png'
  # 逐页读取PDF
    for pg in range(0, 1):
        page = pdf[pg]
    # 设置缩放和旋转系数
        trans = fitz.Matrix(zoom_x, zoom_y).preRotate(rotation_angle)
        pm = page.getPixmap(matrix=trans, alpha=False)
    # 开始写图像
        pm.writePNG(img_path)
        #pm.writePNG(imgPath)
        return img_path
    pdf.close()
# pdf_path ='D:/123.pdf'
# img_path ='D:/123.png'
# pdf_image(pdf_path,img_path,5,5,0)
def get_FileSize(filePath):
    fsize = os.path.getsize(filePath)
    fsize = fsize/float(1024*1024)
    return round(fsize,2)


# In[50]:


def ocr_fun(pdf_path, ocr_type='patent'):
    ocr = hub.Module(name="chinese_ocr_db_crnn_mobile")
    zoom_x = 1.33333333
    zoom_y = 1.33333333
    if(get_FileSize(pdf_path)>0.75):
        zoom_x = 0.5
        zoom_y = 0.5
    img_path=pdf_image(pdf_path,zoom_x,zoom_y,0)
    # img_path='./patent.png'
    results = ocr.recognize_text(images=[cv.imread(img_path)],
                                use_gpu=False,
                                output_dir='ocr_result',
                                visualization=False,
                                box_thresh=0.5,
                                text_thresh=0.5)
    dict = {'证书号':'','发明名称':'','发明人':'','专利号':'','专利申请日':'','专利权人':'','授权公告日':'','授权公告号':''}
    dict_1 = {'软件名称':'','著作权人':'','开发完成日期':'','权利取得方式':'','权利范围':'','登记号':''}
    if(ocr_type=='software'):
        for result in results:
            data = result['data']
            #save_path = result['save_path']
            for information in data:
                if(information['text'].startswith('软件名称')):
                    dict_1['软件名称']=information['text'].lstrip().lstrip('软件名称').replace('：','')
                elif(information['text'].startswith('著作权人')):
                    dict_1['著作权人']=information['text'].lstrip().lstrip('著作权人').replace('：',' ').lstrip()
                elif(information['text'].startswith('开发完成日期')):
                    dict_1['开发完成日期']=information['text'].lstrip().lstrip('开发完成日期').replace('：','')
                elif(information['text'].startswith('权利取得方式')):
                    dict_1['权利取得方式']=information['text'].lstrip().lstrip('权利取得方式').replace('：','')
                elif(information['text'].startswith('权利范围')):
                    dict_1['权利范围']=information['text'].lstrip().lstrip('权利范围').replace('：','')
                elif(information['text'].startswith('登记号')):
                    dict_1['登记号']=information['text'].lstrip().lstrip('登记号').replace('：','')
                elif(information['text'].startswith('号')):
                    dict_1['登记号']=information['text'].lstrip().lstrip('号').replace('：','')
                elif(information['text'].startswith('记号')):
                    dict_1['登记号']=information['text'].lstrip().lstrip('记号').replace('：','')
#                 print('text: ', information['text'], '\nconfidence: ', information['confidence'], '\ntext_box_position: ', information['text_box_position'])
    elif(ocr_type=='patent'):
        for result in results:
            data = result['data']
            #save_path = result['save_path']
            for information in data:
                if(information['text'].startswith('证书号')):
                    dict['证书号']=information['text'].lstrip().lstrip('证书号').replace('第','').replace('号','')
                elif(information['text'].startswith('发明名称')):
                    dict['发明名称']=information['text'].lstrip().lstrip('发明名称').replace('：','')
                elif(information['text'].startswith('发明人')):
                    dict['发明人']=information['text'].lstrip().lstrip('发明人').replace('；',' ').replace('：',' ').lstrip()
                elif(information['text'].startswith('专利号')):
                    dict['专利号']=information['text'].lstrip().lstrip('专利号').replace('：','')
                elif(information['text'].startswith('号')):
                    dict['专利号']=information['text'].lstrip().lstrip('号').replace('：','')
                elif(information['text'].startswith('利号')):
                    dict['专利号']=information['text'].lstrip().lstrip('利号').replace('：','')
                elif(information['text'].startswith('专利申请日')):
                    dict['专利申请日']=information['text'].lstrip().lstrip('专利申请日').replace('；','').replace('：','')
                elif(information['text'].startswith('专利权人：')):
                    dict['专利权人']=information['text'].lstrip('专利权人').replace('；',' ').replace('：',' ').lstrip()
                elif(information['text'].startswith('授权公告日')):
                    dict['授权公告日']=information['text'].lstrip().lstrip('授权公告日').replace('；','').replace('：','')
                elif(information['text'].startswith('授权公告号')):
                    dict['授权公告号']=information['text'].lstrip().lstrip('授权公告号').replace('：','')
    os.remove(img_path)
    if(ocr_type=='patent'):
        return dict
    else:
        return dict_1
#                 print('text: ', information['text'], '\nconfidence: ', information['confidence'], '\ntext_box_position: ', information['text_box_position'])            


# In[60]:

if __name__ == "__main__":
    dic=ocr_fun(sys.argv[1].replace('-',''),sys.argv[2].replace('-',''))
    print(dic)


# In[62]:


#for keys,contens in dic.items():
     #print(keys+": "+contens)

