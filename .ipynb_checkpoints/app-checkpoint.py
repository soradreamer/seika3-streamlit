from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from array import array
import os
from PIL import Image
import sys
import time
import json
import streamlit as st
from PIL import ImageDraw
from PIL import ImageFont
import io

with open("secret.json") as f:
    secret= json.load(f)
KEY = secret["KEY"]
ENDPOINT = secret["ENDPOINT"]
computervision_client = ComputerVisionClient(ENDPOINT, CognitiveServicesCredentials(KEY))
remote_image_features = [VisualFeatureTypes.categories,VisualFeatureTypes.brands,VisualFeatureTypes.adult,VisualFeatureTypes.color,VisualFeatureTypes.description,VisualFeatureTypes.faces,VisualFeatureTypes.image_type,VisualFeatureTypes.objects,VisualFeatureTypes.tags]

def get_tags(image_stream):
     # ストリームの位置を先頭に戻す
    image_stream.seek(0)
    tags_result = computervision_client.analyze_image_in_stream(image_stream , remote_image_features, language='ja')
    
    tags = tags_result.tags
    tags_name = []
    for tag in tags:
        tags_name.append(tag.name)
    return tags_name

def detect_objects(image_stream):
    # ストリームの位置を先頭に戻す
    image_stream.seek(0)
    object_result = computervision_client.analyze_image_in_stream(image_stream, remote_image_features, language='ja')

    return object_result.objects


st.title("物体検出アプリ")

upload_file = st.file_uploader("Chose an image...",type=["jpg","png"])

if upload_file is not None:
    # 画像を開く
    img = Image.open(upload_file)
    #img_path = f"img/{upload_file.name}"
    #img.save(img_path)
    #objects = detect_objects(img_path)

     # Create an in-memory stream for the image
    image_stream = io.BytesIO()
    img.save(image_stream, format=img.format)
    image_stream.seek(0)  # Rewind the stream to the beginning
    # Detect objects
    objects = detect_objects(image_stream)
    # Reset stream for tag extraction
    image_stream.seek(0)

    # デフォルトフォントを読み込む
    font = ImageFont.load_default(size=50)
    # 描画
    draw = ImageDraw.Draw(img)
    for object in objects:
        x = object.rectangle.x
        y = object.rectangle.y
        w = object.rectangle.w
        h = object.rectangle.h
        caption = object.object_property
        
        a,b,text_w,text_h=draw.textbbox((x,y),caption,font=font)
        draw.rectangle([(x,y),(x+w,y+h)],fill=None,outline="green",width=5)
        draw.rectangle([(x,y),(text_w,text_h)],fill="green",width=5)
        draw.text((x,y),caption,fill="white",font=font)
    st.image(img)


    tags_name = get_tags(image_stream)
    tags_name = ",".join(tags_name)
    st.markdown("**認識されたコンテンツタグ**")
    st.markdown(f">{tags_name}")