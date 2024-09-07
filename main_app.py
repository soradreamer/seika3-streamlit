
from PIL import Image
import streamlit as st
import torch
import torch
import torch.nn as nn
import pandas as pd
from torchvision import  transforms,models
import plotly.graph_objects as go
import os
from PIL import Image, ImageDraw

st.set_page_config(
     page_title='犬種判別アプリ🐾',
     layout="wide",
     initial_sidebar_state="expanded",
    )
col1, col2 = st.columns([1,2])  # 幅1:2の2列のコンテナを用意する

#　最初の読み込み
def main():

    # トリミング画像を初期化
    if 'cropped_image' not in st.session_state:
        st.session_state.cropped_image = None
    col1.title('犬種判別アプリ🐾')
    col1.write('こちらに画像をアップロードしてください。(jpgもしくはpng)')

    # アップロードオブジェクト用意
    upload_file = col1.file_uploader("jpgもしくはpng",type=["jpg","png"])
    
    if upload_file is not None:
        image = Image.open(upload_file)
        image_tr,crop_box =triming_pre(image)
        if col1.button("トリミング"):
            st.session_state.cropped_image=crop_image(image_tr, crop_box)
            col1.image(st.session_state.cropped_image, caption='トリミングされた画像', use_column_width=True)
        if col1.button("予測"):
            if st.session_state.cropped_image is not None:  # トリミング後の画像が存在する場合
                model_val_and_result(st.session_state.cropped_image)
                col1.image(st.session_state.cropped_image, caption='予測に使用した画像', use_column_width=True)
            else:
                model_val_and_result(image)
                col1.image(image, caption='予測に使用した画像', use_column_width=True)

# トリミングの準備、サイドバー表示
def triming_pre(image):
    # トリミングサイズの入力
    st.sidebar.header("トリミング設定")

    # スライドバーで幅と高さの設定
    max_width = image.width
    max_height = image.height
    crop_width = st.sidebar.slider("幅", min_value=1, max_value=max_width, value=max_width)
    crop_height = st.sidebar.slider("高さ", min_value=1, max_value=max_height, value=max_height)
    
    # スライドバーで位置の設定
    x = st.sidebar.slider("X位置", min_value=0, max_value=max_width - crop_width+1, value=0)
    y = st.sidebar.slider("Y位置", min_value=0, max_value=max_height - crop_height+1, value=0)

    # トリミングボックスの計算
    crop_box = (x, y, x + crop_width, y + crop_height)

    # トリミングボックスを描画
    image_with_box = draw_crop_box(image.copy(), crop_box)
    
    col1.image(image_with_box, caption='トリミングボックスを描画した画像', use_column_width=True)

    return image,crop_box

# 赤枠で新たな画像を作る
def crop_image(image, crop_box):
    return image.crop(crop_box)

# スライドバーによって赤枠の描画を変更
def draw_crop_box(image, crop_box):
    # 画像に描画を追加
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = crop_box
    # トリミングボックスの線を描画
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    return image

# 画像の推論、結果表示
def model_val_and_result(image):
    # 犬種_日本語
    class_names_jp=[
    'チワワ',
    'ジャパニーズ・スパニエル',
    'マルチーズ',
    'ペキニーズ',
    'シーズー',
    'ブレンハイム・スパニエル',
    'パピヨン',
    'トイ・テリア',
    'ローデシアン・リッジバック',
    'アフガン・ハウンド',
    'バセット',
    'ビーグル',
    'ブラッドハウンド',
    'ブルーテック',
    'ブラック・アンド・タン・クーンハウンド',
    'ウォーカーハウンド',
    'イングリッシュ・フォックスハウンド',
    'レッドボーン',
    'ボルゾイ',
    'アイリッシュ・ウルフハウンド',
    'イタリアン・グレイハウンド',
    'ウィペット',
    'イビザン・ハウンド',
    'ノルウェージアン・エルクハウンド',
    'オッターハウンド',
    'サルーキ',
    'スコティッシュ・ディアハウンド',
    'ワイマラナー',
    'スタッフォードシャー・ブル・テリア',
    'アメリカン・スタッフォードシャー・テリア',
    'ベドリントン・テリア',
    'ボーダー・テリア',
    'ケリー・ブルー・テリア',
    'アイリッシュ・テリア',
    'ノーフォーク・テリア',
    'ノリッジ・テリア',
    'ヨークシャー・テリア',
    'ワイヤー・ヘアード・フォックス・テリア',
    'レイクランド・テリア',
    'セーリーハム・テリア',
    'エアデール',
    'カーン・テリア',
    'オーストラリアン・テリア',
    'ダンディ・ディンモント',
    'ボストン・ブル',
    'ミニチュア・シュナウザー',
    'ジャイアント・シュナウザー',
    'スタンダード・シュナウザー',
    'スコッチ・テリア',
    'チベタン・テリア',
    'シルキー・テリア',
    'ソフト・コーテッド・ウィートン・テリア',
    'ウェスト・ハイランド・ホワイト・テリア',
    'ラサ',
    'フラット・コーテッド・レトリーバー',
    'カーリー・コーテッド・レトリーバー',
    'ゴールデン・レトリーバー',
    'ラブラドール・レトリーバー',
    'チェサピーク・ベイ・レトリーバー',
    'ジャーマン・ショート・ヘアード・ポインター',
    'ヴィズラ',
    'イングリッシュ・セッター',
    'アイリッシュ・セッター',
    'ゴードン・セッター',
    'ブリタニー・スパニエル',
    'クランバー',
    'イングリッシュ・スプリンガー',
    'ウェルシュ・スプリンガー・スパニエル',
    'コッカー・スパニエル',
    'サセックス・スパニエル',
    'アイリッシュ・ウォーター・スパニエル',
    'クヴァサ',
    'シェパード・ドッグ',
    'グルーネンダール',
    'マリノア',
    'ブリアード',
    'ケルピー',
    'コモンドール',
    'オールド・イングリッシュ・シープドッグ',
    'シェトランド・シープドッグ',
    'コリー',
    'ボーダー・コリー',
    'ブービエ・デ・フランドル',
    'ロットワイラー',
    'ジャーマン・シェパード',
    'ドーベルマン',
    'ミニチュア・ピンシャー',
    'グレーター・スイス・マウンテン・ドッグ',
    'バーンシャー・マウンテン・ドッグ',
    'アッペンツェラー',
    'エントレブッカー',
    'ボクサー',
    'ブル・マスティフ',
    'チベタン・マスティフ',
    'フレンチ・ブルドッグ',
    'グレート・デーン',
    'セント・バーナード',
    'エスキモー・ドッグ',
    'マラミュート',
    'シベリアン・ハスキー',
    'アフェンピンシャー',
    'バセンジ',
    'パグ',
    'レオンベルガー',
    'ニューファンドランド',
    'グレート・ピレニーズ',
    'サモエド',
    'ポメラニアン',
    'チャウ',
    'ケーシュホンド',
    'ブラバンソン・グリフォン',
    'ペンブローク',
    'カーディガン',
    'トイ・プードル',
    'ミニチュア・プードル',
    'スタンダード・プードル',
    'メキシカン・ヘアレス・ドッグ',
    'ディンゴ',
    'ドール',
    'アフリカン・ハンティング・ドッグ'
    ]
    # 犬種_英語
    class_names_en = [
    'Chihuahua',
    'Japanese_spaniel',
    'Maltese_dog',
    'Pekinese',
    'Shih-Tzu',
    'Blenheim_spaniel',
    'papillon',
    'toy_terrier',
    'Rhodesian_ridgeback',
    'Afghan_hound',
    'basset',
    'beagle',
    'bloodhound',
    'bluetick',
    'black-and-tan_coonhound',
    'Walker_hound',
    'English_foxhound',
    'redbone',
    'borzoi',
    'Irish_wolfhound',
    'Italian_greyhound',
    'whippet',
    'Ibizan_hound',
    'Norwegian_elkhound',
    'otterhound',
    'Saluki',
    'Scottish_deerhound',
    'Weimaraner',
    'Staffordshire_bullterrier',
    'American_Staffordshire_terrier',
    'Bedlington_terrier',
    'Border_terrier',
    'Kerry_blue_terrier',
    'Irish_terrier',
    'Norfolk_terrier',
    'Norwich_terrier',
    'Yorkshire_terrier',
    'wire-haired_fox_terrier',
    'Lakeland_terrier',
    'Sealyham_terrier',
    'Airedale',
    'cairn',
    'Australian_terrier',
    'Dandie_Dinmont',
    'Boston_bull',
    'miniature_schnauzer',
    'giant_schnauzer',
    'standard_schnauzer',
    'Scotch_terrier',
    'Tibetan_terrier',
    'silky_terrier',
    'soft-coated_wheaten_terrier',
    'West_Highland_white_terrier',
    'Lhasa',
    'flat-coated_retriever',
    'curly-coated_retriever',
    'golden_retriever',
    'Labrador_retriever',
    'Chesapeake_Bay_retriever',
    'German_short-haired_pointer',
    'vizsla',
    'English_setter',
    'Irish_setter',
    'Gordon_setter',
    'Brittany_spaniel',
    'clumber',
    'English_springer',
    'Welsh_springer_spaniel',
    'cocker_spaniel',
    'Sussex_spaniel',
    'Irish_water_spaniel',
    'kuvasz',
    'schipperke',
    'groenendael',
    'malinois',
    'briard',
    'kelpie',
    'komondor',
    'Old_English_sheepdog',
    'Shetland_sheepdog',
    'collie',
    'Border_collie',
    'Bouvier_des_Flandres',
    'Rottweiler',
    'German_shepherd',
    'Doberman',
    'miniature_pinscher',
    'Greater_Swiss_Mountain_dog',
    'Bernese_mountain_dog',
    'Appenzeller',
    'EntleBucher',
    'boxer',
    'bull_mastiff',
    'Tibetan_mastiff',
    'French_bulldog',
    'Great_Dane',
    'Saint_Bernard',
    'Eskimo_dog',
    'malamute',
    'Siberian_husky',
    'affenpinscher',
    'basenji',
    'pug',
    'Leonberg',
    'Newfoundland',
    'Great_Pyrenees',
    'Samoyed',
    'Pomeranian',
    'chow',
    'keeshond',
    'Brabancon_griffon',
    'Pembroke',
    'Cardigan',
    'toy_poodle',
    'miniature_poodle',
    'standard_poodle',
    'Mexican_hairless',
    'dingo',
    'dhole',
    'African_hunting_dog'
    ]
    
    num_classes = 120 # クラス数

    # 使用するモデル読み込み
    model = models.resnet50(weights='DEFAULT')

    # 最終層の調整
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # 学習したモデルを読み込み
    model.load_state_dict(torch.load('model_03_2_resnet.pth', map_location="cpu"))
    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),  # モデルの期待するサイズにリサイズ
        transforms.ToTensor(),  # テンソルに変換
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正規化
    ])
    input_tensor = preprocess(image)
    # バッチサイズの次元を追加
    input_batch = input_tensor.unsqueeze(0)  

     # モデルの推論
    with torch.no_grad():
        output = model(input_batch)

    # ソフトマックスを適用して確率を計算
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(output[0])
    predicted_label = class_names_jp[predicted_class.item()]
    # 上位10つのクラスを取得
    top10_probabilities, top10_classes = torch.topk(probabilities, 10, dim=1)
    
    # 結果を格納するための空のデータフレームを作成、for文内で格納
    df_result = pd.DataFrame(columns=['Rank', '種類', '確率(%)'])
    for i in range(10):
        class_index = top10_classes[0][i].item()
        class_label_jp = class_names_jp[class_index]
        class_probability = top10_probabilities[0][i].item() * 100  # 確率をパーセンテージに変換
        add_dict= {'Rank':i+1, '種類':class_label_jp, '確率(%)':class_probability}
        df_add  =pd.DataFrame(add_dict,index=[0])
        df_result = pd.concat([df_result,df_add],ignore_index=True)
    
    #
    # 結果表示
    #
    col2.title(f'結果:{df_result["種類"][0]}🐾')
    fig = go.Figure(data=go.Pie(labels=df_result["種類"], values=df_result["確率(%)"]))

    with col2.container():
        col21,col22 = col2.columns(2)
        col22.dataframe(df_result.set_index("Rank"))
        col21.plotly_chart(fig)
        
        with col2.container():
            row1 = st.columns(5)
            row2 = st.columns(5)
            i=0
            for col in row1 + row2:
                tile = col.container(height=280)
                class_index = top10_classes[0][i].item()
                class_label_en = class_names_en[class_index]
                class_label_jp = class_names_jp[class_index]
                sample_img_path = f"./sample/{class_label_en}_sample.jpg"
                i=i+1
                if os.path.exists(sample_img_path):
                # 画像を開く
                    sample_image = Image.open(sample_img_path)
                    # Streamlitで画像を表示
                    tile.image(sample_image, caption=f"{class_label_jp}", use_column_width=True)
                    tile.link_button("詳細", f"https://www.akc.org/dog-breeds/{class_label_en.replace('_','-')}/")

if __name__ == "__main__":
    main()